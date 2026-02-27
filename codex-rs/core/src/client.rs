use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::OnceLock;
use std::time::SystemTime;

use crate::api_bridge::CoreAuthProvider;
use crate::api_bridge::auth_provider_from_auth;
use crate::api_bridge::map_api_error;
use crate::auth::UnauthorizedRecovery;
use chrono::DateTime;
use chrono::Utc;
use codex_api::AggregateStreamExt;
use codex_api::ChatClient as ApiChatClient;
use codex_api::CompactClient as ApiCompactClient;
use codex_api::CompactionInput as ApiCompactionInput;
use codex_api::MemoriesClient as ApiMemoriesClient;
use codex_api::MemorySummarizeInput as ApiMemorySummarizeInput;
use codex_api::MemorySummarizeOutput as ApiMemorySummarizeOutput;
use codex_api::RawMemory as ApiRawMemory;
use codex_api::RequestTelemetry;
use codex_api::ReqwestTransport;
use codex_api::ResponseAppendWsRequest;
use codex_api::ResponseCreateWsRequest;
use codex_api::ResponsesApiRequest;
use codex_api::ResponsesClient as ApiResponsesClient;
use codex_api::ResponsesOptions as ApiResponsesOptions;
use codex_api::ResponsesWebsocketClient as ApiWebSocketResponsesClient;
use codex_api::ResponsesWebsocketConnection as ApiWebSocketConnection;
use codex_api::SseTelemetry;
use codex_api::TransportError;
use codex_api::WebsocketTelemetry;
use codex_api::build_conversation_headers;
use codex_api::common::Reasoning;
use codex_api::common::ResponsesWsRequest;
use codex_api::create_text_param_for_request;
use codex_api::error::ApiError;
use codex_api::requests::responses::Compression;
use codex_otel::OtelManager;

use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::config_types::Verbosity as VerbosityConfig;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::protocol::SessionSource;
use eventsource_stream::Event;
use eventsource_stream::EventStreamError;
use futures::StreamExt;
use http::HeaderMap as ApiHeaderMap;
use http::HeaderValue;
use http::StatusCode as HttpStatusCode;
use reqwest::StatusCode;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::oneshot::error::TryRecvError;
use tokio_tungstenite::tungstenite::Error;
use tokio_tungstenite::tungstenite::Message;
use tracing::trace;
use tracing::warn;

use crate::AuthManager;
use crate::CodexAuth;
use crate::auth::DEFAULT_OAUTH_NAMESPACE;
use crate::auth::RefreshTokenError;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::client_common::ResponseStream;
use crate::config::Config;
use crate::default_client::build_reqwest_client;
use crate::error::CodexErr;
use crate::error::RefreshTokenFailedReason;
use crate::error::Result;
use crate::flags::CODEX_RS_SSE_FIXTURE;
use crate::model_provider_info::ModelProviderInfo;
use crate::model_provider_info::WireApi;
use crate::tools::spec::create_tools_json_for_responses_api;

pub const OPENAI_BETA_HEADER: &str = "OpenAI-Beta";
pub const OPENAI_BETA_RESPONSES_WEBSOCKETS: &str = "responses_websockets=2026-02-04";
pub const X_CODEX_TURN_STATE_HEADER: &str = "x-codex-turn-state";
const DEFAULT_RATE_LIMIT_COOLDOWN_MS: u64 = 30_000;
const DEFAULT_AUTH_FAILURE_COOLDOWN_MS: u64 = 5 * 60_000;
const DEFAULT_PAYMENT_REQUIRED_COOLDOWN_MS: u64 = 60 * 60_000;
const DEFAULT_NETWORK_RETRY_ATTEMPTS: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponsesWebsocketVersion {
    V1,
    V2,
}

pub fn ws_version_from_features(config: &Config) -> Option<ResponsesWebsocketVersion> {
    match (
        config
            .features
            .enabled(crate::features::Feature::ResponsesWebsockets),
        config
            .features
            .enabled(crate::features::Feature::ResponsesWebsocketsV2),
    ) {
        (_, true) => Some(ResponsesWebsocketVersion::V2),
        (true, false) => Some(ResponsesWebsocketVersion::V1),
        (false, false) => None,
    }
}

/// Session-scoped state shared by all [`ModelClient`] clones.
///
/// This is intentionally kept minimal so `ModelClient` does not need to hold a full `Config`. Most
/// configuration is per turn and is passed explicitly to streaming/unary methods.
#[derive(Debug)]
struct ModelClientState {
    auth_manager: Option<Arc<AuthManager>>,
    conversation_id: ThreadId,
    provider: ModelProviderInfo,
    session_source: SessionSource,
    model_verbosity: Option<VerbosityConfig>,
    responses_websocket_version: Option<ResponsesWebsocketVersion>,
    enable_request_compression: bool,
    include_timing_metrics: bool,
    beta_features_header: Option<String>,
    disable_websockets: AtomicBool,
    cached_websocket_session: StdMutex<WebsocketSession>,
}

/// Resolved API client setup for a single request attempt.
///
/// Keeping this as a single bundle ensures prewarm and normal request paths
/// share the same auth/provider setup flow.
struct CurrentClientSetup {
    auth: Option<CodexAuth>,
    api_provider: codex_api::Provider,
    api_auth: CoreAuthProvider,
}

/// A session-scoped client for model-provider API calls.
///
/// This holds configuration and state that should be shared across turns within a Codex session
/// (auth, provider selection, conversation id, feature-gated request behavior, and transport
/// fallback state).
///
/// WebSocket fallback is session-scoped: once a turn activates the HTTP fallback, subsequent turns
/// will also use HTTP for the remainder of the session.
///
/// Turn-scoped settings (model selection, reasoning controls, telemetry context, and turn
/// metadata) are passed explicitly to the relevant methods to keep turn lifetime visible at the
/// call site.
#[derive(Debug, Clone)]
pub struct ModelClient {
    state: Arc<ModelClientState>,
}

/// A turn-scoped streaming session created from a [`ModelClient`].
///
/// The session establishes a Responses WebSocket connection lazily and reuses it across multiple
/// requests within the turn. It also caches per-turn state:
///
/// - The last full request, so subsequent calls can use `response.append` only when the current
///   request is an incremental extension of the previous one.
/// - The `x-codex-turn-state` sticky-routing token, which must be replayed for all requests within
///   the same turn.
///
/// Create a fresh `ModelClientSession` for each Codex turn. Reusing it across turns would replay
/// the previous turn's sticky-routing token into the next turn, which violates the client/server
/// contract and can cause routing bugs.
pub struct ModelClientSession {
    client: ModelClient,
    websocket_session: WebsocketSession,
    /// Turn state for sticky routing.
    ///
    /// This is an `OnceLock` that stores the turn state value received from the server
    /// on turn start via the `x-codex-turn-state` response header. Once set, this value
    /// should be sent back to the server in the `x-codex-turn-state` request header for
    /// all subsequent requests within the same turn to maintain sticky routing.
    ///
    /// This is a contract between the client and server: we receive it at turn start,
    /// keep sending it unchanged between turn requests (e.g., for retries, incremental
    /// appends, or continuation requests), and must not send it between different turns.
    turn_state: Arc<OnceLock<String>>,
}

#[derive(Debug, Clone)]
struct LastResponse {
    response_id: String,
    items_added: Vec<ResponseItem>,
    can_append: bool,
}

#[derive(Debug, Default)]
struct WebsocketSession {
    connection: Option<ApiWebSocketConnection>,
    last_request: Option<ResponsesApiRequest>,
    last_response_rx: Option<oneshot::Receiver<LastResponse>>,
}

enum WebsocketStreamOutcome {
    Stream(ResponseStream),
    FallbackToHttp,
}

impl ModelClient {
    #[allow(clippy::too_many_arguments)]
    /// Creates a new session-scoped `ModelClient`.
    ///
    /// All arguments are expected to be stable for the lifetime of a Codex session. Per-turn values
    /// are passed to [`ModelClientSession::stream`] (and other turn-scoped methods) explicitly.
    pub fn new(
        auth_manager: Option<Arc<AuthManager>>,
        conversation_id: ThreadId,
        provider: ModelProviderInfo,
        session_source: SessionSource,
        model_verbosity: Option<VerbosityConfig>,
        responses_websocket_version: Option<ResponsesWebsocketVersion>,
        enable_request_compression: bool,
        include_timing_metrics: bool,
        beta_features_header: Option<String>,
    ) -> Self {
        Self {
            state: Arc::new(ModelClientState {
                auth_manager,
                conversation_id,
                provider,
                session_source,
                model_verbosity,
                responses_websocket_version,
                enable_request_compression,
                include_timing_metrics,
                beta_features_header,
                disable_websockets: AtomicBool::new(false),
                cached_websocket_session: StdMutex::new(WebsocketSession::default()),
            }),
        }
    }

    /// Creates a fresh turn-scoped streaming session.
    ///
    /// This constructor does not perform network I/O itself; the session opens a websocket lazily
    /// when the first stream request is issued.
    pub fn new_session(&self) -> ModelClientSession {
        ModelClientSession {
            client: self.clone(),
            websocket_session: self.take_cached_websocket_session(),
            turn_state: Arc::new(OnceLock::new()),
        }
    }

    fn take_cached_websocket_session(&self) -> WebsocketSession {
        let mut cached_websocket_session = self
            .state
            .cached_websocket_session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        std::mem::take(&mut *cached_websocket_session)
    }

    fn store_cached_websocket_session(&self, websocket_session: WebsocketSession) {
        *self
            .state
            .cached_websocket_session
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = websocket_session;
    }

    /// Compacts the current conversation history using the Compact endpoint.
    ///
    /// This is a unary call (no streaming) that returns a new list of
    /// `ResponseItem`s representing the compacted transcript.
    ///
    /// The model selection and telemetry context are passed explicitly to keep `ModelClient`
    /// session-scoped.
    pub async fn compact_conversation_history(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        otel_manager: &OtelManager,
    ) -> Result<Vec<ResponseItem>> {
        if prompt.input.is_empty() {
            return Ok(Vec::new());
        }
        let auth_manager = self.state.auth_manager.clone();
        let instructions = prompt.base_instructions.text.clone();
        let mut extra_headers = ApiHeaderMap::new();
        if let SessionSource::SubAgent(sub) = &self.state.session_source {
            let subagent = match sub {
                crate::protocol::SubAgentSource::Review => "review".to_string(),
                crate::protocol::SubAgentSource::Compact => "compact".to_string(),
                crate::protocol::SubAgentSource::ThreadSpawn { .. } => "collab_spawn".to_string(),
                crate::protocol::SubAgentSource::Other(label) => label.clone(),
            };
            if let Ok(val) = HeaderValue::from_str(&subagent) {
                extra_headers.insert("x-openai-subagent", val);
            }
        }

        if let Some(manager) = auth_manager.clone()
            && let Some(rotation) = OAuthRotationPlan::try_new(
                manager,
                &self.state.provider,
                &self.state.config,
                DEFAULT_OAUTH_NAMESPACE,
            )
            .await
        {
            let api_provider = self
                .state
                .provider
                .to_api_provider(Some(AuthMode::ChatGPT))?;
            let request_telemetry = self.build_request_telemetry();
            let response = rotation
                .execute(|_, auth| async {
                    let api_auth = auth_provider_from_auth(Some(auth), &self.state.provider)
                        .map_err(|err| ApiError::Stream(err.to_string()))?;
                    let transport = ReqwestTransport::new(build_reqwest_client());
                    let client = ApiCompactClient::new(transport, api_provider.clone(), api_auth)
                        .with_telemetry(Some(request_telemetry.clone()));
                    let payload = ApiCompactionInput {
                        model: &self.state.model_info.slug,
                        input: &prompt.input,
                        instructions: &instructions,
                    };
                    client.compact_input(&payload, extra_headers.clone()).await
                })
                .await
                .map_err(map_api_error)?;
            return Ok(response);
        }

        let auth = match auth_manager.as_ref() {
            Some(manager) => manager.auth().await,
            None => None,
        };
        let api_provider = self
            .state
            .provider
            .to_api_provider(auth.as_ref().map(|a| a.mode))?;
        let api_auth = auth_provider_from_auth(auth.clone(), &self.state.provider)?;
        let transport = ReqwestTransport::new(build_reqwest_client());
        let request_telemetry = Self::build_request_telemetry(otel_manager);
        let client =
            ApiCompactClient::new(transport, client_setup.api_provider, client_setup.api_auth)
                .with_telemetry(Some(request_telemetry));

        let payload = ApiCompactionInput {
            model: &model_info.slug,
            input: &prompt.input,
            instructions: &instructions,
        };

        client
            .compact_input(&payload, extra_headers)
            .await
    }

    /// Builds websocket handshake headers for both prewarm and turn-time reconnect.
    ///
    /// Callers should pass the current turn-state lock when available so sticky-routing state is
    /// replayed on reconnect within the same turn.
    fn build_websocket_headers(
        &self,
        ws_version: ResponsesWebsocketVersion,
        turn_state: Option<&Arc<OnceLock<String>>>,
        turn_metadata_header: Option<&str>,
    ) -> ApiHeaderMap {
        let turn_metadata_header = parse_turn_metadata_header(turn_metadata_header);
        let mut headers = build_responses_headers(
            self.state.beta_features_header.as_deref(),
            turn_state,
            turn_metadata_header.as_ref(),
        );
        headers.extend(build_conversation_headers(Some(
            self.state.conversation_id.to_string(),
        )));
        let responses_websockets_beta_header = match ws_version {
            ResponsesWebsocketVersion::V2 => RESPONSES_WEBSOCKETS_V2_BETA_HEADER_VALUE,
            ResponsesWebsocketVersion::V1 => OPENAI_BETA_RESPONSES_WEBSOCKETS,
        };
        headers.insert(
            OPENAI_BETA_HEADER,
            HeaderValue::from_static(responses_websockets_beta_header),
        );
        if self.state.include_timing_metrics {
            headers.insert(
                X_RESPONSESAPI_INCLUDE_TIMING_METRICS_HEADER,
                HeaderValue::from_static("true"),
            );
        }
        headers
    }
}

impl Drop for ModelClientSession {
    fn drop(&mut self) {
        let websocket_session = std::mem::take(&mut self.websocket_session);
        self.client
            .store_cached_websocket_session(websocket_session);
    }
}

impl ModelClientSession {
    fn activate_http_fallback(&self, websocket_enabled: bool) -> bool {
        websocket_enabled
            && !self
                .client
                .state
                .disable_websockets
                .swap(true, Ordering::Relaxed)
    }

    fn build_responses_request(
        &self,
        provider: &codex_api::Provider,
        prompt: &Prompt,
        compression: Compression,
    ) -> ApiResponsesOptions {
        build_responses_options_from_state(&self.state, &self.turn_state, prompt, compression)
    }

    fn prepare_websocket_request(
        &self,
        api_prompt: &ApiPrompt,
        options: &ApiResponsesOptions,
    ) -> ResponsesWsRequest {
        prepare_websocket_request_with_last_items(
            &self.state.model_info.slug,
            api_prompt,
            options,
            &self.websocket_last_items,
        )
    }

    fn prepare_websocket_request(
        &mut self,
        payload: ResponseCreateWsRequest,
        request: &ResponsesApiRequest,
        ws_version: ResponsesWebsocketVersion,
    ) -> ResponsesWsRequest {
        let Some(last_response) = self.get_last_response() else {
            return ResponsesWsRequest::ResponseCreate(payload);
        };
        let allow_empty_delta = matches!(ws_version, ResponsesWebsocketVersion::V2);
        let Some(append_items) =
            self.get_incremental_items(request, Some(&last_response), allow_empty_delta)
        else {
            return ResponsesWsRequest::ResponseCreate(payload);
        };

        match ws_version {
            ResponsesWebsocketVersion::V2 => {
                if last_response.response_id.is_empty() {
                    trace!("incremental request failed, no previous response id");
                    return ResponsesWsRequest::ResponseCreate(payload);
                }

                ResponsesWsRequest::ResponseCreate(ResponseCreateWsRequest {
                    previous_response_id: Some(last_response.response_id),
                    input: append_items,
                    ..payload
                })
            }
            ResponsesWebsocketVersion::V1 => {
                if !last_response.can_append {
                    trace!("incremental request failed, can't append");
                    return ResponsesWsRequest::ResponseCreate(payload);
                }
                ResponsesWsRequest::ResponseAppend(ResponseAppendWsRequest {
                    input: append_items,
                    client_metadata: payload.client_metadata,
                })
            }
        }
    }

    /// Opportunistically preconnects a websocket for this turn-scoped client session.
    ///
    /// This performs only connection setup; it never sends prompt payloads.
    pub async fn preconnect_websocket(
        &mut self,
        otel_manager: &OtelManager,
        model_info: &ModelInfo,
    ) -> std::result::Result<(), ApiError> {
        let Some(ws_version) = self.client.active_ws_version(model_info) else {
            return Ok(());
        };
        if self.websocket_session.connection.is_some() {
            return Ok(());
        }

        let client_setup = self.client.current_client_setup().await.map_err(|err| {
            ApiError::Stream(format!(
                "failed to build websocket prewarm client setup: {err}"
            ))
        })?;

        let connection = self
            .client
            .connect_websocket(
                otel_manager,
                client_setup.api_provider,
                client_setup.api_auth,
                ws_version,
                Some(Arc::clone(&self.turn_state)),
                None,
            )
            .await?;
        self.websocket_session.connection = Some(connection);
        Ok(())
    }
    /// Returns a websocket connection for this turn.
    async fn websocket_connection(
        &mut self,
        otel_manager: &OtelManager,
        api_provider: codex_api::Provider,
        api_auth: CoreAuthProvider,
        ws_version: ResponsesWebsocketVersion,
        turn_metadata_header: Option<&str>,
        options: &ApiResponsesOptions,
    ) -> std::result::Result<&ApiWebSocketConnection, ApiError> {
        let needs_new = match self.websocket_session.connection.as_ref() {
            Some(conn) => conn.is_closed().await,
            None => true,
        };

        if needs_new {
            self.websocket_session.last_request = None;
            self.websocket_session.last_response_rx = None;
            let turn_state = options
                .turn_state
                .clone()
                .unwrap_or_else(|| Arc::clone(&self.turn_state));
            let new_conn = self
                .client
                .connect_websocket(
                    otel_manager,
                    api_provider,
                    api_auth,
                    ws_version,
                    Some(turn_state),
                    turn_metadata_header,
                )
                .await?;
            self.websocket_session.connection = Some(new_conn);
        }

        self.websocket_session
            .connection
            .as_ref()
            .ok_or(ApiError::Stream(
                "websocket connection is unavailable".to_string(),
            ))
    }

    fn responses_request_compression(&self, auth: Option<&crate::auth::CodexAuth>) -> Compression {
        responses_request_compression_from_state(&self.state, auth)
    }

    /// Streams a turn via the OpenAI Chat Completions API.
    ///
    /// This path is only used when the provider is configured with
    /// `WireApi::Chat`; it does not support `output_schema` today.
    async fn stream_chat_completions(&self, prompt: &Prompt) -> Result<ApiResponseStream> {
        if prompt.output_schema.is_some() {
            return Err(CodexErr::UnsupportedOperation(
                "output_schema is not supported for Chat Completions API".to_string(),
            ));
        }

        let auth_manager = self.state.auth_manager.clone();
        let instructions = prompt.base_instructions.text.clone();
        let tools_json = create_tools_json_for_chat_completions_api(&prompt.tools)?;
        let api_prompt = build_api_prompt(prompt, instructions, tools_json);
        let conversation_id = self.state.conversation_id.to_string();
        let session_source = self.state.session_source.clone();

        if let Some(manager) = auth_manager.clone()
            && let Some(rotation) = OAuthRotationPlan::try_new(
                manager,
                &self.state.provider,
                &self.state.config,
                DEFAULT_OAUTH_NAMESPACE,
            )
            .await
        {
            let api_provider = self
                .state
                .provider
                .to_api_provider(Some(AuthMode::ChatGPT))?;
            let api_stream = rotation
                .execute(|_, auth| async {
                    let api_auth = auth_provider_from_auth(Some(auth), &self.state.provider)
                        .map_err(|err| ApiError::Stream(err.to_string()))?;
                    let transport = ReqwestTransport::new(build_reqwest_client());
                    let (request_telemetry, sse_telemetry) = self.build_streaming_telemetry();
                    let client = ApiChatClient::new(transport, api_provider.clone(), api_auth)
                        .with_telemetry(Some(request_telemetry), Some(sse_telemetry));

                    client
                        .stream_prompt(
                            &self.state.model_info.slug,
                            &api_prompt,
                            Some(conversation_id.clone()),
                            Some(session_source.clone()),
                        )
                        .await
                })
                .await
                .map_err(map_api_error)?;
            return Ok(api_stream);
        }

        let mut auth_recovery = auth_manager
            .as_ref()
            .map(super::auth::AuthManager::unauthorized_recovery);
        loop {
            let auth = match auth_manager.as_ref() {
                Some(manager) => manager.auth().await,
                None => None,
            };
            let api_provider = self
                .state
                .provider
                .to_api_provider(auth.as_ref().map(|a| a.mode))?;
            let api_auth = auth_provider_from_auth(auth.clone(), &self.state.provider)?;
            let transport = ReqwestTransport::new(build_reqwest_client());
            let (request_telemetry, sse_telemetry) = self.build_streaming_telemetry();
            let client = ApiChatClient::new(transport, api_provider, api_auth)
                .with_telemetry(Some(request_telemetry), Some(sse_telemetry));

            let stream_result = client
                .stream_prompt(
                    &self.state.model_info.slug,
                    &api_prompt,
                    Some(conversation_id.clone()),
                    Some(session_source.clone()),
                )
                .await;

            match stream_result {
                Ok(stream) => return Ok(stream),
                Err(ApiError::Transport(TransportError::Http { status, .. }))
                    if status == StatusCode::UNAUTHORIZED =>
                {
                    handle_unauthorized(status, &mut auth_recovery).await?;
                    continue;
                }
                Err(err) => return Err(map_api_error(err)),
            }
        }
    }

    /// Streams a turn via the OpenAI Responses API.
    ///
    /// Handles SSE fixtures, reasoning summaries, verbosity, and the
    /// `text` controls used for output schemas.
    #[allow(clippy::too_many_arguments)]
    async fn stream_responses_api(
        &self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        otel_manager: &OtelManager,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        turn_metadata_header: Option<&str>,
    ) -> Result<ResponseStream> {
        if let Some(path) = &*CODEX_RS_SSE_FIXTURE {
            warn!(path, "Streaming from fixture");
            let stream = codex_api::stream_from_fixture(
                path,
                self.client.state.provider.stream_idle_timeout(),
            )
            .map_err(map_api_error)?;
            let (stream, _last_request_rx) = map_response_stream(stream, otel_manager.clone());
            return Ok(stream);
        }

        let auth_manager = self.state.auth_manager.clone();
        let api_prompt = self.build_responses_request(prompt)?;

        if let Some(manager) = auth_manager.clone()
            && let Some(rotation) = OAuthRotationPlan::try_new(
                manager,
                &self.state.provider,
                &self.state.config,
                DEFAULT_OAUTH_NAMESPACE,
            )
            .await
        {
            let api_provider = self
                .state
                .provider
                .to_api_provider(Some(AuthMode::ChatGPT))?;
            let api_stream = rotation
                .execute(|_, auth| async {
                    let transport = ReqwestTransport::new(build_reqwest_client());
                    let (request_telemetry, sse_telemetry) = self.build_streaming_telemetry();
                    let compression =
                        responses_request_compression_from_state(&self.state, Some(&auth));
                    let api_auth = auth_provider_from_auth(Some(auth), &self.state.provider)
                        .map_err(|err| ApiError::Stream(err.to_string()))?;
                    let options = self.build_responses_options(prompt, compression);

                    let client = ApiResponsesClient::new(transport, api_provider.clone(), api_auth)
                        .with_telemetry(Some(request_telemetry), Some(sse_telemetry));

                    client
                        .stream_prompt(&self.state.model_info.slug, &api_prompt, options)
                        .await
                })
                .await
                .map_err(map_api_error)?;
            return Ok(map_response_stream(
                api_stream,
                self.state.otel_manager.clone(),
            ));
        }

        let mut auth_recovery = auth_manager
            .as_ref()
            .map(super::auth::AuthManager::unauthorized_recovery);
        loop {
            let client_setup = self.client.current_client_setup().await?;
            let transport = ReqwestTransport::new(build_reqwest_client());
            let (request_telemetry, sse_telemetry) = Self::build_streaming_telemetry(otel_manager);
            let compression = self.responses_request_compression(client_setup.auth.as_ref());
            let options = self.build_responses_options(turn_metadata_header, compression);

            let request = self.build_responses_request(
                &client_setup.api_provider,
                prompt,
                model_info,
                effort,
                summary,
            )?;
            let client = ApiResponsesClient::new(
                transport,
                client_setup.api_provider,
                client_setup.api_auth,
            )
            .with_telemetry(Some(request_telemetry), Some(sse_telemetry));
            let stream_result = client.stream_request(request, options).await;

            match stream_result {
                Ok(stream) => {
                    let (stream, _) = map_response_stream(stream, otel_manager.clone());
                    return Ok(stream);
                }
                Err(ApiError::Transport(
                    unauthorized_transport @ TransportError::Http { status, .. },
                )) if status == StatusCode::UNAUTHORIZED => {
                    handle_unauthorized(unauthorized_transport, &mut auth_recovery).await?;
                    continue;
                }
                Err(err) => return Err(map_api_error(err)),
            }
        }
    }

    /// Streams a turn via the Responses API over WebSocket transport.
    #[allow(clippy::too_many_arguments)]
    async fn stream_responses_websocket(
        &mut self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        ws_version: ResponsesWebsocketVersion,
        otel_manager: &OtelManager,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        turn_metadata_header: Option<&str>,
        warmup: bool,
    ) -> Result<WebsocketStreamOutcome> {
        let auth_manager = self.client.state.auth_manager.clone();

        if let Some(manager) = auth_manager.clone()
            && let Some(rotation) = OAuthRotationPlan::try_new(
                manager,
                &self.state.provider,
                &self.state.config,
                DEFAULT_OAUTH_NAMESPACE,
            )
            .await
        {
            let api_provider = self
                .state
                .provider
                .to_api_provider(Some(AuthMode::ChatGPT))?;
            let state = Arc::clone(&self.state);
            let turn_state = Arc::clone(&self.turn_state);
            let api_prompt_for_request = api_prompt.clone();
            let last_items = self.websocket_last_items.clone();
            let connection_cell = Arc::new(tokio::sync::Mutex::new(self.connection.take()));
            let stream_result = rotation
                .execute({
                    let connection_cell = Arc::clone(&connection_cell);
                    let state = Arc::clone(&state);
                    let turn_state = Arc::clone(&turn_state);
                    let api_prompt = api_prompt_for_request.clone();
                    let last_items = last_items.clone();
                    let api_provider = api_provider.clone();
                    move |_, auth| {
                        let connection_cell = Arc::clone(&connection_cell);
                        let state = Arc::clone(&state);
                        let turn_state = Arc::clone(&turn_state);
                        let api_prompt = api_prompt.clone();
                        let last_items = last_items.clone();
                        let api_provider = api_provider.clone();
                        async move {
                            let compression =
                                responses_request_compression_from_state(&state, Some(&auth));
                            let api_auth = auth_provider_from_auth(Some(auth), &state.provider)
                                .map_err(|err| ApiError::Stream(err.to_string()))?;
                            let options = build_responses_options_from_state(
                                &state,
                                &turn_state,
                                prompt,
                                compression,
                            );
                            let request = prepare_websocket_request_with_last_items(
                                &state.model_info.slug,
                                &api_prompt,
                                &options,
                                &last_items,
                            );

                            let mut headers = options.extra_headers.clone();
                            headers.extend(build_conversation_headers(
                                options.conversation_id.clone(),
                            ));
                            let new_conn =
                                ApiWebSocketResponsesClient::new(api_provider.clone(), api_auth)
                                    .connect(headers, options.turn_state.clone())
                                    .await?;
                            let stream = new_conn.stream_request(request).await?;
                            {
                                let mut guard = connection_cell.lock().await;
                                *guard = Some(new_conn);
                            }
                            Ok(stream)
                        }
                    }
                })
                .await
                .map_err(map_api_error)?;
            self.websocket_last_items = api_prompt.input.clone();
            self.connection = connection_cell.lock().await.take();
            return Ok(map_response_stream(
                stream_result,
                self.state.otel_manager.clone(),
            ));
        }

        let mut auth_recovery = auth_manager
            .as_ref()
            .map(super::auth::AuthManager::unauthorized_recovery);
        loop {
            let client_setup = self.client.current_client_setup().await?;
            let compression = self.responses_request_compression(client_setup.auth.as_ref());

            let options = self.build_responses_options(turn_metadata_header, compression);
            let request = self.build_responses_request(
                &client_setup.api_provider,
                prompt,
                model_info,
                effort,
                summary,
            )?;
            let mut ws_payload = ResponseCreateWsRequest {
                client_metadata: build_ws_client_metadata(turn_metadata_header),
                ..ResponseCreateWsRequest::from(&request)
            };
            if warmup {
                ws_payload.generate = Some(false);
            }

            match self
                .websocket_connection(
                    otel_manager,
                    client_setup.api_provider,
                    client_setup.api_auth,
                    ws_version,
                    turn_metadata_header,
                    &options,
                )
                .await
            {
                Ok(_) => {}
                Err(ApiError::Transport(TransportError::Http { status, .. }))
                    if status == StatusCode::UPGRADE_REQUIRED =>
                {
                    return Ok(WebsocketStreamOutcome::FallbackToHttp);
                }
                Err(ApiError::Transport(
                    unauthorized_transport @ TransportError::Http { status, .. },
                )) if status == StatusCode::UNAUTHORIZED => {
                    handle_unauthorized(unauthorized_transport, &mut auth_recovery).await?;
                    continue;
                }
                Err(err) => return Err(map_api_error(err)),
            }

            let ws_request = self.prepare_websocket_request(ws_payload, &request, ws_version);
            self.websocket_session.last_request = Some(request);
            let stream_result = self
                .websocket_session
                .connection
                .as_ref()
                .ok_or_else(|| {
                    map_api_error(ApiError::Stream(
                        "websocket connection is unavailable".to_string(),
                    ))
                })?
                .stream_request(ws_request)
                .await
                .map_err(map_api_error)?;
            let (stream, last_request_rx) =
                map_response_stream(stream_result, otel_manager.clone());
            self.websocket_session.last_response_rx = Some(last_request_rx);
            return Ok(WebsocketStreamOutcome::Stream(stream));
        }
    }

    /// Builds request and SSE telemetry for streaming API calls.
    fn build_streaming_telemetry(
        otel_manager: &OtelManager,
    ) -> (Arc<dyn RequestTelemetry>, Arc<dyn SseTelemetry>) {
        let telemetry = Arc::new(ApiTelemetry::new(otel_manager.clone()));
        let request_telemetry: Arc<dyn RequestTelemetry> = telemetry.clone();
        let sse_telemetry: Arc<dyn SseTelemetry> = telemetry;
        (request_telemetry, sse_telemetry)
    }

    /// Builds telemetry for the Responses API WebSocket transport.
    fn build_websocket_telemetry(otel_manager: &OtelManager) -> Arc<dyn WebsocketTelemetry> {
        let telemetry = Arc::new(ApiTelemetry::new(otel_manager.clone()));
        let websocket_telemetry: Arc<dyn WebsocketTelemetry> = telemetry;
        websocket_telemetry
    }

#[derive(Clone, Debug)]
struct OAuthRotationConfigResolved {
    max_attempts: usize,
    rate_limit_cooldown: Duration,
    auth_failure_cooldown: Duration,
    payment_required_cooldown: Duration,
    network_retry_attempts: u32,
}

struct OAuthRotationPlan {
    auth_manager: Arc<AuthManager>,
    namespace: String,
    candidates: Vec<String>,
    record_by_id: HashMap<String, crate::auth::OAuthPoolRecord>,
    config: OAuthRotationConfigResolved,
}

impl OAuthRotationPlan {
    async fn try_new(
        auth_manager: Arc<AuthManager>,
        provider: &ModelProviderInfo,
        config: &Config,
        namespace: &str,
    ) -> Option<Self> {
        if !provider.is_openai() {
            return None;
        }
        let auth = auth_manager.auth().await?;
        if auth.mode != AuthMode::ChatGPT {
            return None;
        }

        let snapshot = auth_manager.oauth_snapshot(namespace).ok()?;
        if snapshot.records.len() <= 1 {
            return None;
        }

        let record_by_id: HashMap<_, _> = snapshot
            .records
            .into_iter()
            .map(|record| (record.id.clone(), record))
            .collect();
        let candidates: Vec<String> = snapshot
            .ordered_ids
            .into_iter()
            .filter(|id| record_by_id.contains_key(id))
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let resolved = resolve_oauth_rotation_config(config, candidates.len());

        Some(Self {
            auth_manager,
            namespace: namespace.to_string(),
            candidates,
            record_by_id,
            config: resolved,
        })
    }

    fn refreshed_snapshot(&self) -> (Vec<String>, HashMap<String, crate::auth::OAuthPoolRecord>) {
        if let Ok(snapshot) = self.auth_manager.oauth_snapshot(&self.namespace) {
            let record_by_id: HashMap<_, _> = snapshot
                .records
                .into_iter()
                .map(|record| (record.id.clone(), record))
                .collect();
            if !record_by_id.is_empty() {
                let candidates: Vec<String> = snapshot
                    .ordered_ids
                    .into_iter()
                    .filter(|id| self.candidates.contains(id))
                    .collect();
                if !candidates.is_empty() {
                    return (candidates, record_by_id);
                }
                return (self.candidates.clone(), record_by_id);
            }
        }

        (self.candidates.clone(), self.record_by_id.clone())
    }

    async fn execute<T, F, Fut>(&self, mut request_fn: F) -> std::result::Result<T, ApiError>
    where
        F: FnMut(&str, CodexAuth) -> Fut,
        Fut: std::future::Future<Output = std::result::Result<T, ApiError>>,
    {
        let mut attempted: HashSet<String> = HashSet::new();
        let mut refreshed: HashSet<String> = HashSet::new();
        let mut last_error: Option<ApiError> = None;

        let max_attempts = self
            .config
            .max_attempts
            .max(1)
            .min(self.candidates.len().max(1));

        for attempt in 0..max_attempts {
            let (candidates, record_by_id) = self.refreshed_snapshot();
            let Some(record_id) = pick_next_candidate(&candidates, &record_by_id, &attempted)
            else {
                break;
            };
            attempted.insert(record_id.clone());

            let mut network_attempt = 0u32;
            loop {
                let Some(auth) = self.auth_manager.auth_for_record(&record_id) else {
                    break;
                };

                let result = request_fn(&record_id, auth).await;
                match result {
                    Ok(value) => {
                        let _ = self
                            .auth_manager
                            .oauth_record_outcome(&record_id, 200, true, None);
                        return Ok(value);
                    }
                    Err(ApiError::Transport(TransportError::Http {
                        status,
                        url,
                        headers,
                        body,
                    })) => {
                        let status_code = status.as_u16();
                        let headers_ref = headers.as_ref();

                        if status == StatusCode::BAD_REQUEST {
                            let _ = self.auth_manager.oauth_record_outcome(
                                &record_id,
                                status_code,
                                false,
                                None,
                            );
                            return Err(ApiError::Transport(TransportError::Http {
                                status,
                                url,
                                headers,
                                body,
                            }));
                        }

                        if status == StatusCode::PAYMENT_REQUIRED {
                            let exhausted_until =
                                cooldown_until_from_duration(self.config.payment_required_cooldown);
                            let _ = self.auth_manager.oauth_record_exhausted(
                                &record_id,
                                status_code,
                                Some(exhausted_until),
                            );
                            let _ = self
                                .auth_manager
                                .oauth_move_to_back(&self.namespace, &record_id);
                            let err = ApiError::Transport(TransportError::Http {
                                status,
                                url,
                                headers,
                                body,
                            });
                            if attempt + 1 >= max_attempts {
                                return Err(err);
                            }
                            last_error = Some(err);
                            break;
                        }

                        if status == StatusCode::TOO_MANY_REQUESTS {
                            let cooldown = retry_after_duration(headers_ref)
                                .unwrap_or(self.config.rate_limit_cooldown);
                            let cooldown_until = cooldown_until_from_duration(cooldown);
                            let _ = self.auth_manager.oauth_record_outcome(
                                &record_id,
                                status_code,
                                false,
                                Some(cooldown_until),
                            );
                            let _ = self
                                .auth_manager
                                .oauth_move_to_back(&self.namespace, &record_id);
                            let err = ApiError::Transport(TransportError::Http {
                                status,
                                url,
                                headers,
                                body,
                            });
                            if attempt + 1 >= max_attempts {
                                return Err(err);
                            }
                            last_error = Some(err);
                            break;
                        }

                        if is_auth_failure_status(status) {
                            if !refreshed.contains(&record_id) {
                                refreshed.insert(record_id.clone());
                                match self.auth_manager.refresh_record(&record_id).await {
                                    Ok(()) => {
                                        let Some(auth) =
                                            self.auth_manager.auth_for_record(&record_id)
                                        else {
                                            break;
                                        };
                                        let retry = request_fn(&record_id, auth).await;
                                        match retry {
                                            Ok(value) => {
                                                let _ = self.auth_manager.oauth_record_outcome(
                                                    &record_id, 200, true, None,
                                                );
                                                return Ok(value);
                                            }
                                            Err(ApiError::Transport(TransportError::Http {
                                                status,
                                                url,
                                                headers,
                                                body,
                                            })) => {
                                                if status == StatusCode::BAD_REQUEST {
                                                    let _ = self.auth_manager.oauth_record_outcome(
                                                        &record_id,
                                                        status.as_u16(),
                                                        false,
                                                        None,
                                                    );
                                                    return Err(ApiError::Transport(
                                                        TransportError::Http {
                                                            status,
                                                            url,
                                                            headers,
                                                            body,
                                                        },
                                                    ));
                                                }

                                                if status == StatusCode::PAYMENT_REQUIRED {
                                                    let exhausted_until =
                                                        cooldown_until_from_duration(
                                                            self.config.payment_required_cooldown,
                                                        );
                                                    let _ =
                                                        self.auth_manager.oauth_record_exhausted(
                                                            &record_id,
                                                            status.as_u16(),
                                                            Some(exhausted_until),
                                                        );
                                                    let _ = self.auth_manager.oauth_move_to_back(
                                                        &self.namespace,
                                                        &record_id,
                                                    );
                                                    let err =
                                                        ApiError::Transport(TransportError::Http {
                                                            status,
                                                            url,
                                                            headers,
                                                            body,
                                                        });
                                                    if attempt + 1 >= max_attempts {
                                                        return Err(err);
                                                    }
                                                    last_error = Some(err);
                                                    break;
                                                }

                                                if status == StatusCode::TOO_MANY_REQUESTS {
                                                    let cooldown =
                                                        retry_after_duration(headers.as_ref())
                                                            .unwrap_or(
                                                                self.config.rate_limit_cooldown,
                                                            );
                                                    let cooldown_until =
                                                        cooldown_until_from_duration(cooldown);
                                                    let _ = self.auth_manager.oauth_record_outcome(
                                                        &record_id,
                                                        status.as_u16(),
                                                        false,
                                                        Some(cooldown_until),
                                                    );
                                                    let _ = self.auth_manager.oauth_move_to_back(
                                                        &self.namespace,
                                                        &record_id,
                                                    );
                                                    let err =
                                                        ApiError::Transport(TransportError::Http {
                                                            status,
                                                            url,
                                                            headers,
                                                            body,
                                                        });
                                                    if attempt + 1 >= max_attempts {
                                                        return Err(err);
                                                    }
                                                    last_error = Some(err);
                                                    break;
                                                }

                                                let cooldown_until = cooldown_until_from_duration(
                                                    self.config.auth_failure_cooldown,
                                                );
                                                let _ = self.auth_manager.oauth_record_outcome(
                                                    &record_id,
                                                    status.as_u16(),
                                                    false,
                                                    Some(cooldown_until),
                                                );
                                                let _ = self.auth_manager.oauth_move_to_back(
                                                    &self.namespace,
                                                    &record_id,
                                                );
                                                let err =
                                                    ApiError::Transport(TransportError::Http {
                                                        status,
                                                        url,
                                                        headers,
                                                        body,
                                                    });
                                                if attempt + 1 >= max_attempts {
                                                    return Err(err);
                                                }
                                                last_error = Some(err);
                                                break;
                                            }
                                            Err(ApiError::Transport(TransportError::Timeout)) => {
                                                let err =
                                                    ApiError::Transport(TransportError::Timeout);
                                                let _ = self.auth_manager.oauth_record_outcome(
                                                    &record_id, 0, false, None,
                                                );
                                                if network_attempt
                                                    < self.config.network_retry_attempts
                                                {
                                                    network_attempt += 1;
                                                    continue;
                                                }
                                                last_error = Some(err);
                                                break;
                                            }
                                            Err(ApiError::Transport(TransportError::Network(
                                                msg,
                                            ))) => {
                                                let err = ApiError::Transport(
                                                    TransportError::Network(msg),
                                                );
                                                let _ = self.auth_manager.oauth_record_outcome(
                                                    &record_id, 0, false, None,
                                                );
                                                if network_attempt
                                                    < self.config.network_retry_attempts
                                                {
                                                    network_attempt += 1;
                                                    continue;
                                                }
                                                last_error = Some(err);
                                                break;
                                            }
                                            Err(ApiError::Transport(TransportError::Build(
                                                msg,
                                            ))) => {
                                                let err =
                                                    ApiError::Transport(TransportError::Build(msg));
                                                let _ = self.auth_manager.oauth_record_outcome(
                                                    &record_id, 0, false, None,
                                                );
                                                if network_attempt
                                                    < self.config.network_retry_attempts
                                                {
                                                    network_attempt += 1;
                                                    continue;
                                                }
                                                last_error = Some(err);
                                                break;
                                            }
                                            Err(err) => return Err(err),
                                        }
                                    }
                                    Err(err) => {
                                        let requires_relogin = matches!(
                                            err.failed_reason(),
                                            Some(RefreshTokenFailedReason::Expired)
                                                | Some(RefreshTokenFailedReason::Exhausted)
                                                | Some(RefreshTokenFailedReason::Revoked)
                                                | Some(RefreshTokenFailedReason::Other)
                                        );
                                        if requires_relogin {
                                            let _ =
                                                self.auth_manager.oauth_record_requires_relogin(
                                                    &record_id,
                                                    status_code,
                                                );
                                        } else {
                                            let cooldown_until = cooldown_until_from_duration(
                                                self.config.auth_failure_cooldown,
                                            );
                                            let _ = self.auth_manager.oauth_record_outcome(
                                                &record_id,
                                                status_code,
                                                false,
                                                Some(cooldown_until),
                                            );
                                        }
                                        let _ = self
                                            .auth_manager
                                            .oauth_move_to_back(&self.namespace, &record_id);
                                        let err = ApiError::Transport(TransportError::Http {
                                            status,
                                            url,
                                            headers,
                                            body,
                                        });
                                        if attempt + 1 >= max_attempts {
                                            return Err(err);
                                        }
                                        last_error = Some(err);
                                        break;
                                    }
                                }
                            } else {
                                let cooldown_until =
                                    cooldown_until_from_duration(self.config.auth_failure_cooldown);
                                let _ = self.auth_manager.oauth_record_outcome(
                                    &record_id,
                                    status_code,
                                    false,
                                    Some(cooldown_until),
                                );
                                let _ = self
                                    .auth_manager
                                    .oauth_move_to_back(&self.namespace, &record_id);
                                let err = ApiError::Transport(TransportError::Http {
                                    status,
                                    url,
                                    headers,
                                    body,
                                });
                                if attempt + 1 >= max_attempts {
                                    return Err(err);
                                }
                                last_error = Some(err);
                                break;
                            }
                        }

                        let _ = self.auth_manager.oauth_record_outcome(
                            &record_id,
                            status_code,
                            false,
                            None,
                        );
                        let _ = self
                            .auth_manager
                            .oauth_move_to_back(&self.namespace, &record_id);
                        let err = ApiError::Transport(TransportError::Http {
                            status,
                            url,
                            headers,
                            body,
                        });
                        if attempt + 1 >= max_attempts {
                            return Err(err);
                        }
                        last_error = Some(err);
                        break;
                    }
                    Err(ApiError::Transport(TransportError::Timeout)) => {
                        let err = ApiError::Transport(TransportError::Timeout);
                        let _ = self
                            .auth_manager
                            .oauth_record_outcome(&record_id, 0, false, None);
                        if network_attempt < self.config.network_retry_attempts {
                            network_attempt += 1;
                            continue;
                        }
                        last_error = Some(err);
                        break;
                    }
                    Err(ApiError::Transport(TransportError::Network(msg))) => {
                        let err = ApiError::Transport(TransportError::Network(msg));
                        let _ = self
                            .auth_manager
                            .oauth_record_outcome(&record_id, 0, false, None);
                        if network_attempt < self.config.network_retry_attempts {
                            network_attempt += 1;
                            continue;
                        }
                        last_error = Some(err);
                        break;
                    }
                    Err(ApiError::Transport(TransportError::Build(msg))) => {
                        let err = ApiError::Transport(TransportError::Build(msg));
                        let _ = self
                            .auth_manager
                            .oauth_record_outcome(&record_id, 0, false, None);
                        if network_attempt < self.config.network_retry_attempts {
                            network_attempt += 1;
                            continue;
                        }
                        last_error = Some(err);
                        break;
                    }
                    Err(err) => return Err(err),
                }
            }
        }

        match last_error {
            Some(err) => Err(err),
            None => Err(ApiError::Stream("OAuth rotation exhausted".to_string())),
        }
    }
}

fn resolve_oauth_rotation_config(
    config: &Config,
    candidate_len: usize,
) -> OAuthRotationConfigResolved {
    let settings = &config.oauth_rotation;
    let max_attempts = settings.max_attempts.unwrap_or(candidate_len as u32).max(1) as usize;
    let max_attempts = max_attempts.min(candidate_len.max(1));
    let rate_limit_cooldown = Duration::from_millis(
        settings
            .rate_limit_cooldown_ms
            .unwrap_or(DEFAULT_RATE_LIMIT_COOLDOWN_MS),
    );
    let auth_failure_cooldown = Duration::from_millis(
        settings
            .auth_failure_cooldown_ms
            .unwrap_or(DEFAULT_AUTH_FAILURE_COOLDOWN_MS),
    );
    let payment_required_cooldown = Duration::from_millis(
        settings
            .payment_required_cooldown_ms
            .unwrap_or(DEFAULT_PAYMENT_REQUIRED_COOLDOWN_MS),
    );
    let network_retry_attempts = settings
        .network_retry_attempts
        .unwrap_or(DEFAULT_NETWORK_RETRY_ATTEMPTS);

    OAuthRotationConfigResolved {
        max_attempts,
        rate_limit_cooldown,
        auth_failure_cooldown,
        payment_required_cooldown,
        network_retry_attempts,
    }
}

fn pick_next_candidate(
    candidates: &[String],
    record_by_id: &HashMap<String, crate::auth::OAuthPoolRecord>,
    attempted: &HashSet<String>,
) -> Option<String> {
    let now = Utc::now();
    candidates
        .iter()
        .find(|id| {
            if attempted.contains(*id) {
                return false;
            }
            let Some(record) = record_by_id.get(*id) else {
                return false;
            };
            if record.health.requires_relogin {
                return false;
            }
            if record
                .health
                .exhausted_until
                .is_some_and(|until| until > now)
            {
                return false;
            }
            let cooldown = record.health.cooldown_until;
            cooldown.map(|until| until <= now).unwrap_or(true)
        })
        .or_else(|| {
            candidates.iter().find(|id| {
                if attempted.contains(*id) {
                    return false;
                }
                let Some(record) = record_by_id.get(*id) else {
                    return false;
                };
                if record.health.requires_relogin {
                    return false;
                }
                if record
                    .health
                    .exhausted_until
                    .is_some_and(|until| until > now)
                {
                    return false;
                }
                true
            })
        })
        .cloned()
}

fn retry_after_duration(headers: Option<&ApiHeaderMap>) -> Option<Duration> {
    let headers = headers?;
    let value = headers.get("retry-after")?;
    let value = value.to_str().ok()?.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    match httpdate::parse_http_date(value) {
        Ok(time) => Some(time.duration_since(SystemTime::now()).unwrap_or_default()),
        Err(_) => None,
    }
}

fn cooldown_until_from_duration(duration: Duration) -> DateTime<Utc> {
    let chrono_duration =
        chrono::Duration::from_std(duration).unwrap_or_else(|_| chrono::Duration::zero());
    Utc::now() + chrono_duration
}

fn is_auth_failure_status(status: StatusCode) -> bool {
    status == StatusCode::UNAUTHORIZED || status == StatusCode::FORBIDDEN
}

fn responses_request_compression_from_state(
    state: &ModelClientState,
    auth: Option<&CodexAuth>,
) -> Compression {
    if state
        .config
        .features
        .enabled(Feature::EnableRequestCompression)
        && auth.is_some_and(|auth| auth.mode == AuthMode::ChatGPT)
        && state.provider.is_openai()
    {
        Compression::Zstd
    } else {
        Compression::None
    }
}

fn build_responses_options_from_state(
    state: &ModelClientState,
    turn_state: &Arc<OnceLock<String>>,
    prompt: &Prompt,
    compression: Compression,
) -> ApiResponsesOptions {
    let model_info = &state.model_info;

    let default_reasoning_effort = model_info.default_reasoning_level;
    let reasoning = if model_info.supports_reasoning_summaries {
        Some(Reasoning {
            effort: state.effort.or(default_reasoning_effort),
            summary: if state.summary == ReasoningSummaryConfig::None {
                None
            } else {
                Some(state.summary)
            },
        })
    } else {
        None
    };

    let include = if reasoning.is_some() {
        vec!["reasoning.encrypted_content".to_string()]
    } else {
        Vec::new()
    };

    let verbosity = if model_info.support_verbosity {
        state
            .config
            .model_verbosity
            .or(model_info.default_verbosity)
    } else {
        if state.config.model_verbosity.is_some() {
            warn!(
                "model_verbosity is set but ignored as the model does not support verbosity: {}",
                model_info.slug
            );
        }
        None
    };

    let text = create_text_param_for_request(verbosity, &prompt.output_schema);
    let conversation_id = state.conversation_id.to_string();

    ApiResponsesOptions {
        reasoning,
        include,
        prompt_cache_key: Some(conversation_id.clone()),
        text,
        store_override: None,
        conversation_id: Some(conversation_id),
        session_source: Some(state.session_source.clone()),
        extra_headers: build_responses_headers(&state.config, Some(turn_state)),
        compression,
        turn_state: Some(Arc::clone(turn_state)),
    }
}

fn get_incremental_items_from(
    last_items: &[ResponseItem],
    input_items: &[ResponseItem],
) -> Option<Vec<ResponseItem>> {
    // Checks whether the current request input is an incremental append to the previous request.
    // If items in the new request contain all the items from the previous request we build
    // a response.append request otherwise we start with a fresh response.create request.
    let previous_len = last_items.len();
    let can_append =
        previous_len > 0 && input_items.starts_with(last_items) && previous_len < input_items.len();
    if can_append {
        Some(input_items[previous_len..].to_vec())
    } else {
        None
    }
}

fn prepare_websocket_request_with_last_items(
    model_slug: &str,
    api_prompt: &ApiPrompt,
    options: &ApiResponsesOptions,
    last_items: &[ResponseItem],
) -> ResponsesWsRequest {
    if let Some(append_items) = get_incremental_items_from(last_items, &api_prompt.input) {
        return ResponsesWsRequest::ResponseAppend(ResponseAppendWsRequest {
            input: append_items,
        });
    }

    let ApiResponsesOptions {
        reasoning,
        include,
        prompt_cache_key,
        text,
        store_override,
        ..
    } = options;

    let store = store_override.unwrap_or(false);
    let payload = ResponseCreateWsRequest {
        model: model_slug.to_string(),
        instructions: api_prompt.instructions.clone(),
        input: api_prompt.input.clone(),
        tools: api_prompt.tools.clone(),
        tool_choice: "auto".to_string(),
        parallel_tool_calls: api_prompt.parallel_tool_calls,
        reasoning: reasoning.clone(),
        store,
        stream: true,
        include: include.clone(),
        prompt_cache_key: prompt_cache_key.clone(),
        text: text.clone(),
    };

    ResponsesWsRequest::ResponseCreate(payload)
}

/// Adapts the core `Prompt` type into the `codex-api` payload shape.
fn build_api_prompt(prompt: &Prompt, instructions: String, tools_json: Vec<Value>) -> ApiPrompt {
    ApiPrompt {
        instructions,
        input: prompt.get_formatted_input(),
        tools: tools_json,
        parallel_tool_calls: prompt.parallel_tool_calls,
        output_schema: prompt.output_schema.clone(),
    }
}

        if matches!(ws_version, ResponsesWebsocketVersion::V1) {
            self.preconnect_websocket(otel_manager, model_info)
                .await
                .map_err(map_api_error)?;
            return Ok(());
        }

        match self
            .stream_responses_websocket(
                prompt,
                model_info,
                ws_version,
                otel_manager,
                effort,
                summary,
                turn_metadata_header,
                true,
            )
            .await
        {
            Ok(WebsocketStreamOutcome::Stream(mut stream)) => {
                // Wait for the v2 warmup request to complete before sending the first turn request.
                while let Some(event) = stream.next().await {
                    match event {
                        Ok(ResponseEvent::Completed { .. }) => break,
                        Err(err) => return Err(err),
                        _ => {}
                    }
                }
                Ok(())
            }
            Ok(WebsocketStreamOutcome::FallbackToHttp) => {
                self.try_switch_fallback_transport(otel_manager, model_info);
                Ok(())
            }
            Err(err) => Err(err),
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Streams a single model request within the current turn.
    ///
    /// The caller is responsible for passing per-turn settings explicitly (model selection,
    /// reasoning settings, telemetry context, and turn metadata). This method will prefer the
    /// Responses WebSocket transport when enabled and healthy, and will fall back to the HTTP
    /// Responses API transport otherwise.
    pub async fn stream(
        &mut self,
        prompt: &Prompt,
        model_info: &ModelInfo,
        otel_manager: &OtelManager,
        effort: Option<ReasoningEffortConfig>,
        summary: ReasoningSummaryConfig,
        turn_metadata_header: Option<&str>,
    ) -> Result<ResponseStream> {
        let wire_api = self.client.state.provider.wire_api;
        match wire_api {
            WireApi::Responses => {
                if let Some(ws_version) = self.client.active_ws_version(model_info) {
                    match self
                        .stream_responses_websocket(
                            prompt,
                            model_info,
                            ws_version,
                            otel_manager,
                            effort,
                            summary,
                            turn_metadata_header,
                            false,
                        )
                        .await?
                    {
                        WebsocketStreamOutcome::Stream(stream) => return Ok(stream),
                        WebsocketStreamOutcome::FallbackToHttp => {
                            self.try_switch_fallback_transport(otel_manager, model_info);
                        }
                    }
                }

                self.stream_responses_api(
                    prompt,
                    model_info,
                    otel_manager,
                    effort,
                    summary,
                    turn_metadata_header,
                )
                .await
            }
        }
    }

    /// Permanently disables WebSockets for this Codex session and resets WebSocket state.
    ///
    /// This is used after exhausting the provider retry budget, to force subsequent requests onto
    /// the HTTP transport.
    ///
    /// Returns `true` if this call activated fallback, or `false` if fallback was already active.
    pub(crate) fn try_switch_fallback_transport(
        &mut self,
        otel_manager: &OtelManager,
        model_info: &ModelInfo,
    ) -> bool {
        let websocket_enabled = self.client.active_ws_version(model_info).is_some();
        let activated = self.activate_http_fallback(websocket_enabled);
        if activated {
            warn!("falling back to HTTP");
            otel_manager.counter(
                "codex.transport.fallback_to_http",
                1,
                &[("from_wire_api", "responses_websocket")],
            );

            self.websocket_session.connection = None;
            self.websocket_session.last_request = None;
            self.websocket_session.last_response_rx = None;
        }
        activated
    }
}

/// Parses per-turn metadata into an HTTP header value.
///
/// Invalid values are treated as absent so callers can compare and propagate
/// metadata with the same sanitization path used when constructing headers.
fn parse_turn_metadata_header(turn_metadata_header: Option<&str>) -> Option<HeaderValue> {
    turn_metadata_header.and_then(|value| HeaderValue::from_str(value).ok())
}

fn build_ws_client_metadata(turn_metadata_header: Option<&str>) -> Option<HashMap<String, String>> {
    let turn_metadata_header = parse_turn_metadata_header(turn_metadata_header)?;
    let turn_metadata = turn_metadata_header.to_str().ok()?.to_string();
    let mut client_metadata = HashMap::new();
    client_metadata.insert(X_CODEX_TURN_METADATA_HEADER.to_string(), turn_metadata);
    Some(client_metadata)
}

/// Builds the extra headers attached to Responses API requests.
///
/// These headers implement Codex-specific conventions:
///
/// - `x-codex-beta-features`: comma-separated beta feature keys enabled for the session.
/// - `x-codex-turn-state`: sticky routing token captured earlier in the turn.
/// - `x-codex-turn-metadata`: optional per-turn metadata for observability.
fn build_responses_headers(
    beta_features_header: Option<&str>,
    turn_state: Option<&Arc<OnceLock<String>>>,
    turn_metadata_header: Option<&HeaderValue>,
) -> ApiHeaderMap {
    let mut headers = ApiHeaderMap::new();
    if let Some(value) = beta_features_header
        && !value.is_empty()
        && let Ok(header_value) = HeaderValue::from_str(value)
    {
        headers.insert("x-codex-beta-features", header_value);
    }
    if let Some(turn_state) = turn_state
        && let Some(state) = turn_state.get()
        && let Ok(header_value) = HeaderValue::from_str(state)
    {
        headers.insert(X_CODEX_TURN_STATE_HEADER, header_value);
    }
    if let Some(header_value) = turn_metadata_header {
        headers.insert(X_CODEX_TURN_METADATA_HEADER, header_value.clone());
    }
    headers
}

fn map_response_stream<S>(
    api_stream: S,
    otel_manager: OtelManager,
) -> (ResponseStream, oneshot::Receiver<LastResponse>)
where
    S: futures::Stream<Item = std::result::Result<ResponseEvent, ApiError>>
        + Unpin
        + Send
        + 'static,
{
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent>>(1600);
    let (tx_last_response, rx_last_response) = oneshot::channel::<LastResponse>();

    tokio::spawn(async move {
        let mut logged_error = false;
        let mut tx_last_response = Some(tx_last_response);
        let mut items_added: Vec<ResponseItem> = Vec::new();
        let mut api_stream = api_stream;
        while let Some(event) = api_stream.next().await {
            match event {
                Ok(ResponseEvent::OutputItemDone(item)) => {
                    items_added.push(item.clone());
                    if tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(ResponseEvent::Completed {
                    response_id,
                    token_usage,
                    can_append,
                }) => {
                    if let Some(usage) = &token_usage {
                        otel_manager.sse_event_completed(
                            usage.input_tokens,
                            usage.output_tokens,
                            Some(usage.cached_input_tokens),
                            Some(usage.reasoning_output_tokens),
                            usage.total_tokens,
                        );
                    }
                    if let Some(sender) = tx_last_response.take() {
                        let _ = sender.send(LastResponse {
                            response_id: response_id.clone(),
                            items_added: std::mem::take(&mut items_added),
                            can_append,
                        });
                    }
                    if tx_event
                        .send(Ok(ResponseEvent::Completed {
                            response_id,
                            token_usage,
                            can_append,
                        }))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                Ok(event) => {
                    if tx_event.send(Ok(event)).await.is_err() {
                        return;
                    }
                }
                Err(err) => {
                    let mapped = map_api_error(err);
                    if !logged_error {
                        otel_manager.see_event_completed_failed(&mapped);
                        logged_error = true;
                    }
                    if tx_event.send(Err(mapped)).await.is_err() {
                        return;
                    }
                }
            }
        }
    });

    (ResponseStream { rx_event }, rx_last_response)
}

/// Handles a 401 response by optionally refreshing ChatGPT tokens once.
///
/// When refresh succeeds, the caller should retry the API call; otherwise
/// the mapped `CodexErr` is returned to the caller.
async fn handle_unauthorized(
    transport: TransportError,
    auth_recovery: &mut Option<UnauthorizedRecovery>,
) -> Result<()> {
    if let Some(recovery) = auth_recovery
        && recovery.has_next()
    {
        return match recovery.next().await {
            Ok(_) => Ok(()),
            Err(RefreshTokenError::Permanent(failed)) => Err(CodexErr::RefreshTokenFailed(failed)),
            Err(RefreshTokenError::Transient(other)) => Err(CodexErr::Io(other)),
        };
    }

    Err(map_api_error(ApiError::Transport(transport)))
}

struct ApiTelemetry {
    otel_manager: OtelManager,
}

impl ApiTelemetry {
    fn new(otel_manager: OtelManager) -> Self {
        Self { otel_manager }
    }
}

impl RequestTelemetry for ApiTelemetry {
    fn on_request(
        &self,
        attempt: u64,
        status: Option<HttpStatusCode>,
        error: Option<&TransportError>,
        duration: Duration,
    ) {
        let error_message = error.map(std::string::ToString::to_string);
        self.otel_manager.record_api_request(
            attempt,
            status.map(|s| s.as_u16()),
            error_message.as_deref(),
            duration,
        );
    }
}

impl SseTelemetry for ApiTelemetry {
    fn on_sse_poll(
        &self,
        result: &std::result::Result<
            Option<std::result::Result<Event, EventStreamError<TransportError>>>,
            tokio::time::error::Elapsed,
        >,
        duration: Duration,
    ) {
        self.otel_manager.log_sse_event(result, duration);
    }
}

impl WebsocketTelemetry for ApiTelemetry {
    fn on_ws_request(&self, duration: Duration, error: Option<&ApiError>) {
        let error_message = error.map(std::string::ToString::to_string);
        self.otel_manager
            .record_websocket_request(duration, error_message.as_deref());
    }

    fn on_ws_event(
        &self,
        result: &std::result::Result<Option<std::result::Result<Message, Error>>, ApiError>,
        duration: Duration,
    ) {
        self.otel_manager.record_websocket_event(result, duration);
    }
}

#[cfg(test)]
mod tests {
    use super::ModelClient;
    use codex_otel::OtelManager;
    use codex_protocol::ThreadId;
    use codex_protocol::openai_models::ModelInfo;
    use codex_protocol::protocol::SessionSource;
    use codex_protocol::protocol::SubAgentSource;
    use pretty_assertions::assert_eq;
    use serde_json::json;

    fn test_model_client(session_source: SessionSource) -> ModelClient {
        let provider = crate::model_provider_info::create_oss_provider_with_base_url(
            "https://example.com/v1",
            crate::model_provider_info::WireApi::Responses,
        );
        ModelClient::new(
            None,
            ThreadId::new(),
            provider,
            session_source,
            None,
            None,
            false,
            false,
            None,
        )
    }

    fn test_model_info() -> ModelInfo {
        serde_json::from_value(json!({
            "slug": "gpt-test",
            "display_name": "gpt-test",
            "description": "desc",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "medium", "description": "medium"}
            ],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": 1,
            "upgrade": null,
            "base_instructions": "base instructions",
            "model_messages": null,
            "supports_reasoning_summaries": false,
            "support_verbosity": false,
            "default_verbosity": null,
            "apply_patch_tool_type": null,
            "truncation_policy": {"mode": "bytes", "limit": 10000},
            "supports_parallel_tool_calls": false,
            "context_window": 272000,
            "auto_compact_token_limit": null,
            "experimental_supported_tools": []
        }))
        .expect("deserialize test model info")
    }

    fn test_otel_manager() -> OtelManager {
        OtelManager::new(
            ThreadId::new(),
            "gpt-test",
            "gpt-test",
            None,
            None,
            None,
            "test-originator".to_string(),
            false,
            "test-terminal".to_string(),
            SessionSource::Cli,
        )
    }

    #[test]
    fn build_subagent_headers_sets_other_subagent_label() {
        let client = test_model_client(SessionSource::SubAgent(SubAgentSource::Other(
            "memory_consolidation".to_string(),
        )));
        let headers = client.build_subagent_headers();
        let value = headers
            .get("x-openai-subagent")
            .and_then(|value| value.to_str().ok());
        assert_eq!(value, Some("memory_consolidation"));
    }

    #[tokio::test]
    async fn summarize_memories_returns_empty_for_empty_input() {
        let client = test_model_client(SessionSource::Cli);
        let model_info = test_model_info();
        let otel_manager = test_otel_manager();

        let output = client
            .summarize_memories(Vec::new(), &model_info, None, &otel_manager)
            .await
            .expect("empty summarize request should succeed");
        assert_eq!(output.len(), 0);
    }
}
