//! Team OAuth automation for managing team accounts and member logins.

use crate::auth::{add_oauth_account, AuthCredentialsStoreMode};
use crate::team_oauth::otp_client::{OtpClient, OtpError, EmailMessage};
use crate::team_oauth::team_store::{TeamAccount, TeamMember, TeamRole, TeamStore};
use crate::token_data::{TokenData, parse_chatgpt_jwt_claims};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error, debug};

/// OAuth configuration for ChatGPT.
const OAUTH_CLIENT_ID: &str = "pdlDIXYxOzm5UCRlKqFMjrLWPSQ5q6En";
const OAUTH_ISSUER: &str = "https://auth.openai.com";

/// Errors during team OAuth automation.
#[derive(Debug, thiserror::Error)]
pub enum AutomationError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("OTP error: {0}")]
    Otp(#[from] OtpError),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Team not found: {0}")]
    TeamNotFound(String),

    #[error("Member not found: {0}")]
    MemberNotFound(String),

    #[error("OAuth login failed for {email}: {reason}")]
    OAuthFailed { email: String, reason: String },

    #[error("Invitation failed: {0}")]
    InvitationFailed(String),

    #[error("Timeout waiting for OTP: {0}")]
    OtpTimeout(String),

    #[error("JSON parse error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Result of a team setup operation.
#[derive(Debug, Clone)]
pub struct SetupResult {
    /// Team ID
    pub team_id: String,
    /// Successfully logged in members
    pub logged_in_members: Vec<MemberLoginResult>,
    /// Members that failed to log in
    pub failed_members: Vec<MemberLoginResult>,
}

/// Result of a member login attempt.
#[derive(Debug, Clone)]
pub struct MemberLoginResult {
    /// Member email
    pub email: String,
    /// Member ID
    pub member_id: String,
    /// Whether login was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

/// Response from OAuth token endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenResponse {
    access_token: String,
    refresh_token: String,
    id_token: String,
    expires_in: Option<i64>,
    token_type: String,
}

/// Response from passwordless/start endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PasswordlessStartResponse {
    #[serde(default)]
    email: Option<String>,
    #[serde(default)]
    verification: Option<VerificationInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerificationInfo {
    #[serde(default)]
    code: Option<String>,
}

/// Team OAuth automation manager.
pub struct TeamOAuthAutomation {
    codex_home: PathBuf,
    store_mode: AuthCredentialsStoreMode,
    otp_client: OtpClient,
    http_client: Client,
    team_store: Arc<Mutex<TeamStore>>,
}

impl TeamOAuthAutomation {
    /// Create a new automation instance.
    pub fn new(
        codex_home: PathBuf,
        store_mode: AuthCredentialsStoreMode,
        otp_worker_url: Option<String>,
    ) -> io::Result<Self> {
        let team_store = TeamStore::load(&codex_home)?;
        let otp_client = match otp_worker_url {
            Some(url) => OtpClient::new(url),
            None => OtpClient::default(),
        };

        let http_client = Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
            .build()
            .map_err(|e| io::Error::other(format!("failed to create HTTP client: {e}")))?;

        Ok(Self {
            codex_home,
            store_mode,
            otp_client,
            http_client,
            team_store: Arc::new(Mutex::new(team_store)),
        })
    }

    /// Create automation with Cloudflare email access.
    pub fn with_cloudflare(
        codex_home: PathBuf,
        store_mode: AuthCredentialsStoreMode,
        cf_token: String,
        cf_account_id: String,
    ) -> io::Result<Self> {
        let team_store = TeamStore::load(&codex_home)?;
        let otp_client = OtpClient::with_cloudflare(
            OtpClient::DEFAULT_URL.to_string(),
            cf_token,
            cf_account_id,
        );

        let http_client = Client::builder()
            .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
            .build()
            .map_err(|e| io::Error::other(format!("failed to create HTTP client: {e}")))?;

        Ok(Self {
            codex_home,
            store_mode,
            otp_client,
            http_client,
            team_store: Arc::new(Mutex::new(team_store)),
        })
    }

    /// Full setup: create team, invite members, and auto-login.
    ///
    /// This will:
    /// 1. Store the team account with owner tokens
    /// 2. Add the owner's OAuth to the auth store
    /// 3. Invite specified members (create member records)
    /// 4. Automate OAuth login for each member via email OTP
    pub async fn setup_team(
        &self,
        team_name: String,
        owner_tokens: TokenData,
        member_emails: Vec<String>,
    ) -> Result<SetupResult, AutomationError> {
        info!("Setting up team: {}", team_name);

        // Create team account
        let mut team = TeamAccount::new(team_name.clone(), owner_tokens.clone());

        // Add owner's OAuth to auth store
        let owner_email = owner_tokens.id_token.email.clone().unwrap_or_default();
        let _record_id = add_oauth_account(
            &self.codex_home,
            self.store_mode,
            owner_tokens,
            Some(Utc::now()),
            Some(format!("{} (Owner)", team_name)),
        )?;
        info!("Added owner OAuth account: {}", owner_email);

        // Invite members
        let mut member_results = Vec::new();
        for email in &member_emails {
            let member = team.invite_member(email.clone(), TeamRole::Member);
            info!("Invited member: {} (id: {})", email, member.id);
            member_results.push(MemberLoginResult {
                email: email.clone(),
                member_id: member.id.clone(),
                success: false,
                error: None,
            });
        }

        // Save team to store
        let team_id = team.id.clone();
        {
            let mut store = self.team_store.lock().await;
            store.add_team(team);
            store.save()?;
        }

        // Automate OAuth login for each member
        let mut logged_in = Vec::new();
        let mut failed = Vec::new();

        for (idx, email) in member_emails.iter().enumerate() {
            match self.login_member_via_otp(&team_id, email).await {
                Ok(tokens) => {
                    let member_id = member_results[idx].member_id.clone();

                    // Update member tokens in team store
                    {
                        let mut store = self.team_store.lock().await;
                        store.set_member_tokens(&team_id, &member_id, tokens.clone())?;
                        store.save()?;
                    }

                    // Add to auth store
                    let _ = add_oauth_account(
                        &self.codex_home,
                        self.store_mode,
                        tokens,
                        Some(Utc::now()),
                        Some(format!("{} ({})", email, team_name)),
                    )?;

                    info!("Successfully logged in member: {}", email);
                    member_results[idx].success = true;
                    logged_in.push(member_results[idx].clone());
                }
                Err(e) => {
                    warn!("Failed to login member {}: {}", email, e);
                    member_results[idx].error = Some(e.to_string());
                    failed.push(member_results[idx].clone());
                }
            }
        }

        Ok(SetupResult {
            team_id,
            logged_in_members: logged_in,
            failed_members: failed,
        })
    }

    /// Login a team member using email OTP.
    ///
    /// Complete flow:
    /// 1. Initiate OAuth login (sends OTP email)
    /// 2. Wait for and fetch OTP from email (via Cloudflare)
    /// 3. Complete OAuth with OTP
    /// 4. Return tokens
    pub async fn login_member_via_otp(
        &self,
        team_id: &str,
        email: &str,
    ) -> Result<TokenData, AutomationError> {
        info!("Starting OAuth login for member: {}", email);

        // Step 1: Initiate OAuth login (send OTP email)
        let verification = self.initiate_oauth_login(email).await?;
        info!("OTP email sent to: {}", email);

        // Step 2: Wait for OTP from email
        info!("Waiting for OTP for: {}", email);
        let otp = self.fetch_otp_from_email(email, 60).await?;
        info!("Received OTP for {}: {}", email, &otp[..4.min(otp.len())].to_string() + "***");

        // Step 3: Complete OAuth login with OTP
        let tokens = self.complete_oauth_login(email, &otp).await?;

        Ok(tokens)
    }

    /// Initiate OAuth login by requesting OTP email.
    async fn initiate_oauth_login(&self, email: &str) -> Result<PasswordlessStartResponse, AutomationError> {
        let url = format!("{}/passwordless/start", OAUTH_ISSUER);

        debug!("Sending passwordless start request to: {}", url);

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Origin", "https://chatgpt.com")
            .header("Referer", "https://chatgpt.com/")
            .json(&serde_json::json!({
                "client_id": OAUTH_CLIENT_ID,
                "client_secret": "",  // Public client
                "connection": "email",
                "email": email,
                "send": "code",
                "authParams": {
                    "scope": "openid email profile offline_access"
                }
            }))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to initiate login: {} - {}", status, body),
            });
        }

        let result: PasswordlessStartResponse = response.json().await.map_err(|e| {
            AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to parse response: {}", e),
            }
        })?;

        Ok(result)
    }

    /// Fetch OTP from email using Cloudflare Email Routing API.
    async fn fetch_otp_from_email(&self, email: &str, timeout_secs: u64) -> Result<String, AutomationError> {
        let start = std::time::Instant::now();
        let poll_interval = std::time::Duration::from_secs(5);
        let timeout = std::time::Duration::from_secs(timeout_secs);

        while start.elapsed() < timeout {
            // Try to get OTP from the worker first
            match self.otp_client.get_otp(email).await {
                Ok(otp) => return Ok(otp),
                Err(OtpError::NotFound(_)) => {
                    debug!("OTP not yet available, waiting...");
                }
                Err(e) => {
                    warn!("OTP fetch error: {}, retrying...", e);
                }
            }

            tokio::time::sleep(poll_interval).await;
        }

        Err(AutomationError::OtpTimeout(format!(
            "No OTP received for {} within {} seconds",
            email, timeout_secs
        )))
    }

    /// Complete OAuth login with the OTP code.
    async fn complete_oauth_login(
        &self,
        email: &str,
        otp: &str,
    ) -> Result<TokenData, AutomationError> {
        let url = format!("{}/oauth/token", OAUTH_ISSUER);

        debug!("Completing OAuth login for: {}", email);

        let response = self
            .http_client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("Origin", "https://chatgpt.com")
            .header("Referer", "https://chatgpt.com/")
            .json(&serde_json::json!({
                "grant_type": "http://auth0.com/oauth/grant-type/passwordless/otp",
                "client_id": OAUTH_CLIENT_ID,
                "username": email,
                "otp": otp,
                "realm": "email",
                "scope": "openid email profile offline_access"
            }))
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to complete login: {} - {}", status, body),
            });
        }

        let token_response: TokenResponse = response.json().await.map_err(|e| {
            AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to parse token response: {}", e),
            }
        })?;

        // Parse ID token to get claims
        let id_token = parse_chatgpt_jwt_claims(&token_response.id_token)
            .map_err(|e| AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to parse ID token: {}", e),
            })?;

        Ok(TokenData {
            id_token,
            access_token: token_response.access_token,
            refresh_token: token_response.refresh_token,
            account_id: None,
        })
    }

    /// Add a team with existing tokens (no OAuth flow).
    pub async fn add_team_with_tokens(
        &self,
        team_name: String,
        owner_tokens: TokenData,
    ) -> Result<String, AutomationError> {
        let team = TeamAccount::new(team_name.clone(), owner_tokens.clone());

        // Add owner's OAuth to auth store
        let _record_id = add_oauth_account(
            &self.codex_home,
            self.store_mode,
            owner_tokens,
            Some(Utc::now()),
            Some(format!("{} (Owner)", team_name)),
        )?;

        let team_id = team.id.clone();
        {
            let mut store = self.team_store.lock().await;
            store.add_team(team);
            store.save()?;
        }

        info!("Created team: {} (id: {})", team_name, team_id);
        Ok(team_id)
    }

    /// Invite a member to a team (without auto-login).
    pub async fn invite_member(
        &self,
        team_id: &str,
        email: String,
        role: TeamRole,
    ) -> Result<TeamMember, AutomationError> {
        let mut store = self.team_store.lock().await;

        let team = store.get_team_mut(team_id).ok_or_else(|| {
            AutomationError::TeamNotFound(team_id.to_string())
        })?;

        let member = team.invite_member(email.clone(), role);
        store.save()?;

        info!("Invited {} to team {}", email, team_id);
        Ok(member)
    }

    /// Login all pending members in a team.
    pub async fn login_pending_members(&self, team_id: &str) -> Result<Vec<MemberLoginResult>, AutomationError> {
        let (team_id_clone, pending) = {
            let store = self.team_store.lock().await;
            let team = store.get_team(team_id).ok_or_else(|| {
                AutomationError::TeamNotFound(team_id.to_string())
            })?;
            let pending: Vec<_> = team.pending_members()
                .iter()
                .map(|m| (m.id.clone(), m.email.clone()))
                .collect();
            (team_id.to_string(), pending)
        };

        let mut results = Vec::new();
        for (member_id, email) in pending {
            match self.login_member_via_otp(&team_id_clone, &email).await {
                Ok(tokens) => {
                    // Update member tokens
                    {
                        let mut store = self.team_store.lock().await;
                        store.set_member_tokens(&team_id_clone, &member_id, tokens.clone())?;
                        store.save()?;
                    }

                    // Add to auth store
                    let team = self.team_store.lock().await.get_team(&team_id_clone).cloned();
                    let team_name = team.map(|t| t.name).unwrap_or_default();
                    let _ = add_oauth_account(
                        &self.codex_home,
                        self.store_mode,
                        tokens,
                        Some(Utc::now()),
                        Some(format!("{} ({})", email, team_name)),
                    )?;

                    results.push(MemberLoginResult {
                        email,
                        member_id,
                        success: true,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(MemberLoginResult {
                        email,
                        member_id,
                        success: false,
                        error: Some(e.to_string()),
                    });
                }
            }
        }

        Ok(results)
    }

    /// List all teams.
    pub async fn list_teams(&self) -> Vec<TeamAccount> {
        let store = self.team_store.lock().await;
        store.list_teams().into_iter().cloned().collect()
    }

    /// Get a team by ID.
    pub async fn get_team(&self, team_id: &str) -> Option<TeamAccount> {
        let store = self.team_store.lock().await;
        store.get_team(team_id).cloned()
    }

    /// Remove a team.
    pub async fn remove_team(&self, team_id: &str) -> Result<bool, AutomationError> {
        let mut store = self.team_store.lock().await;
        let removed = store.remove_team(team_id).is_some();
        if removed {
            store.save()?;
        }
        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_add_team_with_tokens() {
        let temp_dir = TempDir::new().unwrap();
        let automation = TeamOAuthAutomation::new(
            temp_dir.path().to_path_buf(),
            AuthCredentialsStoreMode::File,
            None,
        ).unwrap();

        assert!(automation.list_teams().await.is_empty());
    }
}
