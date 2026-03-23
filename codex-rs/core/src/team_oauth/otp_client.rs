//! OTP client for fetching one-time passwords from email.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Default URL for the chatgpt-email-otp worker.
pub const DEFAULT_OTP_WORKER_URL: &str = "http://chatgpt-email-otp.gwizz.workers.dev";

/// Email domain for team members.
pub const EMAIL_DOMAIN: &str = "arcaneai.site";

/// Cloudflare API endpoint.
const CF_API_BASE: &str = "https://api.cloudflare.com/client/v4";

/// Cloudflare account ID for email routing.
const CF_ACCOUNT_ID: &str = "81f96723a96b1ae6b3f2f2cc7693be33";

/// Errors from the OTP client.
#[derive(Debug, Error)]
pub enum OtpError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("OTP not found for email: {0}")]
    NotFound(String),

    #[error("OTP request timed out")]
    Timeout,

    #[error("Worker returned error: {0}")]
    WorkerError(String),

    #[error("Failed to parse response: {0}")]
    ParseError(String),

    #[error("Cloudflare API error: {0}")]
    CloudflareError(String),

    #[error("Email parsing error: {0}")]
    EmailError(String),

    #[error("No emails found")]
    NoEmails,
}

/// Response from the OTP worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtpResponse {
    /// The OTP code
    pub otp: String,
    /// When the OTP was received
    pub received_at: Option<String>,
    /// Email address the OTP was sent to
    pub email: String,
}

/// Request to fetch OTP for an email.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtpRequest {
    /// Email address to fetch OTP for
    pub email: String,
    /// Optional timeout in seconds to wait for OTP
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
}

/// Email message from Cloudflare.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    pub id: String,
    pub from: String,
    pub to: String,
    pub subject: String,
    pub text: Option<String>,
    pub html: Option<String>,
    pub created_at: String,
}

/// Cloudflare API response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CfResponse<T> {
    success: bool,
    errors: Vec<CfError>,
    result: Option<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CfError {
    code: i32,
    message: String,
}

/// Cloudflare email routing rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailRule {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub matchers: Vec<CfEmailMatcher>,
    pub actions: Vec<CfEmailAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailMatcher {
    #[serde(rename = "type")]
    pub matcher_type: String,
    pub field: Option<String>,
    pub value: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailAction {
    #[serde(rename = "type")]
    pub action_type: String,
    pub value: Option<Vec<String>>,
}

/// Client for fetching OTP codes from email.
#[derive(Debug, Clone)]
pub struct OtpClient {
    client: Client,
    base_url: String,
    cf_token: Option<String>,
    cf_account_id: Option<String>,
}

impl Default for OtpClient {
    fn default() -> Self {
        Self::new(DEFAULT_OTP_WORKER_URL.to_string())
    }
}

impl OtpClient {
    /// Default worker URL for reference.
    pub const DEFAULT_URL: &'static str = DEFAULT_OTP_WORKER_URL;

    /// Create a new OTP client with the given worker URL.
    pub fn new(base_url: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url,
            cf_token: None,
            cf_account_id: None,
        }
    }

    /// Create an OTP client with Cloudflare API access for email routing.
    pub fn with_cloudflare(base_url: String, cf_token: String, cf_account_id: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url,
            cf_token: Some(cf_token),
            cf_account_id: Some(cf_account_id),
        }
    }

    /// Create with hardcoded Cloudflare credentials.
    pub fn with_default_cloudflare() -> Self {
        // Use the provided Cloudflare token
        Self::with_cloudflare(
            DEFAULT_OTP_WORKER_URL.to_string(),
            "cfat_wCOFVocsOfJ7lgZhWwjdG0tGxkhJeWRvxgImawmt053481b0".to_string(),
            CF_ACCOUNT_ID.to_string(),
        )
    }

    /// Fetch OTP for an email address.
    pub async fn get_otp(&self, email: &str) -> Result<String, OtpError> {
        // Try Cloudflare first if configured
        if self.cf_token.is_some() {
            match self.get_otp_from_cloudflare(email).await {
                Ok(otp) => return Ok(otp),
                Err(e) => {
                    tracing::debug!("Cloudflare OTP fetch failed: {}, trying worker...", e);
                }
            }
        }

        // Fall back to worker
        self.get_otp_from_worker(email, 30).await
    }

    /// Get OTP from the worker endpoint.
    async fn get_otp_from_worker(&self, email: &str, timeout_secs: u64) -> Result<String, OtpError> {
        let url = format!("{}/otp", self.base_url);
        let request = OtpRequest {
            email: email.to_string(),
            timeout: Some(timeout_secs),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(Duration::from_secs(timeout_secs + 10))
            .send()
            .await?;

        let status = response.status();
        if status.is_client_error() || status.is_server_error() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OtpError::WorkerError(format!(
                "Worker returned {}: {}",
                status, error_text
            )));
        }

        let otp_response: OtpResponse = response
            .json()
            .await
            .map_err(|e| OtpError::ParseError(e.to_string()))?;

        if otp_response.otp.is_empty() {
            return Err(OtpError::NotFound(email.to_string()));
        }

        Ok(otp_response.otp)
    }

    /// Get OTP from Cloudflare Email Routing.
    async fn get_otp_from_cloudflare(&self, email: &str) -> Result<String, OtpError> {
        let cf_token = self.cf_token.as_ref().ok_or_else(|| {
            OtpError::CloudflareError("Cloudflare token not configured".to_string())
        })?;
        let cf_account_id = self.cf_account_id.as_ref().ok_or_else(|| {
            OtpError::CloudflareError("Cloudflare account ID not configured".to_string())
        })?;

        // Fetch recent emails from Cloudflare Workers KV or Email Routing logs
        // Note: Cloudflare Email Routing doesn't have a direct API to read emails,
        // but we can use Workers KV to store emails if the worker is set up to do so.

        // For now, try to fetch from a custom endpoint that the worker might expose
        let url = format!(
            "{}/accounts/{}/storage/kv/namespaces",
            CF_API_BASE, cf_account_id
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", cf_token))
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(OtpError::CloudflareError(format!(
                "Failed to list KV namespaces: {}",
                error_text
            )));
        }

        // Try to fetch emails from KV namespace
        // The worker should store emails with key pattern: "email:{email}:{timestamp}"
        let emails = self.fetch_emails_from_kv(email, cf_token, cf_account_id).await?;

        // Find the most recent OTP email
        for email_msg in emails.iter().rev() {
            if let Some(text) = &email_msg.text {
                if let Some(otp) = self.extract_otp_from_email(text) {
                    return Ok(otp);
                }
            }
            if let Some(html) = &email_msg.html {
                if let Some(otp) = self.extract_otp_from_email(html) {
                    return Ok(otp);
                }
            }
        }

        Err(OtpError::NotFound(email.to_string()))
    }

    /// Fetch emails from Cloudflare KV storage.
    async fn fetch_emails_from_kv(
        &self,
        email: &str,
        cf_token: &str,
        cf_account_id: &str,
    ) -> Result<Vec<EmailMessage>, OtpError> {
        // First, find the KV namespace ID for emails
        let namespace_id = self.find_email_namespace(cf_token, cf_account_id).await?;

        // List keys with prefix "email:{email}:"
        let prefix = format!("email:{}:", email);
        let url = format!(
            "{}/accounts/{}/storage/kv/namespaces/{}/keys?prefix={}",
            CF_API_BASE, cf_account_id, namespace_id, prefix
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", cf_token))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(OtpError::CloudflareError("Failed to list email keys".to_string()));
        }

        #[derive(Deserialize)]
        struct KeyResult {
            name: String,
        }
        #[derive(Deserialize)]
        struct KeysResponse {
            result: Vec<KeyResult>,
        }

        let keys_response: KeysResponse = response.json().await.map_err(|e| {
            OtpError::ParseError(format!("Failed to parse keys response: {}", e))
        })?;

        let mut emails = Vec::new();
        for key in keys_response.result.iter().take(10) {
            let value_url = format!(
                "{}/accounts/{}/storage/kv/namespaces/{}/values/{}",
                CF_API_BASE, cf_account_id, namespace_id, key.name
            );

            if let Ok(response) = self
                .client
                .get(&value_url)
                .header("Authorization", format!("Bearer {}", cf_token))
                .send()
                .await
            {
                if response.status().is_success() {
                    if let Ok(email) = response.json::<EmailMessage>().await {
                        emails.push(email);
                    }
                }
            }
        }

        Ok(emails)
    }

    /// Find the KV namespace ID for email storage.
    async fn find_email_namespace(&self, cf_token: &str, cf_account_id: &str) -> Result<String, OtpError> {
        let url = format!(
            "{}/accounts/{}/storage/kv/namespaces",
            CF_API_BASE, cf_account_id
        );

        let response = self
            .client
            .get(&url)
            .header("Authorization", format!("Bearer {}", cf_token))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(OtpError::CloudflareError("Failed to list KV namespaces".to_string()));
        }

        #[derive(Deserialize)]
        struct Namespace {
            id: String,
            title: String,
        }
        #[derive(Deserialize)]
        struct NamespacesResponse {
            result: Vec<Namespace>,
        }

        let ns_response: NamespacesResponse = response.json().await.map_err(|e| {
            OtpError::ParseError(format!("Failed to parse namespaces response: {}", e))
        })?;

        // Look for a namespace that might contain emails
        for ns in ns_response.result {
            if ns.title.to_lowercase().contains("email")
                || ns.title.to_lowercase().contains("otp")
                || ns.title.to_lowercase().contains("chatgpt")
            {
                return Ok(ns.id);
            }
        }

        // Return the first namespace if no match found
        ns_response
            .result
            .first()
            .map(|ns| ns.id.clone())
            .ok_or_else(|| OtpError::CloudflareError("No KV namespaces found".to_string()))
    }

    /// Check if the worker is healthy.
    pub async fn health_check(&self) -> Result<bool, OtpError> {
        let url = format!("{}/health", self.base_url);

        let response = self.client.get(&url).send().await?;

        Ok(response.status().is_success())
    }

    /// Wait for OTP by polling.
    pub async fn wait_for_otp(
        &self,
        email: &str,
        poll_interval_secs: u64,
        max_attempts: u32,
    ) -> Result<String, OtpError> {
        for attempt in 0..max_attempts {
            match self.get_otp(email).await {
                Ok(otp) => return Ok(otp),
                Err(OtpError::NotFound(_)) => {
                    if attempt + 1 < max_attempts {
                        tracing::debug!(
                            "OTP not found for {} (attempt {}/{}), waiting {}s...",
                            email,
                            attempt + 1,
                            max_attempts,
                            poll_interval_secs
                        );
                        tokio::time::sleep(Duration::from_secs(poll_interval_secs)).await;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(OtpError::NotFound(format!(
            "No OTP received after {} attempts for {}",
            max_attempts, email
        )))
    }

    /// Generate a team member email address using the configured domain.
    pub fn generate_member_email(&self, prefix: &str) -> String {
        format!("{}@{}", prefix, EMAIL_DOMAIN)
    }

    /// Extract OTP code from an email message body.
    pub fn extract_otp_from_email(&self, body: &str) -> Option<String> {
        // Common OTP patterns:
        // - 6-digit code: \b\d{6}\b
        // - 8-digit code: \b\d{8}\b
        // - In "code: XXXXXX" format
        // - OpenAI specific patterns

        // Try OpenAI specific pattern first
        let openai_patterns = [
            r"verification code[:\s]+(\d{6})",
            r"code[:\s]+(\d{6})",
            r"your code is[:\s]+(\d{6})",
            r"enter[:\s]+(\d{6})",
        ];

        for pattern in &openai_patterns {
            if let Ok(re) = regex_lite::Regex::new(pattern) {
                if let Some(caps) = re.captures(body) {
                    if let Some(code) = caps.get(1) {
                        return Some(code.as_str().to_string());
                    }
                }
            }
        }

        // Fall back to any 6-8 digit code
        let re = regex_lite::Regex::new(r"\b(\d{6,8})\b").ok()?;
        let caps = re.captures(body)?;
        caps.get(1).map(|m| m.as_str().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_member_email() {
        let client = OtpClient::default();
        let email = client.generate_member_email("user1");
        assert_eq!(email, "user1@arcaneai.site");
    }

    #[test]
    fn test_extract_otp_from_email() {
        let client = OtpClient::default();

        // Test OpenAI patterns
        let body = "Your verification code is: 123456";
        assert_eq!(client.extract_otp_from_email(body), Some("123456".to_string()));

        let body = "Enter code: 654321 to continue";
        assert_eq!(client.extract_otp_from_email(body), Some("654321".to_string()));

        let body = "Your code is 999888";
        assert_eq!(client.extract_otp_from_email(body), Some("999888".to_string()));

        // Test 8-digit OTP
        let body = "Use code 12345678 to verify";
        assert_eq!(client.extract_otp_from_email(body), Some("12345678".to_string()));

        // Test no OTP
        let body = "No code here";
        assert_eq!(client.extract_otp_from_email(body), None);
    }
}
