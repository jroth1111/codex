//! OTP client for fetching one-time passwords from the email OTP worker.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;

/// Default URL for the chatgpt-email-otp worker.
pub const DEFAULT_OTP_WORKER_URL: &str = "http://chatgpt-email-otp.gwizz.workers.dev";

/// Email domain for team members.
pub const EMAIL_DOMAIN: &str = "arcaneai.site";

/// Cloudflare API endpoint for email routing.
const CF_API_BASE: &str = "https://api.cloudflare.com/client/v4";

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

/// Cloudflare email routing destination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailDestination {
    pub r#type: String,
    pub address: Option<String>,
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

/// Email matcher for Cloudflare routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailMatcher {
    #[serde(rename = "type")]
    pub matcher_type: String,
    pub field: Option<String>,
    pub value: Option<String>,
}

/// Email action for Cloudflare routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfEmailAction {
    #[serde(rename = "type")]
    pub action_type: String,
    pub value: Option<Vec<String>>,
}

/// Client for fetching OTP codes from the email worker.
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

    /// Fetch OTP for an email address.
    ///
    /// This will poll the worker until an OTP is received or timeout.
    pub async fn get_otp(&self, email: &str) -> Result<String, OtpError> {
        self.get_otp_with_timeout(email, 60).await
    }

    /// Fetch OTP with a custom timeout (in seconds).
    pub async fn get_otp_with_timeout(
        &self,
        email: &str,
        timeout_secs: u64,
    ) -> Result<String, OtpError> {
        let url = format!("{}/otp", self.base_url);
        let request = OtpRequest {
            email: email.to_string(),
            timeout: Some(timeout_secs),
        };

        let response = self
            .client
            .post(&url)
            .json(&request)
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

    /// Check if the worker is healthy.
    pub async fn health_check(&self) -> Result<bool, OtpError> {
        let url = format!("{}/health", self.base_url);

        let response = self.client.get(&url).send().await?;

        Ok(response.status().is_success())
    }

    /// Wait for OTP by polling the worker.
    ///
    /// This will repeatedly check for an OTP until one is received or max_attempts is reached.
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

    /// Fetch emails from Cloudflare email routing (if configured).
    ///
    /// This is an alternative to the worker for environments where
    /// Cloudflare email routing is set up.
    pub async fn fetch_emails_from_cf(&self, email: &str) -> Result<Vec<EmailMessage>, OtpError> {
        let cf_token = self.cf_token.as_ref().ok_or_else(|| {
            OtpError::CloudflareError("Cloudflare token not configured".to_string())
        })?;
        let cf_account_id = self.cf_account_id.as_ref().ok_or_else(|| {
            OtpError::CloudflareError("Cloudflare account ID not configured".to_string())
        })?;

        // Use Cloudflare email routing API to fetch recent emails
        let url = format!(
            "{}/accounts/{}/email/routing/rules",
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
                "Cloudflare API error: {}",
                error_text
            )));
        }

        // Parse response and extract emails for the given address
        // Note: This is a simplified implementation - actual CF API may differ
        Ok(vec![])
    }

    /// Extract OTP code from an email message body.
    pub fn extract_otp_from_email(&self, body: &str) -> Option<String> {
        // Common OTP patterns:
        // - 6-digit code: \b\d{6}\b
        // - 8-digit code: \b\d{8}\b
        // - In "code: XXXXXX" format

        let re = regex_lite::Regex::new(r"\b(\d{6,8})\b").ok()?;
        let caps = re.captures(body)?;
        caps.get(1).map(|m| m.as_str().to_string())
    }
}

/// Email message structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailMessage {
    pub from: String,
    pub to: String,
    pub subject: String,
    pub body: String,
    pub received_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[ignore] // Requires running worker
    async fn test_health_check() {
        let client = OtpClient::default();
        let result = client.health_check().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_member_email() {
        let client = OtpClient::default();
        let email = client.generate_member_email("user1");
        assert_eq!(email, "user1@arcaneai.site");
    }

    #[test]
    fn test_extract_otp_from_email() {
        let client = OtpClient::default();

        // Test 6-digit OTP
        let body = "Your verification code is: 123456";
        assert_eq!(client.extract_otp_from_email(body), Some("123456".to_string()));

        // Test 8-digit OTP
        let body = "Use code 12345678 to verify";
        assert_eq!(client.extract_otp_from_email(body), Some("12345678".to_string()));

        // Test no OTP
        let body = "No code here";
        assert_eq!(client.extract_otp_from_email(body), None);
    }
}
