//! Team OAuth automation for managing team accounts and member logins.

use crate::auth::{add_oauth_account, AuthCredentialsStoreMode};
use crate::team_oauth::otp_client::{OtpClient, OtpError};
use crate::team_oauth::team_store::{TeamAccount, TeamMember, TeamRole, TeamStore};
use crate::token_data::TokenData;
use chrono::Utc;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

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

    /// Set up a team with the given owner tokens and invite members.
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
        let mut team = TeamAccount::new(team_name, owner_tokens.clone());

        // Add owner's OAuth to auth store
        let owner_email = owner_tokens.id_token.email.clone().unwrap_or_default();
        add_oauth_account(
            &self.codex_home,
            self.store_mode,
            owner_tokens.clone(),
            Some(Utc::now()),
            Some(format!("{} (Owner)", team.name)),
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
            store.add_team(team.clone());
            store.save()?;
        }

        // Automate OAuth login for each member
        let mut logged_in = Vec::new();
        let mut failed = Vec::new();

        for (idx, email) in member_emails.iter().enumerate() {
            match self.login_member_via_otp(&team_id, email).await {
                Ok(tokens) => {
                    // Update member tokens in team store
                    let member_id = member_results[idx].member_id.clone();
                    {
                        let mut store = self.team_store.lock().await;
                        store.set_member_tokens(&team_id, &member_id, tokens.clone())?;
                        store.save()?;
                    }

                    // Add to auth store
                    add_oauth_account(
                        &self.codex_home,
                        self.store_mode,
                        tokens,
                        Some(Utc::now()),
                        Some(format!("{} ({})", email, team.name)),
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
    /// This initiates the OAuth flow and waits for OTP from email.
    pub async fn login_member_via_otp(
        &self,
        team_id: &str,
        email: &str,
    ) -> Result<TokenData, AutomationError> {
        info!("Starting OAuth login for member: {}", email);

        // Initiate OAuth login (send OTP email)
        self.initiate_oauth_login(email).await?;

        // Wait for OTP from email worker
        info!("Waiting for OTP for: {}", email);
        let otp = self.otp_client.wait_for_otp(email, 5, 12).await?;
        info!("Received OTP for: {}", email);

        // Complete OAuth login with OTP
        let tokens = self.complete_oauth_login(email, &otp).await?;

        Ok(tokens)
    }

    /// Initiate OAuth login by requesting OTP email.
    async fn initiate_oauth_login(&self, email: &str) -> Result<(), AutomationError> {
        // Call ChatGPT's passwordless/login endpoint to send OTP email
        let url = format!("{}/passwordless/start", OAUTH_ISSUER);

        let response = self
            .http_client
            .post(&url)
            .form(&[
                ("client_id", OAUTH_CLIENT_ID),
                ("connection", "email"),
                ("email", email),
                ("send", "link"),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(AutomationError::OAuthFailed {
                email: email.to_string(),
                reason: format!("Failed to initiate login: {} - {}", status, body),
            });
        }

        Ok(())
    }

    /// Complete OAuth login with the OTP code.
    async fn complete_oauth_login(
        &self,
        email: &str,
        otp: &str,
    ) -> Result<TokenData, AutomationError> {
        let url = format!("{}/oauth/token", OAUTH_ISSUER);

        let response = self
            .http_client
            .post(&url)
            .form(&[
                ("grant_type", "http://auth0.com/oauth/grant-type/passwordless/otp"),
                ("client_id", OAUTH_CLIENT_ID),
                ("username", email),
                ("otp", otp),
                ("realm", "email"),
                ("scope", "openid email profile offline_access"),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
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
        let id_token = crate::token_data::parse_chatgpt_jwt_claims(&token_response.id_token)
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
        add_oauth_account(
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

        // This would need actual tokens to work
        // Just testing the structure
        assert!(automation.list_teams().await.is_empty());
    }
}
