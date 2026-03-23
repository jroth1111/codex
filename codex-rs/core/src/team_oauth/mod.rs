//! Team OAuth automation module.
//!
//! This module provides automation for managing team accounts:
//! - Store team OAuth tokens
//! - Invite team members
//! - Automate OAuth login for members via email OTP
//!
//! ## Usage
//!
//! ```rust,ignore
//! use codex_core::team_oauth::{TeamOAuthAutomation, TeamRole};
//! use codex_core::auth::AuthCredentialsStoreMode;
//!
//! async fn setup_team() {
//!     let automation = TeamOAuthAutomation::with_cloudflare(
//!         codex_home,
//!         AuthCredentialsStoreMode::Auto,
//!         "your-cf-token".to_string(),
//!         "your-cf-account-id".to_string(),
//!     ).unwrap();
//!
//!     let result = automation.setup_team(
//!         "My Team".to_string(),
//!         owner_tokens,
//!         vec!["user1@arcaneai.site".to_string()],
//!     ).await.unwrap();
//! }
//! ```

mod automation;
mod otp_client;
mod team_store;

pub use automation::{AutomationError, MemberLoginResult, SetupResult, TeamOAuthAutomation};
pub use otp_client::{EmailMessage, OtpClient, OtpError, OtpRequest, OtpResponse};
pub use team_store::{TeamAccount, TeamMember, TeamRole, TeamStore};
