//! Team OAuth automation module.
//!
//! This module provides automation for managing team accounts:
//! - Store team OAuth tokens
//! - Invite team members
//! - Automate OAuth login for members via email OTP

mod automation;
mod otp_client;
mod team_store;

pub use automation::TeamOAuthAutomation;
pub use otp_client::OtpClient;
pub use otp_client::OtpError;
pub use team_store::TeamAccount;
pub use team_store::TeamMember;
pub use team_store::TeamRole;
pub use team_store::TeamStore;
