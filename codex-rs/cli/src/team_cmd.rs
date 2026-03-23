//! Team OAuth CLI commands.

use clap::{Args, Parser, Subcommand};
use codex_core::auth::AuthCredentialsStoreMode;
use codex_core::config::find_codex_home;
use codex_core::team_oauth::{
    OtpClient, TeamOAuthAutomation, TeamRole,
};
use codex_core::token_data::{TokenData, parse_chatgpt_jwt_claims};
use std::io::{self, BufRead};
use std::path::PathBuf;

/// Cloudflare account ID for email routing.
const CF_ACCOUNT_ID: &str = "81f96723a96b1ae6b3f2f2cc7693be33";

/// Manage team accounts and member OAuth tokens.
#[derive(Debug, Parser)]
pub struct TeamCommand {
    #[command(subcommand)]
    subcommand: TeamSubcommand,

    /// Override OTP worker URL (default: http://chatgpt-email-otp.gwizz.workers.dev)
    #[arg(long = "otp-worker", value_name = "URL", global = true)]
    otp_worker_url: Option<String>,

    /// Cloudflare API token for email access
    #[arg(long = "cf-token", value_name = "TOKEN", global = true, env = "CF_API_TOKEN")]
    cf_token: Option<String>,

    /// CODEX_HOME directory (default: ~/.codex)
    #[arg(long = "codex-home", value_name = "DIR", global = true, hide = true)]
    codex_home: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum TeamSubcommand {
    /// Create a new team with the owner's OAuth tokens.
    Create(CreateTeamArgs),

    /// List all teams.
    List,

    /// Show details for a team.
    Show(ShowTeamArgs),

    /// Remove a team.
    Remove(RemoveTeamArgs),

    /// Invite a member to a team.
    Invite(InviteMemberArgs),

    /// Automate OAuth login for team members via email OTP.
    Login(LoginMembersArgs),

    /// Full setup: create team, invite members, and auto-login.
    Setup(SetupTeamArgs),

    /// Test OTP client connection.
    TestOtp,
}

#[derive(Debug, Args)]
struct CreateTeamArgs {
    /// Team name
    #[arg(value_name = "NAME")]
    name: String,

    /// Access token for the team owner
    #[arg(long = "access-token", value_name = "TOKEN")]
    access_token: Option<String>,

    /// Refresh token for the team owner
    #[arg(long = "refresh-token", value_name = "TOKEN")]
    refresh_token: Option<String>,

    /// ID token (JWT) for the team owner
    #[arg(long = "id-token", value_name = "TOKEN")]
    id_token: Option<String>,

    /// Read tokens from stdin as JSON
    #[arg(long = "stdin")]
    stdin: bool,
}

#[derive(Debug, Args)]
struct ShowTeamArgs {
    /// Team ID
    #[arg(value_name = "TEAM_ID")]
    team_id: String,
}

#[derive(Debug, Args)]
struct RemoveTeamArgs {
    /// Team ID
    #[arg(value_name = "TEAM_ID")]
    team_id: String,
}

#[derive(Debug, Args)]
struct InviteMemberArgs {
    /// Team ID
    #[arg(value_name = "TEAM_ID")]
    team_id: String,

    /// Member email address
    #[arg(value_name = "EMAIL")]
    email: String,

    /// Member role (admin or member)
    #[arg(long = "role", value_name = "ROLE", default_value = "member")]
    role: String,
}

#[derive(Debug, Args)]
struct LoginMembersArgs {
    /// Team ID
    #[arg(value_name = "TEAM_ID")]
    team_id: String,

    /// Specific member email to login (if omitted, logs in all pending members)
    #[arg(value_name = "EMAIL")]
    email: Option<String>,
}

#[derive(Debug, Args)]
struct SetupTeamArgs {
    /// Team name
    #[arg(value_name = "NAME")]
    name: String,

    /// Owner's access token
    #[arg(long = "access-token", value_name = "TOKEN")]
    access_token: Option<String>,

    /// Owner's refresh token
    #[arg(long = "refresh-token", value_name = "TOKEN")]
    refresh_token: Option<String>,

    /// Owner's ID token (JWT)
    #[arg(long = "id-token", value_name = "TOKEN")]
    id_token: Option<String>,

    /// Read owner tokens from stdin as JSON
    #[arg(long = "stdin")]
    stdin: bool,

    /// Member email addresses to invite and auto-login
    #[arg(value_name = "EMAILS")]
    member_emails: Vec<String>,
}

impl TeamCommand {
    pub async fn run(self) -> anyhow::Result<()> {
        let codex_home = match self.codex_home {
            Some(path) => path,
            None => find_codex_home()?,
        };

        let automation = match &self.cf_token {
            Some(token) => {
                TeamOAuthAutomation::with_cloudflare(
                    codex_home,
                    AuthCredentialsStoreMode::Auto,
                    self.otp_worker_url.clone(),
                    token.clone(),
                    CF_ACCOUNT_ID.to_string(),
                )?
            }
            None => {
                TeamOAuthAutomation::new(
                    codex_home,
                    AuthCredentialsStoreMode::Auto,
                    self.otp_worker_url.clone(),
                )?
            }
        };

        match self.subcommand {
            TeamSubcommand::Create(args) => run_create_team(&automation, args).await,
            TeamSubcommand::List => run_list_teams(&automation).await,
            TeamSubcommand::Show(args) => run_show_team(&automation, args).await,
            TeamSubcommand::Remove(args) => run_remove_team(&automation, args).await,
            TeamSubcommand::Invite(args) => run_invite_member(&automation, args).await,
            TeamSubcommand::Login(args) => run_login_members(&automation, args).await,
            TeamSubcommand::Setup(args) => run_setup_team(&automation, args).await,
            TeamSubcommand::TestOtp => run_test_otp(&self).await,
        }
    }
}

async fn run_create_team(automation: &TeamOAuthAutomation, args: CreateTeamArgs) -> anyhow::Result<()> {
    let tokens = read_tokens(&args.access_token, &args.refresh_token, &args.id_token, args.stdin)?;
    let team_id = automation.add_team_with_tokens(args.name, tokens).await?;

    eprintln!("Created team: {}", team_id);
    println!("{}", team_id);
    Ok(())
}

async fn run_list_teams(automation: &TeamOAuthAutomation) -> anyhow::Result<()> {
    let teams = automation.list_teams().await;

    if teams.is_empty() {
        eprintln!("No teams found.");
        return Ok(());
    }

    eprintln!("Teams:");
    for team in teams {
        let active = team.active_members().len();
        let pending = team.pending_members().len();
        eprintln!(
            "  {} ({}) - {} active, {} pending",
            team.name, team.id, active, pending
        );
    }
    Ok(())
}

async fn run_show_team(automation: &TeamOAuthAutomation, args: ShowTeamArgs) -> anyhow::Result<()> {
    let team = automation.get_team(&args.team_id).await
        .ok_or_else(|| anyhow::anyhow!("Team not found: {}", args.team_id))?;

    eprintln!("Team: {} ({})", team.name, team.id);
    eprintln!("Created: {}", team.created_at.format("%Y-%m-%d %H:%M:%S UTC"));
    eprintln!();

    let owner_email = team.owner_tokens.id_token.email.as_deref().unwrap_or("unknown");
    eprintln!("Owner: {}", owner_email);

    if !team.members.is_empty() {
        eprintln!();
        eprintln!("Members:");
        for member in &team.members {
            let status = if member.tokens.is_some() { "active" } else { "pending" };
            eprintln!(
                "  {} [{}] - {}",
                member.email,
                status,
                member.role.to_string().to_lowercase()
            );
        }
    }
    Ok(())
}

async fn run_remove_team(automation: &TeamOAuthAutomation, args: RemoveTeamArgs) -> anyhow::Result<()> {
    let removed = automation.remove_team(&args.team_id).await?;

    if removed {
        eprintln!("Removed team: {}", args.team_id);
    } else {
        eprintln!("Team not found: {}", args.team_id);
    }
    Ok(())
}

async fn run_invite_member(automation: &TeamOAuthAutomation, args: InviteMemberArgs) -> anyhow::Result<()> {
    let role = match args.role.to_lowercase().as_str() {
        "owner" => TeamRole::Owner,
        "admin" => TeamRole::Admin,
        "member" => TeamRole::Member,
        _ => anyhow::bail!("Invalid role: {}. Use 'owner', 'admin', or 'member'.", args.role),
    };

    let member = automation.invite_member(&args.team_id, args.email.clone(), role).await?;

    eprintln!("Invited {} to team {} (member_id: {})", args.email, args.team_id, member.id);
    Ok(())
}

async fn run_login_members(automation: &TeamOAuthAutomation, args: LoginMembersArgs) -> anyhow::Result<()> {
    if let Some(email) = &args.email {
        // Login specific member
        eprintln!("Logging in member: {}", email);
        match automation.login_member_via_otp(&args.team_id, email).await {
            Ok(tokens) => {
                let member_email = tokens.id_token.email.as_deref().unwrap_or(email);
                eprintln!("✓ Successfully logged in: {}", member_email);
            }
            Err(e) => {
                eprintln!("✗ Failed to login {}: {}", email, e);
                return Err(e.into());
            }
        }
    } else {
        // Login all pending members
        let results = automation.login_pending_members(&args.team_id).await?;

        if results.is_empty() {
            eprintln!("No pending members to log in.");
            return Ok(());
        }

        eprintln!("Login results:");
        for result in &results {
            if result.success {
                eprintln!("  ✓ {}", result.email);
            } else {
                eprintln!("  ✗ {} - {}", result.email, result.error.as_deref().unwrap_or("unknown error"));
            }
        }
    }

    Ok(())
}

async fn run_setup_team(automation: &TeamOAuthAutomation, args: SetupTeamArgs) -> anyhow::Result<()> {
    if args.member_emails.is_empty() {
        anyhow::bail!("At least one member email is required for setup.");
    }

    let owner_tokens = read_tokens(&args.access_token, &args.refresh_token, &args.id_token, args.stdin)?;

    eprintln!("Setting up team '{}' with {} member(s)...", args.name, args.member_emails.len());

    let result = automation.setup_team(
        args.name,
        owner_tokens,
        args.member_emails.clone(),
    ).await?;

    eprintln!();
    eprintln!("Team ID: {}", result.team_id);

    if !result.logged_in_members.is_empty() {
        eprintln!();
        eprintln!("Successfully logged in {} member(s):", result.logged_in_members.len());
        for member in &result.logged_in_members {
            eprintln!("  ✓ {}", member.email);
        }
    }

    if !result.failed_members.is_empty() {
        eprintln!();
        eprintln!("Failed to login {} member(s):", result.failed_members.len());
        for member in &result.failed_members {
            eprintln!("  ✗ {} - {}", member.email, member.error.as_deref().unwrap_or("unknown error"));
        }
    }

    Ok(())
}

async fn run_test_otp(cmd: &TeamCommand) -> anyhow::Result<()> {
    let otp_client = match &cmd.cf_token {
        Some(token) => {
            OtpClient::with_cloudflare(
                cmd.otp_worker_url.clone().unwrap_or_else(|| OtpClient::DEFAULT_URL.to_string()),
                token.clone(),
                CF_ACCOUNT_ID.to_string(),
            )
        }
        None => {
            OtpClient::new(cmd.otp_worker_url.clone().unwrap_or_else(|| OtpClient::DEFAULT_URL.to_string()))
        }
    };

    eprintln!("Testing OTP client connection...");

    // Test worker health
    match otp_client.health_check().await {
        Ok(true) => eprintln!("  ✓ Worker health check passed"),
        Ok(false) => eprintln!("  ✗ Worker health check failed"),
        Err(e) => eprintln!("  ✗ Worker health check error: {}", e),
    }

    // Test email generation
    let test_email = otp_client.generate_member_email("test");
    eprintln!("  Generated email: {}", test_email);

    // Test OTP extraction
    let test_body = "Your verification code is: 123456";
    match otp_client.extract_otp_from_email(test_body) {
        Some(otp) => eprintln!("  ✓ OTP extraction test passed: {}", otp),
        None => eprintln!("  ✗ OTP extraction test failed"),
    }

    Ok(())
}

fn read_tokens(
    access_token: &Option<String>,
    refresh_token: &Option<String>,
    id_token: &Option<String>,
    from_stdin: bool,
) -> anyhow::Result<TokenData> {
    if from_stdin {
        let stdin = io::stdin();
        let mut lines = stdin.lock().lines();
        let line = lines.next().ok_or_else(|| anyhow::anyhow!("Expected JSON on stdin"))??;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        let access = json["access_token"].as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing access_token in JSON"))?
            .to_string();
        let refresh = json["refresh_token"].as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing refresh_token in JSON"))?
            .to_string();
        let id = json["id_token"].as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing id_token in JSON"))?
            .to_string();

        return build_token_data(access, refresh, id);
    }

    let access = access_token.clone()
        .ok_or_else(|| anyhow::anyhow!("--access-token is required (or use --stdin)"))?;
    let refresh = refresh_token.clone()
        .ok_or_else(|| anyhow::anyhow!("--refresh-token is required (or use --stdin)"))?;
    let id = id_token.clone()
        .ok_or_else(|| anyhow::anyhow!("--id-token is required (or use --stdin)"))?;

    build_token_data(access, refresh, id)
}

fn build_token_data(access: String, refresh: String, id: String) -> anyhow::Result<TokenData> {
    let id_token_info = parse_chatgpt_jwt_claims(&id)?;
    Ok(TokenData {
        id_token: id_token_info,
        access_token: access,
        refresh_token: refresh,
        account_id: None,
    })
}
