//! Team account storage structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::fs;
use std::io;

use crate::token_data::TokenData;

/// Team account with associated member OAuth tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamAccount {
    /// Unique team identifier
    pub id: String,
    /// Team name/label
    pub name: String,
    /// Team owner's OAuth tokens
    pub owner_tokens: TokenData,
    /// Team members with their OAuth tokens
    pub members: Vec<TeamMember>,
    /// When the team was created
    pub created_at: DateTime<Utc>,
    /// When the team was last updated
    pub updated_at: DateTime<Utc>,
}

/// A team member with their OAuth tokens.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    /// Unique member identifier
    pub id: String,
    /// Member email address
    pub email: String,
    /// Member's OAuth tokens (None if not yet logged in)
    pub tokens: Option<TokenData>,
    /// When the member was invited
    pub invited_at: DateTime<Utc>,
    /// When the member completed OAuth login
    pub logged_in_at: Option<DateTime<Utc>>,
    /// Member's role in the team
    pub role: TeamRole,
}

/// Role of a team member.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TeamRole {
    Admin,
    Member,
}

impl std::fmt::Display for TeamRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeamRole::Admin => write!(f, "Admin"),
            TeamRole::Member => write!(f, "Member"),
        }
    }
}

/// Storage for team accounts.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TeamStore {
    /// Map of team ID to team account
    pub teams: HashMap<String, TeamAccount>,
    /// Path to the storage file
    #[serde(skip)]
    path: Option<PathBuf>,
}

impl TeamStore {
    const FILENAME: &'static str = "teams.json";

    /// Load team store from CODEX_HOME.
    pub fn load(codex_home: &Path) -> io::Result<Self> {
        let path = codex_home.join(Self::FILENAME);
        if !path.exists() {
            return Ok(Self {
                teams: HashMap::new(),
                path: Some(path),
            });
        }

        let contents = fs::read_to_string(&path)?;
        let mut store: TeamStore = serde_json::from_str(&contents)
            .map_err(|e| io::Error::other(format!("failed to parse teams.json: {e}")))?;
        store.path = Some(path);
        Ok(store)
    }

    /// Save team store to disk.
    pub fn save(&self) -> io::Result<()> {
        let path = self.path.as_ref().ok_or_else(|| {
            io::Error::other("team store path not set")
        })?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let contents = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::other(format!("failed to serialize teams: {e}")))?;
        fs::write(path, contents)?;

        Ok(())
    }

    /// Add a new team account.
    pub fn add_team(&mut self, team: TeamAccount) {
        self.teams.insert(team.id.clone(), team);
    }

    /// Get a team by ID.
    pub fn get_team(&self, id: &str) -> Option<&TeamAccount> {
        self.teams.get(id)
    }

    /// Get a mutable team by ID.
    pub fn get_team_mut(&mut self, id: &str) -> Option<&mut TeamAccount> {
        self.teams.get_mut(id)
    }

    /// Remove a team by ID.
    pub fn remove_team(&mut self, id: &str) -> Option<TeamAccount> {
        self.teams.remove(id)
    }

    /// List all teams.
    pub fn list_teams(&self) -> Vec<&TeamAccount> {
        self.teams.values().collect()
    }

    /// Add a member to a team.
    pub fn add_member(&mut self, team_id: &str, member: TeamMember) -> io::Result<()> {
        let team = self.teams.get_mut(team_id).ok_or_else(|| {
            io::Error::other(format!("team not found: {team_id}"))
        })?;
        team.members.push(member);
        team.updated_at = Utc::now();
        Ok(())
    }

    /// Update a member's OAuth tokens after successful login.
    pub fn set_member_tokens(
        &mut self,
        team_id: &str,
        member_id: &str,
        tokens: TokenData,
    ) -> io::Result<()> {
        let team = self.teams.get_mut(team_id).ok_or_else(|| {
            io::Error::other(format!("team not found: {team_id}"))
        })?;

        let member = team.members.iter_mut().find(|m| m.id == member_id).ok_or_else(|| {
            io::Error::other(format!("member not found: {member_id} in team {team_id}"))
        })?;

        member.tokens = Some(tokens);
        member.logged_in_at = Some(Utc::now());
        team.updated_at = Utc::now();
        Ok(())
    }
}

impl TeamAccount {
    /// Create a new team account.
    pub fn new(name: String, owner_tokens: TokenData) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            owner_tokens,
            members: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a member to the team (invitation).
    pub fn invite_member(&mut self, email: String, role: TeamRole) -> TeamMember {
        let member = TeamMember {
            id: uuid::Uuid::new_v4().to_string(),
            email,
            tokens: None,
            invited_at: Utc::now(),
            logged_in_at: None,
            role,
        };
        self.members.push(member.clone());
        self.updated_at = Utc::now();
        member
    }

    /// Get member by email.
    pub fn get_member_by_email(&self, email: &str) -> Option<&TeamMember> {
        self.members.iter().find(|m| m.email == email)
    }

    /// Get member by ID.
    pub fn get_member(&self, id: &str) -> Option<&TeamMember> {
        self.members.iter().find(|m| m.id == id)
    }

    /// List members without OAuth tokens (pending login).
    pub fn pending_members(&self) -> Vec<&TeamMember> {
        self.members.iter().filter(|m| m.tokens.is_none()).collect()
    }

    /// List members with OAuth tokens (logged in).
    pub fn active_members(&self) -> Vec<&TeamMember> {
        self.members.iter().filter(|m| m.tokens.is_some()).collect()
    }
}
