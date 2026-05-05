/// manifest.rs — CAG pack manifest structures and serialization
use crate::scanner::ScannedFile;
use anyhow::Result;
use blake3::Hasher;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Profiles define which file patterns to include in the CAG pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub name: String,
    pub description: String,
    pub include_patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
    pub max_files: usize,
    pub max_file_bytes: usize,
}

impl Profile {
    /// The canonical kryonix-core profile — repo structure, NixOS modules, docs
    pub fn kryonix_core() -> Self {
        Self {
            name: "kryonix-core".into(),
            description: "Core Kryonix repo context: NixOS config, docs, modules, hosts".into(),
            include_patterns: vec![
                "**/*.nix".into(),
                "**/*.md".into(),
                "**/*.toml".into(),
                "**/*.json".into(),
                "**/AGENTS.md".into(),
                "**/README*".into(),
                "docs/**".into(),
                "docs/ai/**".into(),
                "hosts/**".into(),
                "modules/**".into(),
                "profiles/**".into(),
                "features/**".into(),
                "packages/**/*.nix".into(),
                "packages/**/*.md".into(),
                ".ai/skills/brain/**".into(),
                "flake.nix".into(),
                "flake.lock".into(),
            ],
            exclude_patterns: vec![
                "**/.git/**".into(),
                "**/result/**".into(),
                "**/node_modules/**".into(),
                "**/.direnv/**".into(),
                "**/.venv/**".into(),
                "**/target/**".into(),
                "**/__pycache__/**".into(),
                "**/*.pyc".into(),
                "**/*.lock".into(),
                "**/Cargo.lock".into(),
                "**/.obsidian/**".into(),
                "**/themes/**".into(),
            ],
            max_files: 2000,
            max_file_bytes: 256 * 1024, // 256 KB per file
        }
    }

    /// The vault profile — Obsidian notes only
    pub fn kryonix_vault() -> Self {
        Self {
            name: "kryonix-vault".into(),
            description: "Kryonix Obsidian vault: notes, skills, playbooks".into(),
            include_patterns: vec!["**/*.md".into()],
            exclude_patterns: vec!["**/.obsidian/**".into(), "**/themes/**".into()],
            max_files: 5000,
            max_file_bytes: 128 * 1024,
        }
    }

    pub fn by_name(name: &str) -> Option<Self> {
        match name {
            "kryonix-core" => Some(Self::kryonix_core()),
            "kryonix-vault" => Some(Self::kryonix_vault()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Profile;

    #[test]
    fn test_kryonix_core_includes_canonical_ai_docs() {
        let profile = Profile::kryonix_core();
        assert!(profile.include_patterns.iter().any(|p| p == "docs/ai/**"));
        assert!(profile
            .include_patterns
            .iter()
            .any(|p| p == ".ai/skills/brain/**"));
    }
}

/// Metadata for a single file in the CAG pack
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub size_bytes: u64,
    pub blake3: String,
    pub content: String, // filtered, safe content
}

/// The full CAG manifest — produced by `build`, read by `route`/`ask`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CagManifest {
    pub version: u32,
    pub profile: String,
    pub repo_root: String,
    pub built_at: DateTime<Utc>,
    pub total_files: usize,
    pub total_bytes: u64,
    /// Content fingerprint (blake3 of all file hashes)
    pub content_hash: String,
    pub files: Vec<FileEntry>,
    /// Tag → list of file paths (used by router)
    pub tags: HashMap<String, Vec<String>>,
}

impl CagManifest {
    pub fn from_scanned(profile: &Profile, repo_root: &Path, files: Vec<ScannedFile>) -> Self {
        let total_bytes: u64 = files.iter().map(|f| f.size_bytes).sum();
        let total_files = files.len();

        // Compute content_hash from all individual hashes
        let mut hasher = Hasher::new();
        for f in &files {
            hasher.update(f.blake3.as_bytes());
        }
        let content_hash = hasher.finalize().to_hex().to_string();

        // Build tag index
        let mut tags: HashMap<String, Vec<String>> = HashMap::new();
        for f in &files {
            for tag in &f.tags {
                tags.entry(tag.clone()).or_default().push(f.path.clone());
            }
        }

        let entries = files
            .into_iter()
            .map(|f| FileEntry {
                path: f.path,
                size_bytes: f.size_bytes,
                blake3: f.blake3,
                content: f.content,
            })
            .collect();

        Self {
            version: 1,
            profile: profile.name.clone(),
            repo_root: repo_root.display().to_string(),
            built_at: Utc::now(),
            total_files,
            total_bytes,
            content_hash,
            files: entries,
            tags,
        }
    }

    /// Save manifest to `{dir}/manifest.json`
    pub fn save(&self, dir: &Path) -> Result<()> {
        fs::create_dir_all(dir)?;
        let path = dir.join("manifest.json");
        let json = serde_json::to_string_pretty(self)?;
        fs::write(&path, json)?;
        Ok(())
    }

    /// Load manifest from `{dir}/manifest.json`
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("manifest.json");
        let data = fs::read_to_string(&path)?;
        let manifest: Self = serde_json::from_str(&data)?;
        Ok(manifest)
    }

    /// Return a public summary view — no file content
    pub fn summary(&self) -> serde_json::Value {
        serde_json::json!({
            "version": self.version,
            "profile": self.profile,
            "repo_root": self.repo_root,
            "built_at": self.built_at,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "content_hash": &self.content_hash[..16],
            "tag_count": self.tags.len(),
        })
    }
}
