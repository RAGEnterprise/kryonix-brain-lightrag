/// scanner.rs — File scanning and content collection for CAG packs
use crate::manifest::Profile;
use crate::security::filter_content;
use anyhow::{Context, Result};
use blake3::Hasher;
use ignore::WalkBuilder;
use std::path::Path;
use tracing::{debug, warn};

/// A file discovered during scan, ready for inclusion in a CAG manifest
#[derive(Debug, Clone)]
pub struct ScannedFile {
    pub path: String, // relative to repo root
    pub size_bytes: u64,
    pub blake3: String,
    pub content: String,   // filtered, safe
    pub tags: Vec<String>, // semantic tags for routing
}

/// Tag a file path with semantic labels for routing
fn derive_tags(rel_path: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let p = rel_path.to_lowercase();

    // Host-based tags
    if p.contains("hosts/glacier") {
        tags.push("glacier".into());
    }
    if p.contains("hosts/inspiron") {
        tags.push("inspiron".into());
    }
    if p.contains("hosts/") {
        tags.push("host-config".into());
    }

    // Module-based tags
    if p.contains("modules/") {
        tags.push("nixos-module".into());
    }
    if p.contains("profiles/") {
        tags.push("profile".into());
    }
    if p.contains("features/") {
        tags.push("feature".into());
    }
    if p.contains("home/") {
        tags.push("home-manager".into());
    }
    if p.contains("desktop/") || p.contains("hyprland") {
        tags.push("desktop".into());
    }
    if p.contains("packages/") {
        tags.push("package".into());
    }

    // Service/topic tags
    if p.contains("nvidia") || p.contains("gpu") {
        tags.push("gpu".into());
    }
    if p.contains("ollama") {
        tags.push("ollama".into());
    }
    if p.contains("brain") {
        tags.push("brain".into());
    }
    if p.contains("lightrag") {
        tags.push("lightrag".into());
    }
    if p.contains("mcp") {
        tags.push("mcp".into());
    }
    if p.contains("tailscale") {
        tags.push("tailscale".into());
    }
    if p.contains("audio") || p.contains("pipewire") {
        tags.push("audio".into());
    }
    if p.contains("bluetooth") {
        tags.push("bluetooth".into());
    }
    if p.contains("nixos-local") || p.contains("local-sources") || p.contains("knowledge-sources") {
        tags.push("local-sources".into());
        tags.push("nixos-sources".into());
    }
    if p.contains("cli") || p.contains("command") || p.contains("commands") || p.contains("usage") {
        tags.push("cli".into());
    }
    if p.contains("operations")
        || p.contains("rebuild")
        || p.contains("switch")
        || p.contains("doctor")
        || p.contains("contract")
        || p.contains("contrato")
    {
        tags.push("operations".into());
    }
    if p.contains("storage") || p.contains("disk") || p.contains("btrfs") {
        tags.push("storage".into());
    }
    if p.contains("networking") || p.contains("firewall") {
        tags.push("networking".into());
    }
    if p.contains("gaming") || p.contains("steam") {
        tags.push("gaming".into());
    }
    if p.contains("virtuali") || p.contains("libvirt") {
        tags.push("virtualization".into());
    }
    if p.contains("ssh") {
        tags.push("ssh".into());
    }

    // File type tags
    if p.ends_with(".nix") {
        tags.push("nix".into());
    }
    if p.ends_with(".md") {
        tags.push("docs".into());
    }
    if p.ends_with(".toml") {
        tags.push("toml".into());
    }
    if p.ends_with(".json") {
        tags.push("json".into());
    }

    // Special files
    if p == "flake.nix" || p.ends_with("/flake.nix") {
        tags.push("flake".into());
    }
    if p.contains("agents.md") || p.contains("agents/") {
        tags.push("agent".into());
    }
    if p.contains("docs/cli/") || p.contains("docs/cli.md") || p.ends_with("cli.md") || p.contains("kryonix-cli") {
        tags.push("cli".into());
        tags.push("operations".into());
    }
    if p.contains("docs/operations.md") || p.ends_with("operations.md") {
        tags.push("operations".into());
    }

    tags.sort();
    tags.dedup();
    tags
}

/// Compute blake3 hash of raw bytes
fn hash_bytes(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

/// Scan the repo for files matching the profile, returning filtered ScannedFile entries
pub fn scan_repo(repo_root: &Path, profile: &Profile) -> Result<Vec<ScannedFile>> {
    let mut results: Vec<ScannedFile> = Vec::new();

    let walker = WalkBuilder::new(repo_root)
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .follow_links(false)
        .build();

    for entry in walker {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!("Walk error: {}", e);
                continue;
            }
        };

        if !entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
            continue;
        }

        let abs_path = entry.path();
        let rel_path = match abs_path.strip_prefix(repo_root) {
            Ok(p) => p.to_string_lossy().to_string(),
            Err(_) => continue,
        };

        // Check exclude patterns first
        let excluded = profile
            .exclude_patterns
            .iter()
            .any(|pat| glob_match(pat, &rel_path));
        if excluded {
            debug!("Excluding: {}", rel_path);
            continue;
        }

        // Check include patterns
        let included = profile
            .include_patterns
            .iter()
            .any(|pat| glob_match(pat, &rel_path));
        if !included {
            continue;
        }

        // Read and check size
        let metadata = match std::fs::metadata(abs_path) {
            Ok(m) => m,
            Err(e) => {
                warn!("Metadata error {}: {}", rel_path, e);
                continue;
            }
        };
        let size_bytes = metadata.len();
        if size_bytes > profile.max_file_bytes as u64 {
            debug!("Skipping large file ({} bytes): {}", size_bytes, rel_path);
            continue;
        }

        // Read content
        let raw = match std::fs::read(abs_path) {
            Ok(b) => b,
            Err(e) => {
                warn!("Read error {}: {}", rel_path, e);
                continue;
            }
        };

        // Skip binary files (null bytes)
        if raw.contains(&0u8) {
            debug!("Skipping binary: {}", rel_path);
            continue;
        }

        let raw_str = match String::from_utf8(raw.clone()) {
            Ok(s) => s,
            Err(_) => {
                debug!("Skipping non-UTF8: {}", rel_path);
                continue;
            }
        };

        let blake3 = hash_bytes(&raw);
        let (content, redacted_lines) = filter_content(&raw_str);
        if !redacted_lines.is_empty() {
            warn!(
                "Redacted {} secret line(s) in {}",
                redacted_lines.len(),
                rel_path
            );
        }

        let tags = derive_tags(&rel_path);

        results.push(ScannedFile {
            path: rel_path,
            size_bytes,
            blake3,
            content,
            tags,
        });

        if results.len() >= profile.max_files {
            warn!(
                "Reached max_files limit ({}) — stopping scan",
                profile.max_files
            );
            break;
        }
    }

    // Sort by path for deterministic output
    results.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(results)
}

/// Minimal glob matching: supports `**`, `*`, and literal segments
/// Only intended for simple path-prefix patterns used by CAG profiles.
fn glob_match(pattern: &str, path: &str) -> bool {
    glob_match_impl(pattern, path)
}

fn glob_match_impl(pat: &str, path: &str) -> bool {
    // Normalize separators
    let pat = pat.replace('\\', "/");
    let path = path.replace('\\', "/");

    // Fast cases
    if pat == "**" || pat == "**/*" {
        return true;
    }

    let pat_parts: Vec<&str> = pat.split('/').collect();
    let path_parts: Vec<&str> = path.split('/').collect();

    glob_match_parts(&pat_parts, &path_parts)
}

fn glob_match_parts(pat: &[&str], path: &[&str]) -> bool {
    match (pat.first(), path.first()) {
        (None, None) => true,
        (None, _) => false,
        (Some(&"**"), rest_pat) => {
            // ** matches zero or more path segments
            for i in 0..=path.len() {
                if glob_match_parts(&pat[1..], &path[i..]) {
                    return true;
                }
            }
            false
        }
        (_, None) => false,
        (Some(p), Some(s)) => {
            if segment_matches(p, s) {
                glob_match_parts(&pat[1..], &path[1..])
            } else {
                false
            }
        }
    }
}

fn segment_matches(pat: &str, seg: &str) -> bool {
    if pat == "*" {
        return true;
    }
    if !pat.contains('*') {
        return pat == seg;
    }

    // Simple wildcard: split on * and check prefix/suffix
    let parts: Vec<&str> = pat.split('*').collect();
    let mut pos = 0usize;
    let seg_bytes = seg.as_bytes();
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        let part_bytes = part.as_bytes();
        if i == 0 {
            // Must be prefix
            if !seg.starts_with(part) {
                return false;
            }
            pos = part.len();
        } else {
            // Find part after pos
            let remaining = &seg_bytes[pos..];
            let found = remaining
                .windows(part_bytes.len())
                .position(|w| w == part_bytes);
            match found {
                Some(idx) => pos += idx + part.len(),
                None => return false,
            }
        }
    }
    // Last part must be suffix if pattern doesn't end with *
    if !pat.ends_with('*') {
        let last = parts.last().unwrap();
        if !last.is_empty() && !seg.ends_with(last) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_nix_files() {
        assert!(glob_match("**/*.nix", "hosts/glacier/default.nix"));
        assert!(glob_match("**/*.nix", "flake.nix"));
        assert!(!glob_match("**/*.nix", "docs/README.md"));
    }

    #[test]
    fn test_glob_docs() {
        assert!(glob_match("docs/**", "docs/README.md"));
        assert!(glob_match("docs/**", "docs/ai/PROJECT_CONTEXT.md"));
        assert!(!glob_match("docs/**", "hosts/glacier/default.nix"));
    }

    #[test]
    fn test_glob_exclude_git() {
        assert!(glob_match("**/.git/**", ".git/HEAD"));
        assert!(glob_match("**/.git/**", ".git/refs/heads/main"));
    }

    #[test]
    fn test_glob_flake() {
        assert!(glob_match("flake.nix", "flake.nix"));
        assert!(!glob_match("flake.nix", "hosts/flake.nix"));
    }

    #[test]
    fn test_tags_glacier() {
        let tags = derive_tags("hosts/glacier/default.nix");
        assert!(tags.contains(&"glacier".to_string()));
        assert!(tags.contains(&"host-config".to_string()));
        assert!(tags.contains(&"nix".to_string()));
    }

    #[test]
    fn test_tags_brain() {
        let tags = derive_tags("packages/kryonix-brain-lightrag/kryonix_brain_lightrag/rag.py");
        assert!(tags.contains(&"brain".to_string()));
        assert!(tags.contains(&"package".to_string()));
    }
}
