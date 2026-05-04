/// routing.rs — Query routing logic for CAG packs
use crate::manifest::CagManifest;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A routing decision with matched files ranked by relevance
#[derive(Debug, Serialize, Deserialize)]
pub struct RoutingResult {
    pub query: String,
    pub matched_tags: Vec<String>,
    pub matched_files: Vec<MatchedFile>,
    pub total_tokens_est: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedFile {
    pub path: String,
    pub score: f64,
    pub tags: Vec<String>,
    pub snippet: String, // first 300 chars of content
}

/// Keyword → tag weight mappings for semantic routing
fn keyword_tag_weights() -> HashMap<&'static str, Vec<(&'static str, f64)>> {
    let mut m = HashMap::new();
    // glacier/server
    m.insert("glacier",    vec![("glacier", 2.0), ("host-config", 1.0), ("brain", 0.5)]);
    m.insert("server",     vec![("glacier", 1.5), ("host-config", 1.0)]);
    m.insert("servidor",   vec![("glacier", 1.5), ("host-config", 1.0)]);
    // inspiron/workstation
    m.insert("inspiron",   vec![("inspiron", 2.0), ("host-config", 1.0)]);
    m.insert("workstation",vec![("inspiron", 1.5), ("host-config", 1.0)]);
    // brain/rag
    m.insert("brain",      vec![("brain", 2.0), ("lightrag", 1.5), ("mcp", 0.5)]);
    m.insert("rag",        vec![("lightrag", 2.0), ("brain", 1.5)]);
    m.insert("lightrag",   vec![("lightrag", 2.0), ("brain", 1.0)]);
    m.insert("índice",     vec![("lightrag", 1.5), ("brain", 1.0)]);
    m.insert("indice",     vec![("lightrag", 1.5), ("brain", 1.0)]);
    // vault
    m.insert("vault",      vec![("brain", 1.0), ("docs", 0.5)]);
    m.insert("obsidian",   vec![("brain", 1.0), ("docs", 0.5)]);
    m.insert("nota",       vec![("docs", 1.5), ("brain", 0.5)]);
    m.insert("note",       vec![("docs", 1.5), ("brain", 0.5)]);
    // gpu/nvidia/ollama
    m.insert("nvidia",     vec![("gpu", 2.0), ("glacier", 1.0)]);
    m.insert("gpu",        vec![("gpu", 2.0), ("glacier", 1.0)]);
    m.insert("cuda",       vec![("gpu", 2.0), ("glacier", 1.0)]);
    m.insert("ollama",     vec![("ollama", 2.0), ("glacier", 1.0), ("brain", 0.5)]);
    // mcp
    m.insert("mcp",        vec![("mcp", 2.0), ("brain", 0.5)]);
    // tailscale/rede
    m.insert("tailscale",  vec![("tailscale", 2.0), ("networking", 1.0)]);
    m.insert("rede",       vec![("networking", 2.0), ("tailscale", 0.5)]);
    m.insert("network",    vec![("networking", 2.0), ("tailscale", 0.5)]);
    m.insert("firewall",   vec![("networking", 2.0)]);
    m.insert("ssh",        vec![("ssh", 2.0), ("networking", 0.5)]);
    // nixos/flake
    m.insert("nix",        vec![("nix", 2.0), ("flake", 0.5)]);
    m.insert("nixos",      vec![("nix", 2.0), ("host-config", 1.0)]);
    m.insert("flake",      vec![("flake", 2.0), ("nix", 1.0)]);
    m.insert("módulo",     vec![("nixos-module", 2.0), ("nix", 1.0)]);
    m.insert("modulo",     vec![("nixos-module", 2.0), ("nix", 1.0)]);
    m.insert("module",     vec![("nixos-module", 2.0), ("nix", 1.0)]);
    m.insert("rebuild",    vec![("nix", 1.5), ("host-config", 1.0)]);
    m.insert("switch",     vec![("nix", 1.5), ("host-config", 1.0)]);
    // audio
    m.insert("audio",      vec![("audio", 2.0)]);
    m.insert("pipewire",   vec![("audio", 2.0)]);
    m.insert("bluetooth",  vec![("bluetooth", 2.0), ("audio", 0.5)]);
    m.insert("som",        vec![("audio", 2.0)]);
    // gaming
    m.insert("gaming",     vec![("gaming", 2.0)]);
    m.insert("steam",      vec![("gaming", 2.0)]);
    m.insert("gamemode",   vec![("gaming", 2.0)]);
    // desktop/hyprland
    m.insert("desktop",    vec![("desktop", 2.0)]);
    m.insert("hyprland",   vec![("desktop", 2.0)]);
    m.insert("caelestia",  vec![("desktop", 2.0)]);
    // storage
    m.insert("storage",    vec![("storage", 2.0)]);
    m.insert("btrfs",      vec![("storage", 2.0)]);
    m.insert("disco",      vec![("storage", 2.0)]);
    m.insert("disk",       vec![("storage", 2.0)]);
    // docs/agents
    m.insert("agents",     vec![("agent", 2.0), ("docs", 1.0)]);
    m.insert("doc",        vec![("docs", 1.5)]);
    m.insert("roadmap",    vec![("docs", 1.5)]);
    m
}

/// Route a natural-language query to the most relevant files in the manifest
pub fn route_query(manifest: &CagManifest, query: &str, top_k: usize) -> RoutingResult {
    let kw_weights = keyword_tag_weights();
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower.split_whitespace().collect();

    // Accumulate tag scores from keywords
    let mut tag_scores: HashMap<String, f64> = HashMap::new();
    for word in &words {
        // Exact match
        if let Some(pairs) = kw_weights.get(word) {
            for (tag, weight) in pairs {
                *tag_scores.entry(tag.to_string()).or_default() += weight;
            }
        }
        // Substring match (e.g., "bluetooth" matches "bluetoothctl")
        for (kw, pairs) in &kw_weights {
            let kw_s: &str = kw;
            let word_s: &str = word;
            if word_s.contains(kw_s) || kw_s.contains(word_s) {
                for (tag, weight) in pairs {
                    *tag_scores.entry(tag.to_string()).or_default() += weight * 0.5;
                }
            }
        }
    }

    let matched_tags: Vec<String> = {
        let mut scored: Vec<(String, f64)> = tag_scores.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored.into_iter().take(8).map(|(t, _)| t).collect()
    };

    // Score each file in the manifest
    let mut file_scores: Vec<(usize, f64)> = manifest.files.iter().enumerate()
        .map(|(i, f)| {
            let file_tags: std::collections::HashSet<&str> =
                manifest.tags.iter()
                    .filter(|(_, paths)| paths.contains(&f.path))
                    .map(|(tag, _)| tag.as_str())
                    .collect();

            let score: f64 = matched_tags.iter()
                .map(|t| {
                    if file_tags.contains(t.as_str()) {
                        tag_scores.get(t).copied().unwrap_or(0.0)
                    } else {
                        0.0
                    }
                })
                .sum();

            // Bonus: direct keyword match in path
            let path_lower = f.path.to_lowercase();
            let path_bonus: f64 = words.iter()
                .filter(|w| w.len() > 3 && { let s: &str = w; path_lower.contains(s) })
                .map(|_| 0.5)
                .sum();

            (i, score + path_bonus)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    file_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    file_scores.truncate(top_k);

    let mut total_tokens_est = 0usize;
    let matched_files: Vec<MatchedFile> = file_scores.iter()
        .map(|(i, score)| {
            let f = &manifest.files[*i];
            let snippet: String = f.content.chars().take(300).collect();
            let token_est = f.content.len() / 4; // rough: 1 token ≈ 4 chars
            total_tokens_est += token_est;

            // Collect this file's tags from manifest.tags
            let file_tags: Vec<String> = manifest.tags.iter()
                .filter(|(_, paths)| paths.contains(&f.path))
                .map(|(tag, _)| tag.clone())
                .collect();

            MatchedFile {
                path: f.path.clone(),
                score: (*score * 100.0).round() / 100.0,
                tags: file_tags,
                snippet,
            }
        })
        .collect();

    RoutingResult {
        query: query.to_string(),
        matched_tags,
        matched_files,
        total_tokens_est,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{CagManifest, FileEntry};
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_test_manifest() -> CagManifest {
        let mut tags: HashMap<String, Vec<String>> = HashMap::new();
        tags.insert("glacier".into(), vec!["hosts/glacier/default.nix".into()]);
        tags.insert("host-config".into(), vec!["hosts/glacier/default.nix".into(), "hosts/inspiron/default.nix".into()]);
        tags.insert("brain".into(), vec!["packages/kryonix-brain-lightrag/README.md".into()]);
        tags.insert("nix".into(), vec!["hosts/glacier/default.nix".into(), "hosts/inspiron/default.nix".into(), "flake.nix".into()]);

        CagManifest {
            version: 1,
            profile: "test".into(),
            repo_root: "/etc/kryonix".into(),
            built_at: Utc::now(),
            total_files: 3,
            total_bytes: 1000,
            content_hash: "abc123".into(),
            files: vec![
                FileEntry { path: "hosts/glacier/default.nix".into(), size_bytes: 400, blake3: "h1".into(), content: "# Glacier host config\nservices.ollama.enable = true;".into() },
                FileEntry { path: "hosts/inspiron/default.nix".into(), size_bytes: 300, blake3: "h2".into(), content: "# Inspiron workstation config".into() },
                FileEntry { path: "flake.nix".into(), size_bytes: 300, blake3: "h3".into(), content: "# flake outputs".into() },
            ],
            tags,
        }
    }

    #[test]
    fn test_route_glacier_query() {
        let manifest = make_test_manifest();
        let result = route_query(&manifest, "Como funciona o Glacier no Kryonix?", 10);
        assert!(!result.matched_files.is_empty());
        assert_eq!(result.matched_files[0].path, "hosts/glacier/default.nix");
    }

    #[test]
    fn test_route_nix_query() {
        let manifest = make_test_manifest();
        let result = route_query(&manifest, "Como funciona o flake do NixOS?", 10);
        assert!(!result.matched_files.is_empty());
        // flake.nix or glacier should match
        let paths: Vec<&str> = result.matched_files.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.iter().any(|p| p.contains("flake") || p.contains("glacier")));
    }

    #[test]
    fn test_matched_tags_not_empty_for_known_query() {
        let manifest = make_test_manifest();
        let result = route_query(&manifest, "Como funciona o Glacier?", 10);
        assert!(!result.matched_tags.is_empty());
    }
}
