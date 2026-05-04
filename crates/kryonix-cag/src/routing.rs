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
    pub suggested_strategy: StrategySuggestion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySuggestion {
    pub strategy: String,
    pub reason: String,
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
    m.insert("docs",       vec![("docs", 1.5)]);
    m.insert("documento",  vec![("docs", 1.5)]);
    m.insert("roadmap",    vec![("docs", 1.5)]);
    m.insert("como",       vec![]); // Prevent noise
    m.insert("no",         vec![]); // Prevent noise
    m
}

/// Determine path-based multiplier for scoring (Tiers)
fn get_path_multiplier(path: &str, query_lower: &str) -> f64 {
    let path_lower = path.to_lowercase();
    
    // Check if query is about disks
    let disk_keywords = ["disco", "disk", "partição", "particao", "partition", "instalação", "instalacao", "install", "formatação", "formatacao", "format", "mount", "filesystem", "fs", "nvme", "sda", "vda", "disko", "zfs", "btrfs", "ext4", "storage"];
    let is_disk_query = disk_keywords.iter().any(|k| query_lower.contains(k));

    // Check if query is about archive
    let archive_keywords = ["antigo", "legacy", "archive", "histórico", "historico", "history", "vault"];
    let is_archive_query = archive_keywords.iter().any(|k| query_lower.contains(k));

    // Tier 4: Disk/Install Penalty (0.05x)
    // Aggressive penalty for infrastructure files when not explicitly asked
    if (path_lower.contains("disks.nix") || 
        path_lower.contains("disko") || 
        path_lower.contains("partition") || 
        path_lower.contains("install") ||
        path_lower.contains("hardware-configuration")) && !is_disk_query {
        return 0.05;
    }
    
    // Tier 4: Archive/Legacy Penalty (0.1x)
    if (path_lower.contains("archive/") || 
        path_lower.contains("legacy/") || 
        path_lower.contains("antigo/")) && !is_archive_query {
        return 0.1;
    }

    // Tier 1: Canonical Docs (3.0x)
    // Highest priority for documented "Source of Truth"
    if path_lower.contains("docs/hosts/") || 
       path_lower.contains("docs/ai/") || 
       path_lower.contains(".ai/skills/") || 
       path_lower.ends_with("readme.md") || 
       path_lower.ends_with("agents.md") {
        return 3.0;
    }

    // Tier 2: Key Configs (1.5x)
    if path_lower.contains("hosts/glacier/default.nix") || 
       path_lower.contains("profiles/glacier-ai.nix") || 
       path_lower.contains("modules/nixos/services/brain.nix") ||
       path_lower.contains("modules/nixos/ai/") ||
       path_lower.contains("flake.nix") {
        return 1.5;
    }

    1.0
}

/// Suggest the best search strategy (cag, rag, hybrid) for a query
pub fn suggest_strategy(query: &str) -> StrategySuggestion {
    let query_lower = query.to_lowercase();
    
    // RAG/Hybrid triggers: vault, deep knowledge, history, conversations
    let rag_triggers = [
        "vault", "histórico", "historico", "nota antiga", "conversa anterior", 
        "brain", "lightrag", "pensamento", "log", "incidente", "decisão", 
        "grounding", "conhecimento"
    ];
    if rag_triggers.iter().any(|t| query_lower.contains(t)) {
        return StrategySuggestion {
            strategy: "hybrid".to_string(),
            reason: "Query indicates a search for deep knowledge, history, or incident logs found in the vault/RAG.".to_string(),
        };
    }
    
    // CAG triggers: repo structure, specific configs, nix files, current implementation
    let cag_triggers = [
        "como funciona", "onde fica", "configuração", "configuracao", "nix", 
        "flake", "host", "glacier", "inspiron", "código", "codigo", "implementação",
        "implementacao", "módulo", "modulo", "package", "pacote"
    ];
    if cag_triggers.iter().any(|t| query_lower.contains(t)) {
        return StrategySuggestion {
            strategy: "cag".to_string(),
            reason: "Query is about repository structure, configuration, or active code implementation.".to_string(),
        };
    }
    
    // Default to hybrid for safety
    StrategySuggestion {
        strategy: "hybrid".to_string(),
        reason: "General query, hybrid provides best coverage by combining repo code and vault knowledge.".to_string(),
    }
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
        // Substring match (strict: length > 3 for both to avoid noise)
        for (kw, pairs) in &kw_weights {
            let kw_s: &str = kw;
            let word_s: &str = word;
            if word_s.len() > 3 && kw_s.len() > 3 && (word_s.contains(kw_s) || kw_s.contains(word_s)) {
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

            // 1. Base Score from matched tags
            let base_score: f64 = matched_tags.iter()
                .map(|t| {
                    if file_tags.contains(t.as_str()) {
                        tag_scores.get(t).copied().unwrap_or(0.0)
                    } else {
                        0.0
                    }
                })
                .sum();

            // 2. Direct Keyword Match in Path (Path Bonus)
            let path_lower = f.path.to_lowercase();
            let path_bonus: f64 = words.iter()
                .filter(|w| w.len() > 3 && { let s: &str = w; path_lower.contains(s) })
                .map(|_| 0.5)
                .sum();

            let total_pre_multiplier = base_score + path_bonus;

            // 3. Tiered Priority and Penalties
            let multiplier = get_path_multiplier(&f.path, &query_lower);
            
            (i, total_pre_multiplier * multiplier)
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
        suggested_strategy: suggest_strategy(query),
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

    #[test]
    fn test_suggest_strategy() {
        // RAG trigger
        let res = suggest_strategy("Procure no vault uma nota sobre o cérebro");
        assert_eq!(res.strategy, "hybrid");
        assert!(res.reason.to_lowercase().contains("vault") || res.reason.to_lowercase().contains("knowledge"));

        // CAG trigger
        let res = suggest_strategy("Como funciona o host glacier?");
        assert_eq!(res.strategy, "cag");
        assert!(res.reason.to_lowercase().contains("repository") || res.reason.to_lowercase().contains("configuration"));

        // Default
        let res = suggest_strategy("Qualquer coisa genérica");
        assert_eq!(res.strategy, "hybrid");
    }

    #[test]
    fn test_path_multiplier_canonical_boost() {
        // Tier 1
        assert_eq!(get_path_multiplier("AGENTS.md", "general query"), 3.0);
        assert_eq!(get_path_multiplier("docs/hosts/glacier.md", "glacier"), 3.0);

        // Tier 2
        assert_eq!(get_path_multiplier("hosts/glacier/default.nix", "glacier"), 1.5);

        // Tier 3 (Default)
        assert_eq!(get_path_multiplier("modules/nixos/audio/default.nix", "audio"), 1.0);
    }

    #[test]
    fn test_path_multiplier_disk_penalty() {
        // Penalized: query is NOT about disks
        assert_eq!(get_path_multiplier("hosts/glacier/disks.nix", "Como funciona o glacier?"), 0.05);
        assert_eq!(get_path_multiplier("hosts/glacier/disko.nix", "glacier config"), 0.05);

        // NOT Penalized: query IS about disks
        assert_eq!(get_path_multiplier("hosts/glacier/disks.nix", "Como particionar o disco?"), 1.0);
        assert_eq!(get_path_multiplier("hosts/glacier/disko.nix", "disko configuration"), 1.0);
    }

    #[test]
    fn test_path_multiplier_archive_penalty() {
        // Penalized: query is NOT about history
        assert_eq!(get_path_multiplier("archive/old_config.nix", "general query"), 0.1);

        // NOT Penalized: query IS about history
        assert_eq!(get_path_multiplier("archive/old_config.nix", "nota antiga no vault"), 1.0);
        assert_eq!(get_path_multiplier("legacy/old.md", "histórico do projeto"), 1.0);
    }

    #[test]
    fn test_route_with_penalties() {
        let mut tags: HashMap<String, Vec<String>> = HashMap::new();
        tags.insert("glacier".into(), vec!["hosts/glacier/default.nix".into(), "hosts/glacier/disks.nix".into()]);
        tags.insert("docs".into(), vec!["AGENTS.md".into()]);

        let manifest = CagManifest {
            version: 1,
            profile: "test".into(),
            repo_root: "/etc/kryonix".into(),
            built_at: Utc::now(),
            total_files: 3,
            total_bytes: 1000,
            content_hash: "abc123".into(),
            files: vec![
                FileEntry { path: "hosts/glacier/default.nix".into(), size_bytes: 400, blake3: "h1".into(), content: "# config".into() },
                FileEntry { path: "hosts/glacier/disks.nix".into(), size_bytes: 300, blake3: "h2".into(), content: "# disks".into() },
                FileEntry { path: "AGENTS.md".into(), size_bytes: 300, blake3: "h3".into(), content: "# agents".into() },
            ],
            tags,
        };

        // General query: disks.nix should be last even if it has same tags
        let result = route_query(&manifest, "Como funciona o glacier?", 10);
        let paths: Vec<&str> = result.matched_files.iter().map(|f| f.path.as_str()).collect();

        // glacier/default.nix should be before glacier/disks.nix
        let idx_default = paths.iter().position(|&p| p == "hosts/glacier/default.nix").unwrap();
        let idx_disks = paths.iter().position(|&p| p == "hosts/glacier/disks.nix").unwrap();
        assert!(idx_default < idx_disks);

        // Let's try query "glacier docs"
        let result = route_query(&manifest, "glacier docs", 10);
        let paths: Vec<&str> = result.matched_files.iter().map(|f| f.path.as_str()).collect();
        // AGENTS.md should be before disks.nix
        let idx_agents = paths.iter().position(|&p| p == "AGENTS.md").unwrap();
        let idx_disks_2 = paths.iter().position(|&p| p == "hosts/glacier/disks.nix").unwrap();
        assert!(idx_agents < idx_disks_2);
    }
}
