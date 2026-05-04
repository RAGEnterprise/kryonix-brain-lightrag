use crate::manifest::CagManifest;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedFile {
    pub path: String,
    pub score: f64,
    pub tags: Vec<String>,
    pub snippet: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySuggestion {
    pub strategy: String,
    pub confidence: f64,
    pub confidence_label: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingResult {
    pub query: String,
    pub profile: String,
    pub strategy: String,
    pub confidence: f64,
    pub confidence_label: String,
    pub reason: String,
    pub matched_tags: Vec<String>,
    pub matched_files: Vec<MatchedFile>,
    pub total_tokens_est: usize,
}

fn confidence_label(score: f64) -> String {
    if score >= 0.75 {
        "Alta".to_string()
    } else if score >= 0.45 {
        "Média".to_string()
    } else {
        "Baixa".to_string()
    }
}

fn has_any(query_lower: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| query_lower.contains(needle))
}

fn keyword_tag_weights() -> HashMap<&'static str, Vec<(&'static str, f64)>> {
    HashMap::from([
        ("glacier", vec![("glacier", 2.5), ("host-config", 1.5)]),
        ("inspiron", vec![("inspiron", 2.5), ("host-config", 1.0)]),
        ("brain", vec![("brain", 2.0), ("lightrag", 1.5), ("mcp", 0.5)]),
        ("rag", vec![("lightrag", 2.0), ("brain", 1.0)]),
        ("vault", vec![("brain", 1.0), ("docs", 0.8)]),
        ("bancos", vec![("local-sources", 2.5), ("docs", 1.0)]),
        ("fontes", vec![("local-sources", 2.5), ("docs", 1.0)]),
        ("locais", vec![("local-sources", 2.0), ("docs", 0.8)]),
        ("local", vec![("local-sources", 1.5), ("docs", 0.5)]),
        ("nixpkgs", vec![("local-sources", 1.5), ("nix", 1.0)]),
        ("home-manager", vec![("local-sources", 1.5), ("nix", 1.0)]),
        ("noogle", vec![("local-sources", 1.5), ("nix", 0.5)]),
        ("nvidia", vec![("gpu", 2.0), ("glacier", 1.0)]),
        ("cuda", vec![("gpu", 2.0), ("glacier", 1.0)]),
        ("ollama", vec![("ollama", 2.0), ("glacier", 1.0), ("brain", 0.5)]),
        ("tailscale", vec![("tailscale", 2.0), ("networking", 1.0)]),
        ("ssh", vec![("ssh", 2.0), ("networking", 0.5)]),
        ("nix", vec![("nix", 2.0), ("flake", 0.5)]),
        ("nixos", vec![("nix", 2.0), ("host-config", 1.0)]),
        ("flake", vec![("flake", 2.0), ("nix", 1.0)]),
        ("rebuild", vec![("nix", 1.5), ("host-config", 1.0), ("operations", 1.0)]),
        ("switch", vec![("nix", 1.5), ("host-config", 1.0), ("operations", 1.0)]),
        ("cli", vec![("cli", 2.5), ("docs", 1.0)]),
        ("comandos", vec![("operations", 1.5), ("cli", 1.0)]),
        ("comando", vec![("operations", 1.5), ("cli", 1.0)]),
        ("disko", vec![("storage", 2.0)]),
        ("disco", vec![("storage", 2.0)]),
        ("disk", vec![("storage", 2.0)]),
        ("iso", vec![("storage", 2.0)]),
        ("live", vec![("storage", 1.5)]),
        ("docs", vec![("docs", 1.5)]),
        ("roadmap", vec![("docs", 1.5)]),
    ])
}

fn is_disk_query(query_lower: &str) -> bool {
    has_any(query_lower, &["disco", "disk", "partição", "particao", "particionamento", "disko", "storage", "filesystem", "mount", "install", "instalação", "instalacao"])
}

fn is_iso_query(query_lower: &str) -> bool {
    has_any(query_lower, &["iso", "live", "usb", "bootable", "flash"])
}

fn is_archive_query(query_lower: &str) -> bool {
    has_any(query_lower, &["antigo", "legacy", "archive", "histórico", "historico", "history", "vault"])
}

fn is_rebuild_query(query_lower: &str) -> bool {
    has_any(query_lower, &["rebuild", "switch", "seguro", "glacier"])
}

fn is_local_sources_query(query_lower: &str) -> bool {
    has_any(query_lower, &["bancos", "fontes", "locais", "local", "sources", "nixpkgs", "home-manager", "noogle"])
}

fn is_operations_query(query_lower: &str) -> bool {
    has_any(query_lower, &["cli", "kryonix", "operacional", "comando", "comandos", "boot", "test", "check", "fmt", "doctor", "switch", "rebuild"])
}

fn get_path_multiplier(path: &str, query_lower: &str) -> f64 {
    let path_lower = path.to_lowercase();
    let disk_query = is_disk_query(query_lower);
    let iso_query = is_iso_query(query_lower);
    let archive_query = is_archive_query(query_lower);
    let rebuild_query = is_rebuild_query(query_lower);
    let local_sources_query = is_local_sources_query(query_lower);
    let operations_query = is_operations_query(query_lower);

    if local_sources_query
        && (path_lower.contains("docs/ai/nixos-local-knowledge-sources.md")
            || path_lower.contains(".ai/skills/brain/nixos-local-sources.md"))
    {
        return 8.0;
    }

    if rebuild_query
        && (path_lower.contains("docs/hosts/glacier-rebuild.md")
            || path_lower.contains("docs/hosts/glacier-switch.md")
            || path_lower.contains(".ai/skills/commands/rebuild-nixos.md"))
    {
        return 7.5;
    }

    if (disk_query || iso_query)
        && (path_lower.contains("glacier-live-iso.md") || path_lower.contains("disks.nix") || path_lower.contains("disko") || path_lower.contains("hardware-configuration"))
    {
        return if path_lower.contains("glacier-live-iso.md") { 6.0 } else { 5.0 };
    }

    if operations_query
        && (path_lower.contains("docs/cli.md") || path_lower.contains("docs/operations.md") || path_lower.contains("packages/kryonix-cli.nix"))
    {
        return 6.0;
    }

    if archive_query && path_lower.contains("archive/") {
        return 4.0;
    }

    if path_lower.contains("docs/hosts/") || path_lower.contains("docs/ai/") || path_lower.contains(".ai/skills/") {
        return 3.0;
    }

    if path_lower.contains("hosts/glacier/default.nix")
        || path_lower.contains("profiles/glacier-ai.nix")
        || path_lower.contains("modules/nixos/services/brain.nix")
        || path_lower.contains("packages/kryonix-cli.nix")
    {
        return 1.8;
    }

    1.0
}

pub fn suggest_strategy(query: &str) -> StrategySuggestion {
    let query_lower = query.to_lowercase();
    let rag_score = if has_any(&query_lower, &["vault", "brain", "lightrag", "histórico", "historico", "incidente", "decisão", "grounding"]) { 0.8 } else { 0.0 };
    let cag_score = if has_any(&query_lower, &["nix", "flake", "glacier", "inspiron", "bancos", "fontes", "locais", "cli", "comando", "rebuild", "switch"]) { 0.9 } else { 0.0 };

    if rag_score > 0.0 && cag_score > 0.0 {
        return StrategySuggestion {
            strategy: "hybrid".to_string(),
            confidence: 0.8,
            confidence_label: confidence_label(0.8),
            reason: "Query touches both repository implementation and vault/knowledge concepts.".to_string(),
        };
    }

    if rag_score > 0.0 {
        return StrategySuggestion {
            strategy: "rag".to_string(),
            confidence: 0.8,
            confidence_label: confidence_label(0.8),
            reason: "Query points to deep knowledge, history, or vault content.".to_string(),
        };
    }

    StrategySuggestion {
        strategy: "cag".to_string(),
        confidence: if cag_score > 0.0 { 0.98 } else { 0.2 },
        confidence_label: confidence_label(if cag_score > 0.0 { 0.98 } else { 0.2 }),
        reason: "Query is about repository structure, configuration, or active implementation.".to_string(),
    }
}

pub fn route_query(manifest: &CagManifest, query: &str, top_k: usize) -> RoutingResult {
    let kw_weights = keyword_tag_weights();
    let query_lower = query.to_lowercase();
    let words: Vec<&str> = query_lower.split_whitespace().collect();

    let mut tag_scores: HashMap<String, f64> = HashMap::new();
    for word in &words {
        if let Some(weights) = kw_weights.get(word) {
            for (tag, weight) in weights {
                *tag_scores.entry(tag.to_string()).or_default() += weight;
            }
        }
    }

    let mut scored_tags: Vec<(String, f64)> = tag_scores.iter().map(|(tag, score)| (tag.clone(), *score)).collect();
    scored_tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let matched_tags: Vec<String> = scored_tags.into_iter().take(8).map(|(tag, _)| tag).collect();

    let mut file_scores: Vec<(usize, f64)> = manifest
        .files
        .iter()
        .enumerate()
        .map(|(index, file)| {
            let file_tags: HashSet<&str> = manifest
                .tags
                .iter()
                .filter(|(_, paths)| paths.contains(&file.path))
                .map(|(tag, _)| tag.as_str())
                .collect();

            let mut score = 0.0;
            for tag in &matched_tags {
                if file_tags.contains(tag.as_str()) {
                    score += tag_scores.get(tag).copied().unwrap_or(0.0);
                }
            }

            let path_lower = file.path.to_lowercase();
            for word in &words {
                if word.len() > 3 && path_lower.contains(word) {
                    score += 0.5;
                }
            }

            (index, score * get_path_multiplier(&file.path, &query_lower))
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    file_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    file_scores.truncate(top_k);

    let strategy = suggest_strategy(query);
    let mut total_tokens_est = 0usize;

    let matched_files: Vec<MatchedFile> = file_scores
        .into_iter()
        .map(|(index, score)| {
            let file = &manifest.files[index];
            total_tokens_est += file.content.len() / 4;
            let tags = manifest
                .tags
                .iter()
                .filter(|(_, paths)| paths.contains(&file.path))
                .map(|(tag, _)| tag.clone())
                .collect();

            MatchedFile {
                path: file.path.clone(),
                score: (score * 100.0).round() / 100.0,
                tags,
                snippet: file.content.chars().take(300).collect(),
            }
        })
        .collect();

    RoutingResult {
        query: query.to_string(),
        profile: manifest.profile.clone(),
        strategy: strategy.strategy,
        confidence: strategy.confidence,
        confidence_label: strategy.confidence_label,
        reason: strategy.reason,
        matched_tags,
        matched_files,
        total_tokens_est,
    }
}
