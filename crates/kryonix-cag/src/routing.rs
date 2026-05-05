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
        (
            "brain",
            vec![("brain", 2.0), ("lightrag", 1.5), ("mcp", 0.5)],
        ),
        ("rag", vec![("lightrag", 2.0), ("brain", 1.0)]),
        ("vault", vec![("brain", 1.0), ("docs", 0.8)]),
        (
            "bancos",
            vec![
                ("local-sources", 4.0),
                ("nixos-sources", 2.0),
                ("docs", 1.0),
            ],
        ),
        (
            "fontes",
            vec![
                ("local-sources", 4.0),
                ("nixos-sources", 2.0),
                ("docs", 1.0),
            ],
        ),
        (
            "locais",
            vec![
                ("local-sources", 3.5),
                ("nixos-sources", 1.5),
                ("docs", 0.8),
            ],
        ),
        ("local", vec![("local-sources", 2.5), ("docs", 0.5)]),
        ("source", vec![("local-sources", 2.0), ("docs", 0.5)]),
        (
            "sources",
            vec![
                ("local-sources", 3.5),
                ("nixos-sources", 2.0),
                ("docs", 1.0),
            ],
        ),
        (
            "nixos",
            vec![
                ("local-sources", 3.0),
                ("nixos-sources", 2.5),
                ("host-config", 1.5),
                ("nix", 1.0),
            ],
        ),
        (
            "nixpkgs",
            vec![("local-sources", 2.5), ("nix", 1.0), ("docs", 0.8)],
        ),
        (
            "home-manager",
            vec![("local-sources", 2.5), ("nix", 1.0), ("docs", 0.8)],
        ),
        (
            "noogle",
            vec![("local-sources", 2.5), ("nix", 0.8), ("docs", 0.8)],
        ),
        ("nvidia", vec![("gpu", 2.0), ("glacier", 1.0)]),
        ("cuda", vec![("gpu", 2.0), ("glacier", 1.0)]),
        (
            "ollama",
            vec![("ollama", 2.0), ("glacier", 1.0), ("brain", 0.5)],
        ),
        ("tailscale", vec![("tailscale", 2.0), ("networking", 1.0)]),
        ("ssh", vec![("ssh", 2.0), ("networking", 0.5)]),
        ("nix", vec![("nix", 2.0), ("flake", 0.5)]),
        ("flake", vec![("flake", 2.0), ("nix", 1.0)]),
        (
            "rebuild",
            vec![("nix", 1.5), ("host-config", 1.0), ("operations", 3.0)],
        ),
        (
            "switch",
            vec![("nix", 1.5), ("host-config", 1.0), ("operations", 3.0)],
        ),
        ("cli", vec![("cli", 2.5), ("docs", 1.0)]),
        ("comandos", vec![("operations", 1.5), ("cli", 1.0)]),
        ("comando", vec![("operations", 1.5), ("cli", 1.0)]),
        (
            "kryonix",
            vec![("cli", 2.5), ("operations", 1.5), ("docs", 1.0)],
        ),
        ("check", vec![("operations", 2.0), ("cli", 1.0)]),
        ("home", vec![("operations", 1.5), ("cli", 1.0)]),
        ("update", vec![("operations", 1.5), ("cli", 0.5)]),
        ("boot", vec![("operations", 1.5), ("cli", 0.5)]),
        ("test", vec![("operations", 1.5), ("cli", 0.5)]),
        ("apply", vec![("operations", 1.0), ("cli", 0.5)]),
        ("disko", vec![("storage", 2.0)]),
        ("disco", vec![("storage", 2.0)]),
        ("disk", vec![("storage", 2.0)]),
        ("iso", vec![("storage", 2.0)]),
        ("live", vec![("storage", 1.5)]),
        ("docs", vec![("docs", 1.5)]),
        ("roadmap", vec![("docs", 1.5)]),
        ("contrato", vec![("operations", 4.0), ("cli", 2.5)]),
        ("contract", vec![("operations", 4.0), ("cli", 2.5)]),
        ("canônico", vec![("operations", 3.5), ("cli", 2.0)]),
        ("canonico", vec![("operations", 3.5), ("cli", 2.0)]),
        ("canônica", vec![("operations", 3.5), ("cli", 2.0)]),
        ("canonica", vec![("operations", 3.5), ("cli", 2.0)]),
    ])
}


fn is_disk_query(query_lower: &str) -> bool {
    has_any(
        query_lower,
        &[
            "disco",
            "disk",
            "partição",
            "particao",
            "particionamento",
            "disko",
            "storage",
            "filesystem",
            "mount",
            "install",
            "instalação",
            "instalacao",
        ],
    )
}

fn is_iso_query(query_lower: &str) -> bool {
    has_any(query_lower, &["iso", "live", "usb", "bootable", "flash"])
}

fn is_archive_query(query_lower: &str) -> bool {
    has_any(
        query_lower,
        &[
            "antigo",
            "legacy",
            "archive",
            "histórico",
            "historico",
            "history",
            "vault",
        ],
    )
}

fn is_rebuild_query(query_lower: &str) -> bool {
    has_any(query_lower, &["rebuild", "switch", "seguro", "glacier"])
}

fn is_local_sources_query(query_lower: &str) -> bool {
    has_any(
        query_lower,
        &[
            "bancos",
            "fontes",
            "locais",
            "local",
            "source",
            "sources",
            "nixos",
            "nixpkgs",
            "home-manager",
            "noogle",
            "nix-dev",
            "onde estao",
            "onde estão",
            "onde fica",
            "onde ficam",
            "bancos locais",
            "fontes locais",
        ],
    )
}

fn is_operations_query(query_lower: &str) -> bool {
    has_any(
        query_lower,
        &[
            "cli",
            "kryonix",
            "operacional",
            "comando",
            "comandos",
            "boot",
            "test",
            "check",
            "fmt",
            "doctor",
            "switch",
            "rebuild",
            "home",
            "update",
            "apply",
            "contrato",
            "contract",
            "canônico",
            "canônica",
            "canonica",
        ],
    )
}

fn get_path_multiplier(path: &str, query_lower: &str) -> f64 {
    let path_lower = path.to_lowercase();
    let disk_query = is_disk_query(query_lower);
    let iso_query = is_iso_query(query_lower);
    let archive_query = is_archive_query(query_lower);
    let rebuild_query = is_rebuild_query(query_lower);
    let local_sources_query = is_local_sources_query(query_lower);
    let operations_query = is_operations_query(query_lower);

    if operations_query && path_lower.contains("docs/cli/kryonix_command_contract.md") {
        return 15.0;
    }

    if local_sources_query
        && (path_lower.contains("docs/ai/nixos-local-knowledge-sources.md")
            || path_lower.contains(".ai/skills/brain/nixos-local-sources.md"))
    {
        return 10.0;
    }

    if local_sources_query
        && (path_lower.contains("docs/ai/") || path_lower.contains(".ai/skills/brain/"))
    {
        return 6.0;
    }

    if local_sources_query
        && (path_lower.contains("hosts/glacier/default.nix")
            || path_lower.contains("profiles/glacier-ai.nix")
            || path_lower.contains("modules/nixos/services/brain.nix")
            || path_lower.contains("hardware-configuration.nix"))
    {
        return 0.2;
    }

    if rebuild_query
        && (path_lower.contains("docs/hosts/glacier-rebuild.md")
            || path_lower.contains("docs/hosts/glacier-switch.md")
            || path_lower.contains("docs/cli.md")
            || path_lower.contains("docs/operations.md")
            || path_lower.contains(".ai/skills/commands/rebuild-nixos.md"))
    {
        return 8.5;
    }

    if operations_query
        && (path_lower.contains("docs/cli.md")
            || path_lower.contains("docs/operations.md")
            || path_lower.contains("docs/hosts/glacier-rebuild.md")
            || path_lower.contains("docs/hosts/glacier-switch.md")
            || path_lower.contains(".ai/skills/commands/rebuild-nixos.md")
            || path_lower.contains("packages/kryonix-cli.nix"))
    {
        return 7.0;
    }

    if operations_query
        && (path_lower.contains("hosts/glacier/default.nix")
            || path_lower.contains("profiles/glacier-ai.nix")
            || path_lower.contains("modules/nixos/services/brain.nix")
            || path_lower.contains("hardware-configuration.nix"))
    {
        return 0.2;
    }

    if (disk_query || iso_query)
        && (path_lower.contains("glacier-live-iso.md")
            || path_lower.contains("disks.nix")
            || path_lower.contains("disko")
            || path_lower.contains("hardware-configuration"))
    {
        return if path_lower.contains("glacier-live-iso.md") {
            6.0
        } else {
            5.0
        };
    }

    if archive_query && path_lower.contains("archive/") {
        return 4.0;
    }

    if path_lower.contains("docs/hosts/")
        || path_lower.contains("docs/ai/")
        || path_lower.contains(".ai/skills/")
    {
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
    let rag_score = if has_any(
        &query_lower,
        &[
            "vault",
            "brain",
            "lightrag",
            "histórico",
            "historico",
            "incidente",
            "decisão",
            "grounding",
        ],
    ) {
        0.8
    } else {
        0.0
    };
    let cag_score = if has_any(
        &query_lower,
        &[
            "nix", "flake", "glacier", "inspiron", "bancos", "fontes", "locais", "cli", "comando",
            "rebuild", "switch", "kryonix", "check", "home", "update", "boot", "test",
        ],
    ) {
        0.9
    } else {
        0.0
    };

    if rag_score > 0.0 && cag_score > 0.0 {
        return StrategySuggestion {
            strategy: "hybrid".to_string(),
            confidence: 0.8,
            confidence_label: confidence_label(0.8),
            reason: "Query touches both repository implementation and vault/knowledge concepts."
                .to_string(),
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
        reason: "Query is about repository structure, configuration, or active implementation."
            .to_string(),
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

    let mut scored_tags: Vec<(String, f64)> = tag_scores
        .iter()
        .map(|(tag, score)| (tag.clone(), *score))
        .collect();
    scored_tags.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let matched_tags: Vec<String> = scored_tags
        .into_iter()
        .take(8)
        .map(|(tag, _)| tag)
        .collect();

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

#[cfg(test)]
mod tests {
    use super::route_query;
    use crate::manifest::{CagManifest, FileEntry, Profile};
    use chrono::Utc;
    use std::collections::HashMap;

    fn make_manifest(files: Vec<(&str, Vec<&str>)>) -> CagManifest {
        let entries: Vec<FileEntry> = files
            .iter()
            .map(|(path, _)| FileEntry {
                path: (*path).to_string(),
                size_bytes: 128,
                blake3: format!("hash-{}", path),
                content: format!("content for {}", path),
            })
            .collect();

        let mut tags: HashMap<String, Vec<String>> = HashMap::new();
        for (path, file_tags) in files {
            for tag in file_tags {
                tags.entry(tag.to_string())
                    .or_default()
                    .push(path.to_string());
            }
        }

        let total_bytes = entries.iter().map(|file| file.size_bytes).sum();

        CagManifest {
            version: 1,
            profile: Profile::kryonix_core().name,
            repo_root: "/etc/kryonix".into(),
            built_at: Utc::now(),
            total_files: entries.len(),
            total_bytes,
            content_hash: "test-hash".into(),
            files: entries,
            tags,
        }
    }

    #[test]
    fn test_route_local_sources_query_prefers_canonical_docs() {
        let manifest = make_manifest(vec![
            (
                "docs/ai/nixos-local-knowledge-sources.md",
                vec!["docs", "local-sources", "nixos-sources"],
            ),
            (
                ".ai/skills/brain/nixos-local-sources.md",
                vec!["docs", "local-sources", "nixos-sources"],
            ),
            (
                "hosts/glacier/default.nix",
                vec!["glacier", "host-config", "nix"],
            ),
        ]);

        let result = route_query(&manifest, "Onde estão os bancos locais NixOS?", 3);
        let paths: Vec<&str> = result
            .matched_files
            .iter()
            .map(|file| file.path.as_str())
            .collect();

        assert!(paths.contains(&"docs/ai/nixos-local-knowledge-sources.md"));
        assert!(paths.contains(&".ai/skills/brain/nixos-local-sources.md"));
        assert!(!paths.contains(&"hosts/glacier/default.nix"));
    }

    #[test]
    fn test_route_rebuild_query_prefers_cli_docs() {
        let manifest = make_manifest(vec![
            ("docs/hosts/glacier-rebuild.md", vec!["docs", "operations"]),
            ("docs/hosts/glacier-switch.md", vec!["docs", "operations"]),
            ("docs/CLI.md", vec!["docs", "cli", "operations"]),
            (
                "hosts/glacier/default.nix",
                vec!["glacier", "host-config", "nix"],
            ),
        ]);

        let result = route_query(&manifest, "Como faço rebuild seguro do Glacier?", 3);
        let paths: Vec<&str> = result
            .matched_files
            .iter()
            .map(|file| file.path.as_str())
            .collect();

        assert!(paths.contains(&"docs/hosts/glacier-rebuild.md"));
        assert!(paths.contains(&"docs/CLI.md"));
        assert!(!paths.contains(&"hosts/glacier/default.nix"));
    }
}
