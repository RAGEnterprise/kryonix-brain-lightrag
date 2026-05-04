/// main.rs — kryonix-cag CLI binary
/// CAG = Context-Augmented Generation pack builder and router
mod manifest;
mod routing;
mod scanner;
mod security;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use manifest::{CagManifest, Profile};
use routing::route_query;
use scanner::scan_repo;
use security::scan_cag_dir;
use std::path::PathBuf;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "kryonix-cag",
    about = "Kryonix CAG — Context-Augmented Generation pack builder and router",
    version = env!("CARGO_PKG_VERSION"),
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a CAG pack from a repository
    Build {
        /// Profile to use (kryonix-core, kryonix-vault)
        #[arg(long, default_value = "kryonix-core")]
        profile: String,
        /// Repository root directory
        #[arg(long, default_value = "/etc/kryonix")]
        repo: PathBuf,
        /// Output directory for the CAG pack
        #[arg(long, default_value = "/tmp/kryonix-cag")]
        out: PathBuf,
    },
    /// Show status of an existing CAG pack
    Status {
        /// CAG pack directory
        #[arg(long, env = "LIGHTRAG_CAG_DIR", default_value = "/tmp/kryonix-cag")]
        dir: PathBuf,
    },
    /// Route a query to the most relevant files in the CAG pack
    Route {
        /// Query string
        query: String,
        /// CAG pack directory
        #[arg(long, env = "LIGHTRAG_CAG_DIR", default_value = "/tmp/kryonix-cag")]
        dir: PathBuf,
        /// Maximum number of files to return
        #[arg(long, default_value = "10")]
        top_k: usize,
        /// Output format: text (default) or json
        #[arg(long, default_value = "text")]
        format: String,
    },
    /// Clear the CAG pack cache directory
    ClearCache {
        /// CAG pack directory to clear
        #[arg(long, env = "LIGHTRAG_CAG_DIR", default_value = "/tmp/kryonix-cag")]
        dir: PathBuf,
    },
    /// List available profiles
    Profiles,
    /// Run security scan on a CAG directory
    Scan {
        /// CAG pack directory
        #[arg(long, env = "LIGHTRAG_CAG_DIR", default_value = "/tmp/kryonix-cag")]
        dir: PathBuf,
    },
}

fn main() -> Result<()> {
    // Initialize tracing — logs go to stderr, JSON output goes to stdout
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("kryonix_cag=info,warn")),
        )
        .with_writer(std::io::stderr)
        .without_time()
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Build { profile, repo, out } => cmd_build(&profile, &repo, &out),
        Commands::Status { dir } => cmd_status(&dir),
        Commands::Route { query, dir, top_k, format } => cmd_route(&query, &dir, top_k, &format),
        Commands::ClearCache { dir } => cmd_clear_cache(&dir),
        Commands::Profiles => cmd_profiles(),
        Commands::Scan { dir } => cmd_scan(&dir),
    }
}

fn cmd_build(profile_name: &str, repo: &PathBuf, out: &PathBuf) -> Result<()> {
    let profile = Profile::by_name(profile_name)
        .ok_or_else(|| anyhow::anyhow!(
            "Unknown profile '{}'. Use 'kryonix-cag profiles' to list available profiles.",
            profile_name
        ))?;

    info!("Building CAG pack: profile={} repo={}", profile_name, repo.display());

    if !repo.exists() {
        anyhow::bail!("Repository root does not exist: {}", repo.display());
    }

    let files = scan_repo(repo, &profile)
        .with_context(|| format!("Scanning repo at {}", repo.display()))?;

    info!("Scanned {} files", files.len());

    let manifest = CagManifest::from_scanned(&profile, repo, files);
    manifest.save(out)
        .with_context(|| format!("Saving manifest to {}", out.display()))?;

    // Print summary to stdout as JSON
    let summary = manifest.summary();
    println!("{}", serde_json::to_string_pretty(&summary)?);

    info!("CAG pack written to {}", out.display());
    Ok(())
}

fn cmd_status(dir: &PathBuf) -> Result<()> {
    if !dir.exists() {
        anyhow::bail!("CAG directory does not exist: {}", dir.display());
    }

    let manifest = CagManifest::load(dir)
        .with_context(|| format!("Loading manifest from {}", dir.display()))?;

    let summary = manifest.summary();
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

fn cmd_route(query: &str, dir: &PathBuf, top_k: usize, format: &str) -> Result<()> {
    let manifest = CagManifest::load(dir)
        .with_context(|| format!("Loading manifest from {}", dir.display()))?;

    let result = route_query(&manifest, query, top_k);

    match format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        _ => {
            // Text format — human-readable
            eprintln!("Query: {}", result.query);
            eprintln!("Matched tags: {}", result.matched_tags.join(", "));
            eprintln!("Top {} files (est. {} tokens):", result.matched_files.len(), result.total_tokens_est);
            for (i, f) in result.matched_files.iter().enumerate() {
                println!("[{}] {} (score={}, tags={})", i + 1, f.path, f.score, f.tags.join(","));
            }
        }
    }
    Ok(())
}

fn cmd_clear_cache(dir: &PathBuf) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(dir)
            .with_context(|| format!("Removing {}", dir.display()))?;
        info!("Cleared CAG cache: {}", dir.display());
        println!("{{\"status\":\"cleared\",\"dir\":\"{}\"}}", dir.display());
    } else {
        warn!("CAG directory does not exist, nothing to clear: {}", dir.display());
        println!("{{\"status\":\"not_found\",\"dir\":\"{}\"}}", dir.display());
    }
    Ok(())
}

fn cmd_profiles() -> Result<()> {
    let profiles = vec![
        Profile::kryonix_core(),
        Profile::kryonix_vault(),
    ];
    let list: Vec<serde_json::Value> = profiles.iter().map(|p| {
        serde_json::json!({
            "name": p.name,
            "description": p.description,
            "max_files": p.max_files,
        })
    }).collect();
    println!("{}", serde_json::to_string_pretty(&list)?);
    Ok(())
}

fn cmd_scan(dir: &PathBuf) -> Result<()> {
    let findings = scan_cag_dir(dir)?;
    if findings.is_empty() {
        println!("{{\"status\":\"clean\",\"secrets_found\":false}}");
    } else {
        let result = serde_json::json!({
            "status": "FAIL",
            "secrets_found": true,
            "findings": findings.iter().map(|(path, lines)| {
                serde_json::json!({"path": path, "lines": lines})
            }).collect::<Vec<_>>()
        });
        println!("{}", serde_json::to_string_pretty(&result)?);
        anyhow::bail!("Secret leak detected in CAG pack");
    }
    Ok(())
}
