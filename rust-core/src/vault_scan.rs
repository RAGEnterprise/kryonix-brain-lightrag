use std::path::{Path, PathBuf};
use std::collections::HashMap;
use sha1::{Sha1, Digest};
use walkdir::WalkDir;
use serde::Serialize;
use std::fs;

#[derive(Serialize)]
struct ScanResult {
    vault: String,
    files_total: usize,
    empty_files: Vec<String>,
    duplicate_hashes: HashMap<String, Vec<String>>,
    invalid_frontmatter: Vec<String>,
    ok: bool,
}

fn calculate_hash(content: &[u8]) -> String {
    let mut hasher = Sha1::new();
    hasher.update(content);
    let result = hasher.finalize();
    format!("{:x}", result)
}

fn check_frontmatter(content: &str) -> bool {
    if !content.starts_with("---") {
        return true; // No frontmatter is valid for this check
    }
    let parts: Vec<&str> = content.split("---").collect();
    if parts.len() < 3 {
        return false; // Incomplete frontmatter
    }
    // Basic YAML check could go here, but for now we just check delimiters
    true
}

fn main() {
    tracing_subscriber::fmt::init();
    
    let vault_path = std::env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let mut files_total = 0;
    let mut empty_files = Vec::new();
    let mut hashes: HashMap<String, Vec<String>> = HashMap::new();
    let mut invalid_frontmatter = Vec::new();

    for entry in WalkDir::new(&vault_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "md"))
    {
        files_total += 1;
        let path = entry.path();
        let rel_path = path.strip_prefix(&vault_path).unwrap_or(path).to_string_lossy().to_string();
        
        if let Ok(content) = fs::read(path) {
            if content.is_empty() {
                empty_files.push(rel_path.clone());
            } else {
                let hash = calculate_hash(&content);
                hashes.entry(hash).or_default().push(rel_path.clone());
                
                if let Ok(text) = String::from_utf8(content) {
                    if !check_frontmatter(&text) {
                        invalid_frontmatter.push(rel_path);
                    }
                }
            }
        }
    }

    let duplicate_hashes: HashMap<String, Vec<String>> = hashes
        .into_iter()
        .filter(|(_, paths)| paths.len() > 1)
        .collect();

    let result = ScanResult {
        vault: vault_path,
        files_total,
        empty_files,
        duplicate_hashes,
        invalid_frontmatter,
        ok: true,
    };

    println!("{}", serde_json::to_string_pretty(&result).unwrap());
}
