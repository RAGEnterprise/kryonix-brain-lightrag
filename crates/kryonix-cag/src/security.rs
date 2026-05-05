/// security.rs — Secret filtering and safety checks for CAG file content
use anyhow::Result;
use regex::Regex;
use std::sync::OnceLock;

/// Patterns that strongly indicate a line contains a secret
static SECRET_PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();

fn secret_patterns() -> &'static Vec<Regex> {
    SECRET_PATTERNS.get_or_init(|| {
        let patterns = [
            // Token prefixes
            r"ntn_[A-Za-z0-9]{20,}",
            r"ghp_[A-Za-z0-9]{20,}",
            r"github_pat_[A-Za-z0-9_]{20,}",
            r"sk-[A-Za-z0-9]{20,}",
            r"ghs_[A-Za-z0-9]{20,}",
            // PEM blocks — catch all variants: RSA, OPENSSH, EC, DSA, PRIVATE KEY, etc.
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
            r"-----BEGIN [A-Z ]*KEY-----",

            // Common env var names
            r"OPENAI_API_KEY\s*=\s*\S+",
            r"ANTHROPIC_API_KEY\s*=\s*\S+",
            r"NOTION_TOKEN\s*=\s*\S+",
            r"DATABASE_URL\s*=\s*(postgres|mysql|sqlite)://[^\s]+:[^\s]+@",
            // Generic secrets — use r#"..."# for patterns with double-quote chars
            r#"password\s*=\s*['"]?\S{8,}['"]?"#,
            r#"senha\s*=\s*['"]?\S{8,}['"]?"#,
            r#"secret\s*=\s*['"]?\S{8,}['"]?"#,
            r#"api_key\s*=\s*['"]?\S{8,}['"]?"#,
            // Tailscale auth keys
            r"tskey-[A-Za-z0-9-]{20,}",
            // JWT
            r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",
        ];
        patterns
            .iter()
            .filter_map(|p| Regex::new(p).ok())
            .collect()
    })
}

/// Returns true if the line appears to contain a secret
pub fn line_has_secret(line: &str) -> bool {
    let lower = line.to_lowercase();
    secret_patterns()
        .iter()
        .any(|re| re.is_match(line) || re.is_match(&lower))
}

/// Filter a file's content: redact lines that appear to contain secrets
pub fn filter_content(content: &str) -> (String, Vec<usize>) {
    let mut redacted_lines: Vec<usize> = Vec::new();
    let filtered: Vec<String> = content
        .lines()
        .enumerate()
        .map(|(i, line)| {
            if line_has_secret(line) {
                redacted_lines.push(i + 1);
                format!("[REDACTED:line:{}]", i + 1)
            } else {
                line.to_string()
            }
        })
        .collect();
    (filtered.join("\n"), redacted_lines)
}

/// Check a built CAG directory for any leaked secrets
/// Returns list of (path, line_numbers) where secrets were found
pub fn scan_cag_dir(dir: &std::path::Path) -> Result<Vec<(String, Vec<usize>)>> {
    let mut findings: Vec<(String, Vec<usize>)> = Vec::new();

    let manifest_path = dir.join("manifest.json");
    if manifest_path.exists() {
        let data = std::fs::read_to_string(&manifest_path)?;
        let (_, redacted) = filter_content(&data);
        if !redacted.is_empty() {
            findings.push(("manifest.json".to_string(), redacted));
        }
    }
    Ok(findings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_secret_in_normal_text() {
        assert!(!line_has_secret("hosts/glacier/default.nix"));
        assert!(!line_has_secret("This is a normal markdown line."));
    }

    #[test]
    fn test_github_token_detected() {
        assert!(line_has_secret("token = ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"));
    }

    #[test]
    fn test_notion_token_detected() {
        assert!(line_has_secret("NOTION_TOKEN = ntn_XYZABCDEFGHIJKLMNOPQRSTUVWXYZ"));
    }

    #[test]
    fn test_openai_key_detected() {
        assert!(line_has_secret("OPENAI_API_KEY = sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
    }

    #[test]
    fn test_pem_detected() {
        assert!(line_has_secret("-----BEGIN OPENSSH PRIVATE KEY-----"));
    }

    #[test]
    fn test_filter_content_redacts_secrets() {
        let content = "normal line\nghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ = token\nanother line";
        let (filtered, redacted) = filter_content(content);
        assert!(!redacted.is_empty());
        assert!(filtered.contains("[REDACTED:"));
        assert!(filtered.contains("normal line"));
        assert!(filtered.contains("another line"));
    }
}
