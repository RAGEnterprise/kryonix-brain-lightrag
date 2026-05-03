use regex::Regex;
use lazy_static::lazy_static;

lazy_static! {
    static ref REDACT_PATTERN: Regex = Regex::new(r"(?i)(api[-_]?key|secret|token|password|pwd|ssh[-_]?key|private[-_]?key|key)[\s:=]+([^\s,;]+)").unwrap();
}

pub struct SecretScanner;

impl SecretScanner {
    pub fn scan_and_redact(text: &str) -> (String, Vec<String>) {
        let mut findings = Vec::new();
        
        for cap in REDACT_PATTERN.captures_iter(text) {
            if let Some(label) = cap.get(1) {
                findings.push(label.as_str().to_lowercase());
            }
        }
        
        let redacted = REDACT_PATTERN.replace_all(text, "$1: [REDACTED]").to_string();
        
        (redacted, findings)
    }
}
