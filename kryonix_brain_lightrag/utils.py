import re
from pathlib import Path
from datetime import datetime

class SecretScanner:
    # Patterns for common secrets and sensitive data
    PATTERNS = [
        r"(?i)api[-_]?key",
        r"(?i)secret",
        r"(?i)token",
        r"(?i)password",
        r"(?i)pwd",
        r"(?i)ssh[-_]?key",
        r"(?i)private[-_]?key",
        r"BEGIN\s+(?:RSA|OPENSSH|PRIVATE)\s+KEY",
        r"ghp_[a-zA-Z0-9]{36}", # GitHub PAT
    ]

    @classmethod
    def scan_and_redact(cls, text: str) -> tuple[str, list[str]]:
        findings = []
        redacted = text
        
        # Robust redaction for "key: value" or "key = value"
        # Includes common patterns and prefixes
        redact_pattern = r"(?i)(api[-_]?key|secret|token|password|pwd|ssh[-_]?key|private[-_]?key|key)[\s:=]+([^\s,;]+)"
        
        matches = re.findall(redact_pattern, text)
        for label, val in matches:
            findings.append(label)
            
        redacted = re.sub(redact_pattern, r"\1: [REDACTED]", redacted)
        
        # Detect block keys (like SSH private keys)
        for block_pattern in [r"BEGIN\s+(?:RSA|OPENSSH|PRIVATE)\s+KEY"]:
            if re.search(block_pattern, text):
                findings.append("SENSITIVE_BLOCK_KEY")
                redacted = "[REDACTED: SENSITIVE BLOCK KEY DETECTED]"
                
        return redacted, findings

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
