import os
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
from . import config

VAULT_ROOT = config.VAULT_DIR
BACKUP_DIR = VAULT_ROOT / ".backups"

def _ensure_backup_dir():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

def _safe_path(path: str) -> Path:
    """Validate and return a safe path within the vault."""
    # Convert to absolute path if it's not already
    p = Path(path)
    if not p.is_absolute():
        p = (VAULT_ROOT / p).resolve()
    else:
        p = p.resolve()
    
    # Check if the path is within the vault
    if not str(p).startswith(str(VAULT_ROOT.resolve())):
        raise PermissionError(f"Access denied: {path} is outside the vault.")
    
    return p

def _create_backup(path: Path):
    """Create a backup of a file before modification."""
    if not path.exists():
        return
    
    _ensure_backup_dir()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rel_path = path.relative_to(VAULT_ROOT)
    backup_name = f"{rel_path.name}.{timestamp}.bak"
    backup_path = BACKUP_DIR / backup_name
    
    shutil.copy2(path, backup_path)

def obsidian_status() -> Dict:
    """Return status of the Obsidian vault integration."""
    return {
        "vault_path": str(VAULT_ROOT),
        "exists": VAULT_ROOT.exists(),
        "is_directory": VAULT_ROOT.is_dir(),
        "notes_count": len(list(VAULT_ROOT.rglob("*.md"))) if VAULT_ROOT.exists() else 0
    }

def obsidian_list_notes() -> List[str]:
    """List all markdown notes in the vault."""
    if not VAULT_ROOT.exists():
        return []
    return [str(p.relative_to(VAULT_ROOT)) for p in VAULT_ROOT.rglob("*.md")]

def obsidian_read_note(path: str) -> str:
    """Read a note's content."""
    p = _safe_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Note not found: {path}")
    if p.suffix != ".md":
        raise ValueError("Only markdown files can be read as notes.")
    return p.read_text(encoding="utf-8")

def obsidian_write_note(path: str, content: str, backup: bool = True) -> str:
    """Write or overwrite a note."""
    p = _safe_path(path)
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    
    if backup:
        _create_backup(p)
    
    p.write_text(content, encoding="utf-8")
    return f"Note written successfully to {path}"

def obsidian_append_note(path: str, content: str) -> str:
    """Append content to an existing note."""
    p = _safe_path(path)
    if not p.exists():
        return obsidian_write_note(path, content, backup=False)
    
    _create_backup(p)
    with open(p, "a", encoding="utf-8") as f:
        f.write("\n" + content)
    return f"Content appended to {path}"

def obsidian_search_notes(query: str) -> List[Dict]:
    """Search for notes containing the query string."""
    results = []
    # Use ripgrep or simple glob/read
    # For now, simple implementation
    for p in VAULT_ROOT.rglob("*.md"):
        try:
            content = p.read_text(encoding="utf-8")
            if query.lower() in content.lower():
                results.append({
                    "path": str(p.relative_to(VAULT_ROOT)),
                    "snippet": _get_snippet(content, query)
                })
        except:
            continue
    return results

def _get_snippet(content: str, query: str, length: int = 150) -> str:
    idx = content.lower().find(query.lower())
    if idx == -1: return content[:length]
    start = max(0, idx - 75)
    end = min(len(content), idx + 75)
    return content[start:end].replace("\n", " ")

def obsidian_create_daily_note() -> str:
    """Create a daily note if it doesn't exist."""
    today = datetime.now().strftime("%Y-%m-%d")
    path = f"Daily/{today}.md"
    p = _safe_path(path)
    if p.exists():
        return f"Daily note already exists: {path}"
    
    content = f"# Daily Note - {today}\n\n## Tasks\n- [ ] \n\n## Notes\n"
    return obsidian_write_note(path, content, backup=False)

def obsidian_create_moc(title: str, links: List[str]) -> str:
    """Create a Map of Content (MOC)."""
    path = f"MOCs/{title}.md"
    content = f"# MOC: {title}\n\n## Links\n"
    for link in links:
        content += f"- [[{link}]]\n"
    return obsidian_write_note(path, content)

def obsidian_backlinks(note_path: str) -> List[str]:
    """Find notes that link to the given note."""
    # Get note name without extension
    note_name = Path(note_path).stem
    results = []
    # Search for [[note_name]]
    pattern = re.compile(rf"\[\[{re.escape(note_name)}(?:\|.*)?\]\]", re.IGNORECASE)
    
    for p in VAULT_ROOT.rglob("*.md"):
        try:
            content = p.read_text(encoding="utf-8")
            if pattern.search(content):
                results.append(str(p.relative_to(VAULT_ROOT)))
        except:
            continue
    return results

def obsidian_extract_links(note_path: str) -> List[str]:
    """Extract all internal links from a note."""
    content = obsidian_read_note(note_path)
    links = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", content)
    return list(set(links))

def obsidian_propose_note(title: str, content: str, source: str, reason: str) -> str:
    """
    Propose a note to be created in the vault inbox.
    Only writes to 00-inbox/ai-proposals.
    """
    # Import slugify here to avoid circular dependencies if any
    from .rag import slugify
    
    safe_title = slugify(title)
    if not safe_title.endswith(".md"):
        safe_title += ".md"
    
    prop_dir = config.VAULT_PROPOSAL_DIR
    prop_dir.mkdir(parents=True, exist_ok=True)
    
    p = prop_dir / safe_title
    
    # Frontmatter enforcement
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    frontmatter = f"""---
status: proposed
source: {source}
reason: {reason}
timestamp: {timestamp}
generated-by: kryonix-brain
---

"""
    # Ensure content starts with a newline if it doesn't already
    if not content.startswith("\n") and not content.startswith("#"):
        content = "\n" + content
        
    full_content = frontmatter + content
    
    p.write_text(full_content, encoding="utf-8")
    return f"Note proposed successfully at {p.relative_to(VAULT_ROOT)}"

def obsidian_validate_vault() -> Dict:
    """Validate vault accessibility and configuration."""
    status = obsidian_status()
    errors = []
    if not status["exists"]:
        errors.append(f"Vault directory does not exist: {status['vault_path']}")
    
    return {
        "valid": len(errors) == 0,
        "status": status,
        "errors": errors
    }
