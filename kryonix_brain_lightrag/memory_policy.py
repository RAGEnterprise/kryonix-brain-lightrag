"""Regras de captura de memória proativa para a Aura.

Este módulo decide se uma mensagem merece virar proposta no Vault,
classifica o tipo da memória e monta o corpo da nota proposta.
A escrita final continua segura: tudo vai para o inbox `00-inbox/ai-proposals`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Iterable

_EPHEMERAL_MARKERS = (
    "pr #",
    "pull request",
    "issue #",
    "commit",
    "sha",
    "phase",
    "todo",
    "checklist",
    "release-gate",
    "merge",
    "hotfix",
    "bugfix",
    "submitted pr",
    "completed task",
    "worked on",
)

_SECRET_MARKERS = (
    "api key",
    "apikey",
    "token",
    "secret",
    "senha",
    "password",
    "credencial",
    "credential",
    "bearer",
    "private key",
)

_KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "skill_note": (
        "skill",
        "skills",
        "workflow",
        "playbook",
        "prompt",
        "tool",
        "mcp",
        "function calling",
        "tool calling",
        "framework",
    ),
    "user_preference": (
        "prefere",
        "prefer",
        "gosta",
        "sempre",
        "nunca",
        "tom",
        "estilo",
        "voice",
        "conciso",
        "curto",
        "português",
    ),
    "environment_fact": (
        "path",
        "caminho",
        "arquivo",
        "config",
        "env",
        "host",
        "server",
        "linux",
        "nix",
        "workspace",
        "repo",
        "terminal",
    ),
    "project_fact": (
        "kryonix",
        "installer",
        "motor",
        "site",
        "vault",
        "brain",
        "flake",
        "module",
        "branch",
    ),
    "decision_record": (
        "decidido",
        "aprovado",
        "arquitetura",
        "estratégia",
        "plano",
        "policy",
        "contrato",
        "gate",
    ),
    "operational_rule": (
        "sempre",
        "nunca",
        "deve",
        "precisa",
        "regra",
        "guardrail",
        "seguro",
        "aprovação",
    ),
}

_DEFAULT_TITLE_BY_KIND = {
    "skill_note": "MOC_Minhas_Skills_Adicionais",
    "user_preference": "Preferencias_Do_Gabriel",
    "environment_fact": "Ambiente_Kryonix",
    "project_fact": "Memoria_De_Projeto",
    "decision_record": "Decisoes_Relevantes",
    "operational_rule": "Regras_Operacionais",
    "other": "Memoria_Autocapturada",
}


@dataclass(frozen=True, slots=True)
class MemoryCaptureDecision:
    """Resultado da triagem de memória."""

    should_capture: bool
    kind: str
    title: str
    reason: str
    confidence: float
    tags: tuple[str, ...]
    force: bool = False
    source: str = "assistant"

    def to_dict(self) -> dict[str, object]:
        return {
            "should_capture": self.should_capture,
            "kind": self.kind,
            "title": self.title,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "tags": list(self.tags),
            "force": self.force,
            "source": self.source,
        }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _contains_any(text: str, needles: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for needle in needles:
        if needle and needle in text:
            hits.append(needle)
    return hits


def _has_sensitive_content(text: str) -> bool:
    return bool(_contains_any(text, _SECRET_MARKERS))


def _score_kind(text: str) -> tuple[str, list[str], int]:
    best_kind = "other"
    best_hits: list[str] = []
    best_score = 0

    for kind, keywords in _KIND_KEYWORDS.items():
        hits = _contains_any(text, keywords)
        score = len(hits)
        if score > best_score:
            best_kind = kind
            best_hits = hits
            best_score = score

    return best_kind, best_hits, best_score


def _default_title(kind: str, title: str | None) -> str:
    if title and title.strip():
        return title.strip()
    return _DEFAULT_TITLE_BY_KIND.get(kind, _DEFAULT_TITLE_BY_KIND["other"])


def analyze_memory_capture(
    *,
    content: str,
    title: str | None = None,
    source: str = "assistant",
    reason: str = "",
    kind: str = "auto",
    tags: Iterable[str] | None = None,
    force: bool = False,
) -> MemoryCaptureDecision:
    """Classifica se o conteúdo merece proposta no Vault.

    Regras principais:
    - nunca captura segredos;
    - não captura progresso efêmero de tarefa;
    - prioriza factos duráveis, preferências, configuração, skills e decisões;
    - `force=True` só contorna o threshold, não os bloqueios de segurança.
    """

    raw_content = (content or "").strip()
    normalized = _normalize(" ".join([title or "", raw_content, reason or "", source or ""]))
    supplied_tags = [t.strip() for t in (tags or []) if str(t).strip()]

    if not raw_content:
        return MemoryCaptureDecision(
            should_capture=False,
            kind="other",
            title=_default_title("other", title),
            reason="conteúdo vazio",
            confidence=0.0,
            tags=tuple(supplied_tags),
            force=force,
            source=source,
        )

    if _has_sensitive_content(normalized):
        return MemoryCaptureDecision(
            should_capture=False,
            kind="other",
            title=_default_title("other", title),
            reason="conteúdo sensível/credencial detectado",
            confidence=0.0,
            tags=tuple(sorted({*supplied_tags, "sensitive"})),
            force=force,
            source=source,
        )

    if _contains_any(normalized, _EPHEMERAL_MARKERS):
        return MemoryCaptureDecision(
            should_capture=False,
            kind="other",
            title=_default_title("other", title),
            reason="conteúdo efêmero de execução/progresso",
            confidence=0.1,
            tags=tuple(sorted({*supplied_tags, "ephemeral"})),
            force=force,
            source=source,
        )

    detected_kind, kind_hits, kind_score = _score_kind(normalized)

    extra_hits = []
    for keyword in ("memo", "memory", "vault", "lembra", "recorda", "guardar", "salvar"):
        if keyword in normalized:
            extra_hits.append(keyword)

    score = kind_score + len(extra_hits)
    if force:
        score = max(score, 3)

    should_capture = score >= 2
    if kind != "auto" and kind in _KIND_KEYWORDS:
        detected_kind = kind

    if detected_kind == "other" and supplied_tags:
        if any(tag.lower() in {"skill", "skills", "workflow", "playbook"} for tag in supplied_tags):
            detected_kind = "skill_note"

    if detected_kind == "other" and any(term in normalized for term in ("claude skill", "claude skills", "notebooklm")):
        detected_kind = "skill_note"

    if detected_kind == "other" and any(term in normalized for term in ("prefer", "prefere", "gosta", "sempre", "nunca")):
        detected_kind = "user_preference"

    final_title = _default_title(detected_kind, title)
    if detected_kind == "skill_note" and final_title == _DEFAULT_TITLE_BY_KIND["skill_note"]:
        final_title = "MOC_Minhas_Skills_Adicionais"

    confidence = min(0.95, 0.25 + (0.2 * kind_score) + (0.1 * len(extra_hits)))
    if force:
        confidence = max(confidence, 0.85)

    reason_bits = ["triagem automática de memória"]
    if detected_kind != "other":
        reason_bits.append(f"tipo sugerido: {detected_kind}")
    if kind_hits:
        reason_bits.append(f"sinais: {', '.join(kind_hits)}")
    if extra_hits:
        reason_bits.append(f"gatilhos adicionais: {', '.join(extra_hits)}")
    if force:
        reason_bits.append("forçado pelo agente")

    return MemoryCaptureDecision(
        should_capture=should_capture,
        kind=detected_kind,
        title=final_title,
        reason="; ".join(reason_bits),
        confidence=confidence,
        tags=tuple(sorted({*supplied_tags, *kind_hits, *extra_hits})),
        force=force,
        source=source,
    )


def compose_memory_note(
    *,
    title: str,
    content: str,
    source: str,
    reason: str,
    kind: str,
    tags: Iterable[str] = (),
    confidence: float | None = None,
) -> str:
    """Monta o corpo markdown da nota proposta para o inbox."""

    tag_list = [t.strip() for t in tags if str(t).strip()]
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    sections = [
        f"# {title}",
        "",
        "## Metadados",
        f"- Criado em: {created_at}",
        f"- Tipo: {kind}",
        f"- Origem: {source}",
    ]
    if confidence is not None:
        sections.append(f"- Confiança: {confidence:.2f}")
    if tag_list:
        sections.append(f"- Tags: {', '.join(tag_list)}")

    sections.extend([
        "",
        "## Motivo da captura",
        reason or "Memória considerada relevante pela triagem automática.",
        "",
        "## Conteúdo a preservar",
        content.strip(),
        "",
        "## Próximo passo",
        "Promover esta proposta manualmente para a camada canônica quando fizer sentido.",
        "",
    ])

    return "\n".join(sections)
