# kryonix-brain-lightrag

Pacote **Python** de Retrieval-Augmented Generation baseado em grafos para o
ecossistema Kryonix. É o componente que implementa o **LightRAG** local: indexa
documentos, constrói grafo de conhecimento (GraphML) e serve recuperação para
o Kryonix Brain / agentes.

> Estado deste repo no workspace de desenvolvimento: o diretório de trabalho
> atual (`HEAD` desacoplado) está **vazio** — o código fonte vive nas branches
> `main` e nas branches de agente (`claude/optimistic-wozniak-*`). Este README
> descreve o propósito canônico do pacote; o código real deve ser puxado de
> `main` antes de build/contribuição.

## Papel no ecossistema

```
Ollama (LLM)  ──▶  kryonix-brain-lightrag (grafo RAG)  ──▶  MCP / Vault / Agentes
                      ↑
              documentos do Vault Obsidian + notas técnicas
```

- **Onde roda:** servidor `glacier` (IA pesada) ou invocado localmente por
  ferramentas de auditoria no `inspiron`.
- **Storage:** diretório `rag_storage` central (GraphML + vector DB).
- **Status canônico (vault):** VALIDATED via CLI (`kryonix graph stats
  --local`). API daemon na porta 8000 está desativada por padrão — uso é
  estrito via CLI/local até aprovação futura.

## Comandos de operação (via CLI do Kryonix, não direto deste repo)

```sh
kryonix graph stats  --local
kryonix graph top    --local --limit 10
kryonix graph heal   --local
kryonix graph repair --local
```

## Layout esperado (quando populado)

```
kryonix-brain-lightrag/
├── pyproject.toml        # empacotamento (packaging Python)
├── src/                  # módulos LightRAG (indexing, graph, query)
├── scripts/              # ingest do Vault, heal, repair
└── tests/                # validação de grafo
```

## Relação com outros sub-repos

- Consumido pelo motor (`kryonix`) via `packages/kryonix-brain-lightrag`.
- Alimenta o **Kryonix Brain** (Orquestração de IA: LightRAG + MCP + Vault +
  Ollama).
- Documentação derivada vive em
  `kryonix-vault/02-Areas/Kryonix/systems/LightRAG.md`.

## Licença

Source Available / Proprietário — Todos os Direitos Reservados (uso interno
Kryonix).
