# LightRAG Knowledge Graph

Pipeline LightRAG para indexar, consultar e exportar o knowledge graph do projeto.

## Uso rápido

```bash
# Ativar o ambiente virtual
# Windows:
tools\lightrag\.venv\Scripts\activate
# Linux/macOS:
source tools/lightrag/.venv/bin/activate

# Indexar o projeto
rag index --full

# Consultar
rag search "como funciona X"
rag stats

# Exportar pro Obsidian
rag export --clean
```

## Comandos disponíveis

| Comando | Descrição |
|---------|-----------|
| `rag search "termo"` | Busca híbrida com síntese e citações |
| `rag ask "pergunta"` | Alias para search |
| `rag chunks "termo"` | Busca vetorial (sem síntese) |
| `rag local "termo"` | Vizinhança de entidades |
| `rag global "tema"` | Comunidades/temas |
| `rag stats` | Estatísticas do grafo |
| `rag top [N]` | Top-N entidades por conexões |
| `rag find "entidade"` | Procura entidade (substring) |
| `rag show "entidade"` | Detalhes + vizinhos |
| `rag index --full` | Rebuild completo |
| `rag index --incremental` | Só arquivos alterados |
| `rag export --clean` | Re-sync Obsidian |
| `rag insert "texto"` | Inserir texto manualmente |
| `rag shell` | REPL interativo |
| `rag mcp-check` | Validar .mcp.json |

Todos aceitam `--json` para output machine-readable.
