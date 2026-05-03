# Kryonix Brain — Experimental Rust Layer

Este diretório contém uma implementação experimental em Rust para o Kryonix Brain.

## Status Atual: **EXPERIMENTAL / WIP**

A API principal do Kryonix Brain continua sendo a implementação em **Python (FastAPI)** localizada em `kryonix_brain_lightrag/api.py`.

### O que foi implementado em Rust:
- [x] Servidor Axum básico (`src/main.rs`).
- [x] Scanner de Segurança de alta performance (`src/utils.rs`).
- [x] Algoritmo de expansão de grafos multi-hop (`src/lib.rs`).
- [x] Bridge inicial com PyO3 para interoperabilidade com Python.

## Por que Rust?
- **Performance**: Processamento de grafos e vetores em larga escala.
- **Segurança de Memória**: Redução de bugs de concorrência no servidor de API.
- **Tipagem Forte**: Contratos de API mais rigorosos.

## Como Desenvolver / Testar
Para entrar no ambiente de desenvolvimento:
```bash
nix-shell packages/kryonix-brain-lightrag/shell.nix
```

Para compilar o binário experimental:
```bash
cargo build
```

## Roadmap de Estabilização
1.  [ ] Resolver dependência de linking dinâmica com interpretador Python no NixOS.
2.  [ ] Migrar todos os endpoints do FastAPI para Axum.
3.  [ ] Implementar bridge assíncrono robusto para o loop de eventos do Python.
4.  [ ] Validar em ambiente de produção (Glacier).

---
**IMPORTANTE**: Não altere `pyproject.toml` para usar `maturin` como build-backend principal até que o build do Rust esteja 100% estável e reprodutível em CI/CD.
