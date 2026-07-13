# @absolutejs/rag-adapters

Vector-store adapters for [`@absolutejs/rag`](https://github.com/absolutejs/rag) — a
private workspace monorepo. Each adapter is published as its own package and depends
on `@absolutejs/rag`; install only the one for your backend.

| Adapter | Package | Backend | Status |
| --- | --- | --- | --- |
| `pinecone/` | `@absolutejs/rag-pinecone` | Pinecone | ✅ |
| `postgres/` | `@absolutejs/rag-postgres` | PostgreSQL/pgvector | ✅ |
| `sqlite/` | `@absolutejs/rag-sqlite` | SQLite + optional native vec0 | ✅ |

## Why a monorepo

Adapters share the `RAGVectorStore` contract from `@absolutejs/rag`. Keeping them in
one workspace means a contract change can update every adapter in a single PR, with one
CI and one `bun install` — while each still publishes independently so consumers pull
only the backend (and its heavy deps) they actually use.

## Develop

```bash
bun install          # installs every workspace member
bun run typecheck    # across all adapters
bun run test         # across all adapters
bun run build        # across all adapters
```

## License

Apache License 2.0 — each published subpackage ships its own LICENSE.
