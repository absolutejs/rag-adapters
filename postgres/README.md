# @absolutejs/rag-postgres

PostgreSQL storage for AbsoluteJS RAG with pgvector similarity, native full-text
retrieval, metadata filtering and reusable plugin integration.

```bash
bun add @absolutejs/rag @absolutejs/rag-postgres
```

```ts
import { Elysia } from 'elysia';
import { createPostgresRAG } from '@absolutejs/rag-postgres';
import { ragPlugin } from '@absolutejs/rag';

const rag = createPostgresRAG({
  storeOptions: {
    connectionString: process.env.DATABASE_URL,
    dimensions: 1536,
    indexType: 'hnsw',
    lexicalMode: 'native',
  },
});
const app = new Elysia().use(ragPlugin({ path: '/rag', collection: rag.collection }));
```

The driver is Bun.SQL; no separate PostgreSQL client package is required.
Use your model provider to generate embeddings with the configured dimensions.

## Lexical retrieval

Native mode is the default. First use creates a GIN expression index over title,
text, source and JSON metadata string values. Existing tables are indexed too;
index creation can take time and block writes on a populated table, so initialize
it during a planned migration window. Updates and deletions maintain the index
through PostgreSQL.

Queries use PostgreSQL's `simple` configuration, match any query lexeme, rank
with weighted `ts_rank_cd`, and break ties by chunk ID. They are parameterized;
query punctuation is treated as text. Title, body, source and metadata receive
successively lower weights. Only the requested top K rows cross into the app.
Tenant and other supported metadata predicates run before ranking/limiting.
Unsupported filters are rejected rather than partially applied. Native top K
must be an integer between 0 and 10,000.

`lexicalMode: 'portable'` retains the existing RAG lexical scorer and its richer
field-specific ranking. It loads filtered candidates into application memory.
The engines have different tokenization and scores; native mode is not a claim
of identical relevance ordering. Compare your corpus before relying on score
thresholds. Vector retrieval remains unchanged.

## Measured retrieval

[The reproducible benchmark](./benchmarks/lexical.ts) uses a deterministic local
PostgreSQL 15 corpus at 1K, 10K and 100K rows, one selective query with tenant
filtering, one warmup and ten trials. It verifies expected matches and retains
`EXPLAIN ANALYZE` plans in [the results](./benchmarks/lexical-results.json).
At 100K rows, native median/p95 were 2.08/2.79 ms versus 683.03/752.49 ms for
portable scoring. Native transferred 10 rows; portable transferred 50,000.
These are local warm-query measurements, excluding ingestion/index construction,
provider calls and network deployment effects. The 1K fixture has no matching
row in the selected tenant. This is not an Exa comparison or a web-scale benchmark.

```bash
RAG_LEXICAL_TEST_URL=postgres://... bun postgres/benchmarks/lexical.ts
RAG_LEXICAL_TEST_URL=postgres://... bun test postgres/tests
```

Run these from the repository root against an isolated test database. Fixture
schemas are removed after execution. CI runs real PostgreSQL integration tests.

## License

Apache-2.0.
