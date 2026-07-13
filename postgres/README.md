# @absolutejs/rag-postgres

PostgreSQL vector-store adapter for [`@absolutejs/rag`](https://github.com/absolutejs/rag),
with `pgvector` as the first vector implementation. Native vector search, server-side
filtering, and inspectable schema/migration plans.

## Install

```bash
bun add @absolutejs/rag @absolutejs/rag-postgres postgres
```

## Usage

```ts
import { createPostgresRAG } from '@absolutejs/rag-postgres';
import { ragPlugin } from '@absolutejs/rag';

const rag = createPostgresRAG({
	connectionString: process.env.DATABASE_URL,
	vector: {
		provider: 'pgvector',
		dimensions: 1536,
		distanceMetric: 'cosine',
		autoCreateExtension: true,
		autoCreateSchema: true,
		autoCreateTables: true,
		autoCreateIndex: true,
		index: { type: 'hnsw', efSearch: 100, efConstruction: 64, m: 16 }
	},
	schema: { schemaName: 'absolute_rag', chunkTableName: 'chunks' }
});

app.use(ragPlugin({ path: '/rag', collection: rag.collection }));
```

### Schema and migrations

Inspect the generated SQL or apply migrations explicitly:

```ts
const schemaPlan = rag.getSchemaPlan();
const migrationPlan = rag.getMigrationPlan();
await rag.applyMigrations();
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).
