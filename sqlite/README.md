# @absolutejs/rag-sqlite

SQLite vector-store adapter for [`@absolutejs/rag`](https://github.com/absolutejs/rag),
with optional native `sqlite-vec` (vec0) acceleration. Falls back to the owned
JSON/sqlite-table path when no native platform binary is available.

## Install

```bash
bun add @absolutejs/rag @absolutejs/rag-sqlite
```

The native `vec0` accelerator ships as optional platform binaries (macOS arm64/x64,
Linux arm64/x64, Windows x64) that install automatically when supported.

## Usage

```ts
import { createSQLiteRAG, ragPlugin } from '@absolutejs/rag-sqlite';

const rag = createSQLiteRAG({
	storeOptions: {
		path: './rag.sqlite',
		native: { mode: 'vec0' }
	}
});

app.use(ragPlugin({ path: '/rag', collection: rag.collection }));

// Inspect what happened at runtime:
console.log(rag.getNativeSupport().actionableMessage);
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).
