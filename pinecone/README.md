# @absolutejs/rag-pinecone

Pinecone vector-store adapter for [`@absolutejs/rag`](https://github.com/absolutejs/rag).
Treats Pinecone as the backend boundary: server-side filtering, native vector search,
and explicit opt-in index provisioning helpers.

## Install

```bash
bun add @absolutejs/rag @absolutejs/rag-pinecone @pinecone-database/pinecone
```

## Usage

```ts
import { createPineconeRAG } from '@absolutejs/rag-pinecone';

const rag = createPineconeRAG({
	apiKey: process.env.PINECONE_API_KEY,
	indexName: 'absolute-rag-demo',
	namespace: 'production',
	vector: {
		provider: 'pinecone',
		dimensions: 1536,
		distanceMetric: 'cosine'
	}
});

await rag.store.upsert({
	chunks: [
		{
			chunkId: 'doc-1#0',
			text: 'Pinecone stores vectors with attached metadata.',
			title: 'Pinecone overview',
			source: 'https://docs.pinecone.io',
			metadata: { tags: ['vector', 'managed'] }
		}
	]
});

const hits = await rag.collection.search({ query: 'vector database', topK: 4 });
```

### Index provisioning

The Pinecone index is **not** auto-provisioned at runtime. For explicit, opt-in
provisioning use the exported helpers:

```ts
import {
	describePineconeIndex,
	ensurePineconeIndex
} from '@absolutejs/rag-pinecone';

await ensurePineconeIndex({
	apiKey: process.env.PINECONE_API_KEY,
	indexName: 'absolute-rag-demo',
	dimensions: 1536,
	metric: 'cosine',
	waitUntilReady: true
});
```

## License

Apache License 2.0. See [LICENSE](./LICENSE).
