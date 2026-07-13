import { defineImplementation, defineManifest } from '@absolutejs/manifest';
import { Type } from '@sinclair/typebox';
import type { PostgresRAGStoreOptions } from './types';

const MAX_DIMENSIONS = 16_000;

/* No serializable top-level config: the store IS the adapter instance, so all
 * knobs live on the rag/vector-store implementation below. */
export const manifest = defineManifest<Record<never, never>>()({
	contract: 1,
	identity: {
		accent: '#336791',
		category: 'ai',
		description:
			'PostgreSQL (pgvector) `RAGVectorStore` for `@absolutejs/rag`, driven by Bun’s built-in SQL client — no extra driver. Creates the pgvector extension, table, and optional HNSW/IVFFlat index on first use; native vector similarity plus server-side metadata filtering.',
		docsUrl: 'https://github.com/absolutejs/rag-adapters/tree/main/postgres',
		name: '@absolutejs/rag-postgres',
		tagline: 'Keep your searchable content in your Postgres database.'
	},
	implements: [
		defineImplementation<PostgresRAGStoreOptions>()({
			contract: 'rag/vector-store',
			factory: 'createPostgresRAGStore',
			from: '@absolutejs/rag-postgres',
			requires: {
				env: [
					{
						description:
							'Postgres connection string (the content index table lives here; pgvector required)',
						example: 'postgres://user:pass@host/db',
						key: 'DATABASE_URL',
						secret: true
					}
				],
				services: [
					{
						description:
							'Stores the content index durably (needs the pgvector extension)',
						id: 'postgres'
					}
				]
			},
			settings: Type.Object({
				dimensions: Type.Optional(
					Type.Integer({
						description:
							'Must match your embedding provider’s output size (e.g. 1536 for OpenAI text-embedding-3-small).',
						maximum: MAX_DIMENSIONS,
						minimum: 1,
						title: 'Vector dimensions'
					})
				),
				distanceMetric: Type.Optional(
					Type.Union(
						[
							Type.Literal('cosine'),
							Type.Literal('l2'),
							Type.Literal('inner_product')
						],
						{
							description:
								'How similarity is measured. Cosine is the usual choice.',
							title: 'Distance metric'
						}
					)
				),
				indexType: Type.Optional(
					Type.Union(
						[
							Type.Literal('none'),
							Type.Literal('hnsw'),
							Type.Literal('ivfflat')
						],
						{
							description:
								'Speeds up search on large content sets. HNSW is the usual choice; none scans every row.',
							title: 'Vector index'
						}
					)
				),
				schemaName: Type.Optional(
					Type.String({
						description:
							'Postgres schema the table lives in. Default is public.',
						title: 'Schema name'
					})
				),
				tableName: Type.Optional(
					Type.String({
						description:
							'Table your indexed content is stored in. Default is rag_chunks.',
						title: 'Table name'
					})
				)
			}),
			title: 'Postgres (pgvector)',
			wiring: {
				code: 'createPostgresRAGStore({ connectionString: ${env.DATABASE_URL} ?? "", ...${settings} })',
				imports: [
					{
						from: '@absolutejs/rag-postgres',
						names: ['createPostgresRAGStore']
					}
				]
			}
		})
	],
	settings: Type.Object({}),
	wiring: []
});
