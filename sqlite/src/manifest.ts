import { defineImplementation, defineManifest } from '@absolutejs/manifest';
import { Type } from '@sinclair/typebox';
import type { SQLiteRAGStoreOptions } from './types';

const MAX_DIMENSIONS = 16_000;

/* No serializable top-level config: the store IS the adapter instance, so all
 * knobs live on the rag/vector-store implementation below. */
export const manifest = defineManifest<Record<never, never>>()({
	contract: 1,
	identity: {
		accent: '#0f80cc',
		category: 'ai',
		description:
			'SQLite `RAGVectorStore` for `@absolutejs/rag` on Bun’s built-in `bun:sqlite` — a single file, no external database. Optional native sqlite-vec (vec0) acceleration via prebuilt platform packages, with a JSON fallback when the extension is unavailable.',
		docsUrl: 'https://github.com/absolutejs/rag-adapters/tree/main/sqlite',
		name: '@absolutejs/rag-sqlite',
		tagline: 'Keep your searchable content in a single file on your server.'
	},
	implements: [
		defineImplementation<SQLiteRAGStoreOptions>()({
			contract: 'rag/vector-store',
			factory: 'createSQLiteRAGStore',
			from: '@absolutejs/rag-sqlite',
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
				native: Type.Optional(
					Type.Object(
						{
							mode: Type.Literal('vec0', {
								default: 'vec0',
								title: 'Native mode'
							}),
							requireAvailable: Type.Optional(
								Type.Boolean({
									description:
										'Fail at startup if the native extension cannot load, instead of quietly falling back to the slower JSON mode.',
									title: 'Require native search'
								})
							)
						},
						{
							description:
								'Loads the sqlite-vec extension for much faster search on large content sets. Falls back to JSON mode when unavailable.',
							title: 'Native acceleration (sqlite-vec)'
						}
					)
				),
				path: Type.Optional(
					Type.String({
						description:
							'Database file on this machine. Created if missing; leave empty for a temporary in-memory database.',
						examples: ['./var/rag.sqlite'],
						title: 'Database file'
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
			title: 'SQLite (this machine, optional native vec0 acceleration)',
			wiring: {
				code: 'createSQLiteRAGStore(${settings})',
				imports: [
					{
						from: '@absolutejs/rag-sqlite',
						names: ['createSQLiteRAGStore']
					}
				]
			}
		})
	],
	settings: Type.Object({}),
	wiring: []
});
