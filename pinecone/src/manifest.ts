import { defineImplementation, defineManifest } from '@absolutejs/manifest';
import { Type } from '@sinclair/typebox';
import type { PineconeRAGOptions } from './index';

const MAX_DIMENSIONS = 20_000;

/* No serializable top-level config: the store IS the adapter instance, so all
 * knobs live on the rag/vector-store implementation below. */
export const manifest = defineManifest<Record<never, never>>()({
	contract: 1,
	identity: {
		accent: '#1c17ff',
		category: 'ai',
		description:
			'Pinecone `RAGVectorStore` for `@absolutejs/rag`. Serverless or pod indexes, namespaces, metadata filtering within Pinecone limits, and `ensurePineconeIndex` for first-run index creation. Deletes fall back to deterministic chunk ids where serverless cannot metadata-filter.',
		docsUrl: 'https://github.com/absolutejs/rag-adapters/tree/main/pinecone',
		name: '@absolutejs/rag-pinecone',
		tagline: 'Keep your searchable content in Pinecone.'
	},
	implements: [
		defineImplementation<PineconeRAGOptions>()({
			contract: 'rag/vector-store',
			factory: 'createPineconeStore',
			from: '@absolutejs/rag-pinecone',
			requires: {
				env: [
					{
						description: 'Pinecone API key',
						docsUrl: 'https://app.pinecone.io',
						example: 'pcsk_xxxxxxxxx',
						key: 'PINECONE_API_KEY',
						secret: true
					}
				],
				peers: [
					{
						name: '@pinecone-database/pinecone',
						range: '>=7.0.0 <8.0.0',
						reason: 'Pinecone SDK client'
					}
				],
				services: [
					{
						description:
							'Hosts the vector index your content is searched from',
						id: 'pinecone'
					}
				]
			},
			settings: Type.Object({
				indexName: Type.String({
					description:
						'The Pinecone index your content is stored in. Create it in the Pinecone console first (or with ensurePineconeIndex).',
					examples: ['my-site-content'],
					title: 'Index name'
				}),
				namespace: Type.Optional(
					Type.String({
						description:
							'Keeps this app’s content separate from anything else in the same index.',
						title: 'Namespace'
					})
				),
				vector: Type.Object(
					{
						dimensions: Type.Integer({
							description:
								'Must match your embedding provider’s output size (e.g. 1536 for OpenAI text-embedding-3-small).',
							maximum: MAX_DIMENSIONS,
							minimum: 1,
							title: 'Vector dimensions'
						}),
						distanceMetric: Type.Optional(
							Type.Union(
								[
									Type.Literal('cosine'),
									Type.Literal('euclidean'),
									Type.Literal('dotproduct')
								],
								{
									description:
										'How similarity is measured. Use what the index was created with; cosine is the usual choice.',
									title: 'Distance metric'
								}
							)
						),
						provider: Type.Literal('pinecone', {
							default: 'pinecone',
							title: 'Vector provider'
						})
					},
					{ title: 'Vector index' }
				)
			}),
			title: 'Pinecone',
			wiring: {
				code: 'createPineconeStore({ apiKey: ${env.PINECONE_API_KEY} ?? "", ...${settings} })',
				imports: [
					{
						from: '@absolutejs/rag-pinecone',
						names: ['createPineconeStore']
					}
				]
			}
		})
	],
	settings: Type.Object({}),
	wiring: []
});
