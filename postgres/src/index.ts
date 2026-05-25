import type {
	RAGBackendCapabilities,
	RAGCollection,
	RAGVectorStore,
	RAGVectorStoreStatus
} from '@absolutejs/rag/adapter-kit';
import { createRAGCollection, ragPlugin } from '@absolutejs/rag/adapter-kit';
import { createPostgresRAGStore } from './createPostgresRAGStore';
import type { PostgresRAGStoreOptions } from './types';

export const ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME = '@absolutejs/rag-postgres';

export type PostgresRAGCollectionOptions = {
	store?: RAGVectorStore;
	storeOptions?: PostgresRAGStoreOptions;
};

export type PostgresRAGOptions = {
	store?: RAGVectorStore;
	collection?: RAGCollection;
	storeOptions?: PostgresRAGStoreOptions;
};

export type PostgresRAG = {
	store: RAGVectorStore;
	collection: RAGCollection;
	getStatus: () => RAGVectorStoreStatus | undefined;
	getCapabilities: () => RAGBackendCapabilities | undefined;
};

export const createPostgresRAGCollection = (
	options: PostgresRAGCollectionOptions = {}
): RAGCollection => {
	const store =
		options.store ?? createPostgresRAGStore(options.storeOptions ?? {});

	return createRAGCollection({ store });
};

export const createPostgresRAG = (
	options: PostgresRAGOptions = {}
): PostgresRAG => {
	const store =
		options.store ?? createPostgresRAGStore(options.storeOptions ?? {});
	const collection = options.collection ?? createRAGCollection({ store });

	return {
		store,
		collection,
		getStatus: () => collection.getStatus?.() ?? store.getStatus?.(),
		getCapabilities: () =>
			collection.getCapabilities?.() ?? store.getCapabilities?.()
	};
};

export const createPostgreSQLRAG: typeof createPostgresRAG = createPostgresRAG;

export { createPostgresRAGStore, createRAGCollection, ragPlugin };

export type {
	PostgresRAGStoreOptions,
	RAGBackendCapabilities,
	RAGCollection,
	RAGVectorStore,
	RAGVectorStoreStatus
};
