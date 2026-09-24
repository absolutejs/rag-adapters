// Configuration shape for the PostgreSQL (pgvector) RAG store. The store driver
// is Bun's built-in SQL client (`Bun.SQL`), so the connection can be provided
// either as a connection string or a ready-made `Bun.SQL` instance.

export type PostgresRAGStoreOptions = {
	/** Native uses a GIN full-text index; portable retains the in-memory lexical scorer. */
	lexicalMode?: "native" | "portable";
	connectionString?: string;
	sql?: InstanceType<typeof Bun.SQL>;
	dimensions?: number;
	mockEmbedding?: (text: string) => Promise<number[]>;
	tableName?: string;
	schemaName?: string;
	distanceMetric?: "cosine" | "l2" | "inner_product";
	queryMultiplier?: number;
	indexType?: "none" | "hnsw" | "ivfflat";
	indexLists?: number;
	hnswM?: number;
	hnswEfConstruction?: number;
};
