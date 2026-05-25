// Configuration shapes for the SQLite RAG store. The store is built on Bun's
// built-in `bun:sqlite` driver and can optionally activate the native
// sqlite-vec (vec0) backend for accelerated vector search.

import type { Database } from "bun:sqlite";

export type NativeSQLiteRAGStoreOptions = {
	mode: "vec0";
	extensionPath?: string;
	extensionInitSql?: string | string[];
	distanceMetric?: "cosine" | "l2";
	tableName?: string;
	queryMultiplier?: number;
	requireAvailable?: boolean;
	resolveFromAbsolutePackages?: boolean;
};

export type SQLiteRAGStoreOptions = {
	db?: Database;
	path?: string;
	dimensions?: number;
	mockEmbedding?: (text: string) => Promise<number[]>;
	tableName?: string;
	native?: NativeSQLiteRAGStoreOptions;
};
