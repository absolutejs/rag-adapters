import type {
	RAGBackendCapabilities,
	RAGCollection,
	RAGQueryInput,
	RAGQueryResult,
	RAGUpsertInput,
	RAGVectorStore,
	RAGVectorStoreStatus
} from '@absolutejs/rag';
import {
	createRAGCollection,
	createRAGVector,
	normalizeVector
} from '@absolutejs/rag';

export const ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME = '@absolutejs/rag-postgres';

export const POSTGRESQL_RAG_IMPLEMENTATIONS = ['pgvector'] as const;
export const PGVECTOR_DISTANCE_METRICS = [
	'cosine',
	'l2',
	'inner_product'
] as const;
export const PGVECTOR_INDEX_TYPES = ['none', 'hnsw', 'ivfflat'] as const;

export type PostgreSQLRAGVectorImplementation = 'pgvector';
export type PgvectorDistanceMetric = 'cosine' | 'l2' | 'inner_product';
export type PgvectorIndexType = 'none' | 'hnsw' | 'ivfflat';

export type PostgreSQLRAGClient = {
	query: <TRow = Record<string, unknown>>(
		sql: string,
		params?: unknown[]
	) => Promise<{
		rows: TRow[];
		rowCount?: number;
	}>;
	transaction?: <T>(
		run: (client: PostgreSQLRAGClient) => Promise<T>
	) => Promise<T>;
	close?: () => Promise<void>;
};

export type PostgreSQLRAGClientFactory = () =>
	| Promise<PostgreSQLRAGClient>
	| PostgreSQLRAGClient;

export type PostgreSQLRAGSchemaConfig = {
	schemaName?: string;
	chunkTableName?: string;
	migrationTableName?: string;
};

export type PgvectorHNSWConfig = {
	type: 'hnsw';
	m?: number;
	efConstruction?: number;
	efSearch?: number;
	iterativeScan?: 'off' | 'strict_order' | 'relaxed_order';
};

export type PgvectorIVFFlatConfig = {
	type: 'ivfflat';
	lists?: number;
	probes?: number;
	maxProbes?: number;
	iterativeScan?: 'off' | 'strict_order' | 'relaxed_order';
};

export type PgvectorNoIndexConfig = {
	type: 'none';
};

export type PgvectorIndexConfig =
	| PgvectorNoIndexConfig
	| PgvectorHNSWConfig
	| PgvectorIVFFlatConfig;

export type PgvectorConfig = {
	provider: 'pgvector';
	dimensions: number;
	distanceMetric?: PgvectorDistanceMetric;
	extensionName?: 'vector' | string;
	autoCreateExtension?: boolean;
	autoCreateSchema?: boolean;
	autoCreateTables?: boolean;
	autoCreateIndex?: boolean;
	index?: PgvectorIndexConfig;
};

export type PostgreSQLDriverOptions = {
	max?: number;
	prepare?: boolean;
	idle_timeout?: number;
	connect_timeout?: number;
	max_lifetime?: number;
	ssl?: boolean | 'require' | 'allow' | 'prefer' | 'verify-full';
};

export type PostgreSQLRAGOptions = {
	connectionString?: string;
	client?: PostgreSQLRAGClient;
	clientFactory?: PostgreSQLRAGClientFactory;
	driver?: PostgreSQLDriverOptions;
	schema?: PostgreSQLRAGSchemaConfig;
	vector: PgvectorConfig;
	embedding?: RAGVectorStore['embed'];
};

export type PostgreSQLSchemaPlan = {
	implementation: PostgreSQLRAGVectorImplementation;
	extensionSql: string[];
	schemaSql: string[];
	tableSql: string[];
	indexSql: string[];
	querySessionSql: string[];
	migrationTableQualifiedName: string;
};

export type PostgreSQLMigrationStage = 'extension' | 'table' | 'index';

export type PostgreSQLMigrationEntry = {
	name: string;
	stage: PostgreSQLMigrationStage;
	sql: string;
};

export type PostgreSQLMigrationPlan = {
	implementation: PostgreSQLRAGVectorImplementation;
	schemaName: string;
	migrationTableName: string;
	migrationTableQualifiedName: string;
	bootstrapSql: string[];
	migrations: PostgreSQLMigrationEntry[];
	schemaPlan: PostgreSQLSchemaPlan;
};

export type PostgreSQLApplyMigrationsOptions = {
	client?: PostgreSQLRAGClient;
	dryRun?: boolean;
};

export type PostgreSQLApplyMigrationsResult = {
	migrationPlan: PostgreSQLMigrationPlan;
	appliedNames: string[];
	skippedNames: string[];
	pendingNames: string[];
	appliedCount: number;
	pendingCount: number;
	dryRun: boolean;
};

export type PostgreSQLRAG = {
	store: RAGVectorStore;
	collection: RAGCollection;
	getStatus: () => RAGVectorStoreStatus | undefined;
	getCapabilities: () => RAGBackendCapabilities | undefined;
	getSchemaPlan: () => PostgreSQLSchemaPlan;
	getMigrationPlan: () => PostgreSQLMigrationPlan;
	applyMigrations: (
		options?: PostgreSQLApplyMigrationsOptions
	) => Promise<PostgreSQLApplyMigrationsResult>;
};

type ResolvedSchemaConfig = {
	schemaName: string;
	chunkTableName: string;
	migrationTableName: string;
};

type ResolvedPgvectorConfig = PgvectorConfig & {
	dimensions: number;
	distanceMetric: PgvectorDistanceMetric;
	extensionName: string;
	index: PgvectorIndexConfig;
};

type PgvectorDiagnostics = {
	fallbackReason: string | undefined;
	lastInitError: string | undefined;
	lastQueryError: string | undefined;
	lastUpsertError: string | undefined;
	lastMigrationError: string | undefined;
};

const DEFAULT_SCHEMA_NAME = 'absolute_rag';
const DEFAULT_CHUNK_TABLE_NAME = 'chunks';
const DEFAULT_MIGRATION_TABLE_NAME = 'migrations';
const DEFAULT_DIMENSIONS = 1536;

const IDENTIFIER_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

const assertIdentifier = (value: unknown, label: string): void => {
	if (typeof value !== 'string' || !IDENTIFIER_RE.test(value)) {
		throw new Error(
			`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: invalid ${label} "${String(
				value
			)}"`
		);
	}
};

const quoteIdentifier = (value: string): string => {
	assertIdentifier(value, 'identifier');

	return `"${value}"`;
};

const qualifiedTable = (schemaName: string, tableName: string): string =>
	`${quoteIdentifier(schemaName)}.${quoteIdentifier(tableName)}`;

const escapeLiteral = (value: string): string => value.replace(/'/g, "''");

const vectorLiteral = (vector: unknown): string => {
	if (!Array.isArray(vector) || vector.length === 0) {
		throw new Error(
			`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: vector values must be a non-empty array`
		);
	}

	return `[${vector
		.map((value) => {
			if (typeof value !== 'number' || !Number.isFinite(value)) {
				throw new Error(
					`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: vector values must be finite numbers`
				);
			}

			return String(value);
		})
		.join(',')}]`;
};

const makePlaceholder = (
	params: unknown[],
	value: unknown,
	cast = ''
): string => {
	params.push(value);
	const suffix = cast ? `::${cast}` : '';

	return `$${params.length}${suffix}`;
};

const normalizeMetric = (
	metric: PgvectorDistanceMetric | undefined
): PgvectorDistanceMetric => {
	if (metric === 'l2' || metric === 'inner_product') {
		return metric;
	}

	return 'cosine';
};

const normalizeIndex = (
	index: PgvectorIndexConfig | undefined
): PgvectorIndexConfig => {
	if (!index || index.type === undefined) {
		return { type: 'none' };
	}

	if (
		index.type === 'hnsw' ||
		index.type === 'ivfflat' ||
		index.type === 'none'
	) {
		return index;
	}

	throw new Error(
		`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: unsupported pgvector index type "${String(
			(index as { type: unknown }).type
		)}"`
	);
};

const resolveSchemaConfig = (
	options: Partial<PostgreSQLRAGOptions>
): ResolvedSchemaConfig => {
	const schemaName = options.schema?.schemaName ?? DEFAULT_SCHEMA_NAME;
	const chunkTableName =
		options.schema?.chunkTableName ?? DEFAULT_CHUNK_TABLE_NAME;
	const migrationTableName =
		options.schema?.migrationTableName ?? DEFAULT_MIGRATION_TABLE_NAME;

	assertIdentifier(schemaName, 'schema name');
	assertIdentifier(chunkTableName, 'chunk table name');
	assertIdentifier(migrationTableName, 'migration table name');

	return {
		schemaName,
		chunkTableName,
		migrationTableName
	};
};

const resolveVectorConfig = (
	options: Partial<PostgreSQLRAGOptions>
): ResolvedPgvectorConfig => {
	const vector = options?.vector;

	if (!vector || vector.provider !== 'pgvector') {
		throw new Error(
			`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: PostgreSQL RAG currently requires vector.provider = "pgvector"`
		);
	}

	const dimensions = vector.dimensions ?? DEFAULT_DIMENSIONS;
	if (!Number.isInteger(dimensions) || dimensions <= 0) {
		throw new Error(
			`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: dimensions must be a positive integer`
		);
	}

	const distanceMetric = normalizeMetric(vector.distanceMetric);
	const index = normalizeIndex(vector.index);

	return {
		...vector,
		dimensions,
		distanceMetric,
		extensionName: vector.extensionName ?? 'vector',
		index
	};
};

const operatorForMetric = (distanceMetric: PgvectorDistanceMetric): string => {
	switch (distanceMetric) {
		case 'l2':
			return '<->';
		case 'inner_product':
			return '<#>';
		case 'cosine':
		default:
			return '<=>';
	}
};

const operatorClassForMetric = (
	distanceMetric: PgvectorDistanceMetric
): string => {
	switch (distanceMetric) {
		case 'l2':
			return 'vector_l2_ops';
		case 'inner_product':
			return 'vector_ip_ops';
		case 'cosine':
		default:
			return 'vector_cosine_ops';
	}
};

const scoreFromDistance = (
	distance: unknown,
	distanceMetric: PgvectorDistanceMetric
): number => {
	if (typeof distance !== 'number' || !Number.isFinite(distance)) {
		return 0;
	}

	switch (distanceMetric) {
		case 'inner_product':
			return -distance;
		case 'l2':
			return 1 / (1 + Math.abs(distance));
		case 'cosine':
		default:
			return 1 - distance;
	}
};

const createIndexSql = ({
	schemaName,
	chunkTableName,
	distanceMetric,
	index
}: {
	schemaName: string;
	chunkTableName: string;
	distanceMetric: PgvectorDistanceMetric;
	index: PgvectorIndexConfig;
}): string[] => {
	if (!index || index.type === 'none') {
		return [];
	}

	const qualifiedChunkTable = qualifiedTable(schemaName, chunkTableName);
	const opClass = operatorClassForMetric(distanceMetric);
	const indexName = `${chunkTableName}_embedding_${index.type}_${distanceMetric}_idx`;
	const withParts: string[] = [];

	if (index.type === 'hnsw') {
		if (Number.isInteger(index.m) && (index.m ?? 0) > 0) {
			withParts.push(`m = ${index.m}`);
		}
		if (
			Number.isInteger(index.efConstruction) &&
			(index.efConstruction ?? 0) > 0
		) {
			withParts.push(`ef_construction = ${index.efConstruction}`);
		}
	}

	if (
		index.type === 'ivfflat' &&
		Number.isInteger(index.lists) &&
		(index.lists ?? 0) > 0
	) {
		withParts.push(`lists = ${index.lists}`);
	}

	const withClause =
		withParts.length > 0 ? ` WITH (${withParts.join(', ')})` : '';

	return [
		`CREATE INDEX IF NOT EXISTS ${quoteIdentifier(
			indexName
		)} ON ${qualifiedChunkTable} USING ${index.type} (embedding ${opClass})${withClause}`
	];
};

const createQuerySessionSql = ({
	index
}: {
	index: PgvectorIndexConfig;
}): string[] => {
	if (!index || index.type === 'none') {
		return [];
	}

	const sql: string[] = [];

	if (index.type === 'hnsw') {
		if (Number.isInteger(index.efSearch) && (index.efSearch ?? 0) > 0) {
			sql.push(`SET LOCAL hnsw.ef_search = ${index.efSearch}`);
		}
		if (index.iterativeScan && index.iterativeScan !== 'off') {
			sql.push(
				`SET LOCAL hnsw.iterative_scan = '${escapeLiteral(
					index.iterativeScan
				)}'`
			);
		}
	}

	if (index.type === 'ivfflat') {
		if (Number.isInteger(index.probes) && (index.probes ?? 0) > 0) {
			sql.push(`SET LOCAL ivfflat.probes = ${index.probes}`);
		}
		if (Number.isInteger(index.maxProbes) && (index.maxProbes ?? 0) > 0) {
			sql.push(`SET LOCAL ivfflat.max_probes = ${index.maxProbes}`);
		}
		if (index.iterativeScan && index.iterativeScan !== 'off') {
			sql.push(
				`SET LOCAL ivfflat.iterative_scan = '${escapeLiteral(
					index.iterativeScan
				)}'`
			);
		}
	}

	return sql;
};

const stageOrder: PostgreSQLMigrationStage[] = [
	'extension',
	'schema' as PostgreSQLMigrationStage,
	'table',
	'index'
];

const buildMigrationName = (
	stage: PostgreSQLMigrationStage,
	stageIndex: number,
	sql: string
): string => {
	const normalized =
		sql
			.toLowerCase()
			.replace(/[^a-z0-9]+/g, '_')
			.replace(/^_+|_+$/g, '')
			.slice(0, 48) || 'statement';
	const globalOrder = String(stageOrder.indexOf(stage) + 1).padStart(2, '0');
	const localOrder = String(stageIndex + 1).padStart(3, '0');

	return `${globalOrder}_${stage}_${localOrder}_${normalized}`;
};

const createMigrationTableSql = (
	schemaName: string,
	migrationTableName: string
): string =>
	`CREATE TABLE IF NOT EXISTS ${qualifiedTable(
		schemaName,
		migrationTableName
	)} (name TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`;

const filterTrackedTableSql = (
	tableSql: string[],
	schemaName: string,
	migrationTableName: string
): string[] => {
	const migrationTableTarget = qualifiedTable(schemaName, migrationTableName);

	return tableSql.filter((sql) => !sql.includes(migrationTableTarget));
};

export const createPostgresSchemaPlan = (
	options: PostgreSQLRAGOptions
): PostgreSQLSchemaPlan => {
	const schema = resolveSchemaConfig(options ?? {});
	const vector = resolveVectorConfig(options ?? {});
	const qualifiedChunkTable = qualifiedTable(
		schema.schemaName,
		schema.chunkTableName
	);
	const qualifiedMigrationTable = qualifiedTable(
		schema.schemaName,
		schema.migrationTableName
	);

	const extensionSql =
		vector.autoCreateExtension === false
			? []
			: [
					`CREATE EXTENSION IF NOT EXISTS ${quoteIdentifier(
						vector.extensionName
					)}`
				];

	const schemaSql =
		vector.autoCreateSchema === false
			? []
			: [
					`CREATE SCHEMA IF NOT EXISTS ${quoteIdentifier(
						schema.schemaName
					)}`
				];

	const tableSql =
		vector.autoCreateTables === false
			? []
			: [
					`CREATE TABLE IF NOT EXISTS ${qualifiedChunkTable} (id BIGSERIAL PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE, text TEXT NOT NULL, title TEXT, source TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, embedding VECTOR(${vector.dimensions}) NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())`,
					`CREATE INDEX IF NOT EXISTS ${quoteIdentifier(
						`${schema.chunkTableName}_chunk_id_idx`
					)} ON ${qualifiedChunkTable} (chunk_id)`,
					`CREATE INDEX IF NOT EXISTS ${quoteIdentifier(
						`${schema.chunkTableName}_source_idx`
					)} ON ${qualifiedChunkTable} (source)`,
					`CREATE INDEX IF NOT EXISTS ${quoteIdentifier(
						`${schema.chunkTableName}_metadata_idx`
					)} ON ${qualifiedChunkTable} USING GIN (metadata)`,
					createMigrationTableSql(
						schema.schemaName,
						schema.migrationTableName
					)
				];

	const indexSql =
		vector.autoCreateIndex === false
			? []
			: createIndexSql({
					schemaName: schema.schemaName,
					chunkTableName: schema.chunkTableName,
					distanceMetric: vector.distanceMetric,
					index: vector.index
				});

	return {
		implementation: 'pgvector',
		extensionSql,
		schemaSql,
		tableSql,
		indexSql,
		querySessionSql: createQuerySessionSql({ index: vector.index }),
		migrationTableQualifiedName: qualifiedMigrationTable
	};
};

export const createPostgresMigrationPlan = (
	options: PostgreSQLRAGOptions
): PostgreSQLMigrationPlan => {
	const schema = resolveSchemaConfig(options ?? {});
	const schemaPlan = createPostgresSchemaPlan(options ?? ({} as PostgreSQLRAGOptions));
	const bootstrapSql: string[] = [];

	if (schemaPlan.schemaSql.length > 0) {
		bootstrapSql.push(...schemaPlan.schemaSql);
	}

	const migrationTableSql = createMigrationTableSql(
		schema.schemaName,
		schema.migrationTableName
	);
	if (!bootstrapSql.includes(migrationTableSql)) {
		bootstrapSql.push(migrationTableSql);
	}

	const migrations: PostgreSQLMigrationEntry[] = [
		...schemaPlan.extensionSql.map((sql, index) => ({
			stage: 'extension' as PostgreSQLMigrationStage,
			sql,
			stageIndex: index
		})),
		...filterTrackedTableSql(
			schemaPlan.tableSql,
			schema.schemaName,
			schema.migrationTableName
		).map((sql, index) => ({
			stage: 'table' as PostgreSQLMigrationStage,
			sql,
			stageIndex: index
		})),
		...schemaPlan.indexSql.map((sql, index) => ({
			stage: 'index' as PostgreSQLMigrationStage,
			sql,
			stageIndex: index
		}))
	].map((entry) => ({
		name: buildMigrationName(entry.stage, entry.stageIndex, entry.sql),
		stage: entry.stage,
		sql: entry.sql
	}));

	return {
		implementation: schemaPlan.implementation,
		schemaName: schema.schemaName,
		migrationTableName: schema.migrationTableName,
		migrationTableQualifiedName: qualifiedTable(
			schema.schemaName,
			schema.migrationTableName
		),
		bootstrapSql,
		migrations,
		schemaPlan
	};
};

type RawPostgresClient = {
	unsafe: (
		queryText: string,
		params?: unknown[]
	) => Promise<unknown[] & { count?: number }>;
	begin: <T>(run: (transactionSql: RawPostgresClient) => Promise<T>) => Promise<T>;
	end?: (options?: { timeout?: number }) => Promise<void>;
};

const createWrappedPostgresClient = (
	sql: RawPostgresClient,
	rootSql: RawPostgresClient = sql
): PostgreSQLRAGClient => ({
	query: async <TRow = Record<string, unknown>>(
		queryText: string,
		params: unknown[] = []
	) => {
		const rows = await sql.unsafe(queryText, params);

		return {
			rows: rows as TRow[],
			rowCount: typeof rows.count === 'number' ? rows.count : rows.length
		};
	},
	transaction: async <T>(run: (client: PostgreSQLRAGClient) => Promise<T>) =>
		rootSql.begin(async (transactionSql) =>
			run(createWrappedPostgresClient(transactionSql, transactionSql))
		),
	close: async () => {
		if (typeof rootSql.end === 'function') {
			await rootSql.end({ timeout: 5 });
		}
	}
});

const createDefaultPostgresClientFactory = (
	options: PostgreSQLRAGOptions
): (() => Promise<PostgreSQLRAGClient>) | undefined => {
	const connectionString =
		typeof options.connectionString === 'string'
			? options.connectionString.trim()
			: '';

	if (connectionString.length === 0) {
		return undefined;
	}

	let clientPromise: Promise<PostgreSQLRAGClient> | undefined;

	return async () => {
		if (!clientPromise) {
			clientPromise = (async () => {
				const postgresModule = await import('postgres');
				const postgres = postgresModule.default;
				const sql = postgres(connectionString, {
					onnotice: () => {},
					...(options.driver ?? {})
				}) as unknown as RawPostgresClient;

				return createWrappedPostgresClient(sql, sql);
			})();
		}

		return clientPromise;
	};
};

const resolveClientFactory = (
	options: PostgreSQLRAGOptions
): (() => Promise<PostgreSQLRAGClient>) => {
	if (typeof options.clientFactory === 'function') {
		const { clientFactory } = options;

		return async () => clientFactory();
	}

	if (options.client) {
		const { client } = options;

		return async () => client;
	}

	const defaultFactory = createDefaultPostgresClientFactory(options);
	if (defaultFactory) {
		return defaultFactory;
	}

	return async () => {
		throw new Error(
			`${ABSOLUTE_POSTGRESQL_RAG_PACKAGE_NAME}: createPostgresRAG requires connectionString, client, or clientFactory.`
		);
	};
};

const buildMetadataFilter = (
	filter: Record<string, unknown> | undefined
): Record<string, unknown> | undefined => {
	if (!filter) {
		return undefined;
	}

	const metadataEntries = Object.entries(filter).filter(
		([key]) => key !== 'chunkId' && key !== 'title' && key !== 'source'
	);

	if (metadataEntries.length === 0) {
		return undefined;
	}

	return Object.fromEntries(metadataEntries);
};

const parseMetadataValue = (
	value: unknown
): Record<string, unknown> | undefined => {
	if (value === null || value === undefined) {
		return undefined;
	}

	if (typeof value === 'string') {
		try {
			const parsed = JSON.parse(value);
			if (parsed && typeof parsed === 'object') {
				return parsed as Record<string, unknown>;
			}
		} catch {
			return undefined;
		}
	}

	if (typeof value === 'object') {
		return value as Record<string, unknown>;
	}

	return undefined;
};

const createPgvectorStoreStatus = ({
	vector,
	schema,
	diagnostics,
	initialized
}: {
	vector: ResolvedPgvectorConfig;
	schema: ResolvedSchemaConfig;
	diagnostics: PgvectorDiagnostics;
	initialized: boolean;
}): RAGVectorStoreStatus =>
	({
		backend: 'postgres',
		vectorMode: 'native_pgvector',
		dimensions: vector.dimensions,
		native: {
			requested: true,
			available: initialized && !diagnostics.lastInitError,
			active: initialized && !diagnostics.lastInitError,
			mode: 'pgvector',
			extensionName: vector.extensionName,
			schemaName: schema.schemaName,
			tableName: schema.chunkTableName,
			distanceMetric: vector.distanceMetric,
			indexType: vector.index.type,
			fallbackReason: diagnostics.fallbackReason,
			lastInitError: diagnostics.lastInitError,
			lastQueryError: diagnostics.lastQueryError,
			lastUpsertError: diagnostics.lastUpsertError,
			lastMigrationError: diagnostics.lastMigrationError
		}
	}) as unknown as RAGVectorStoreStatus;

const getAppliedMigrationNames = async (
	client: PostgreSQLRAGClient,
	migrationPlan: PostgreSQLMigrationPlan
): Promise<Set<string>> => {
	const result = await client.query<{ name: unknown }>(
		`SELECT name FROM ${migrationPlan.migrationTableQualifiedName} ORDER BY name ASC`
	);

	return new Set(result.rows.map((row) => String(row.name)));
};

const insertAppliedMigration = async (
	client: PostgreSQLRAGClient,
	migrationPlan: PostgreSQLMigrationPlan,
	name: string
): Promise<void> => {
	await client.query(
		`INSERT INTO ${migrationPlan.migrationTableQualifiedName} (name) VALUES ($1) ON CONFLICT (name) DO NOTHING`,
		[name]
	);
};

const executeMigrationSequence = async (
	client: PostgreSQLRAGClient,
	migrationPlan: PostgreSQLMigrationPlan,
	migrations: PostgreSQLMigrationEntry[]
): Promise<string[]> => {
	const appliedNames: string[] = [];

	for (const migration of migrations) {
		await client.query(migration.sql);
		await insertAppliedMigration(client, migrationPlan, migration.name);
		appliedNames.push(migration.name);
	}

	return appliedNames;
};

export const applyPostgresMigrations = async (
	options: PostgreSQLRAGOptions,
	applyOptions: PostgreSQLApplyMigrationsOptions = {}
): Promise<PostgreSQLApplyMigrationsResult> => {
	const migrationPlan = createPostgresMigrationPlan(
		options ?? ({} as PostgreSQLRAGOptions)
	);
	const injectedClient = applyOptions.client;
	const getClient = injectedClient
		? async () => injectedClient
		: resolveClientFactory(options ?? ({} as PostgreSQLRAGOptions));
	const client = await getClient();

	for (const sql of migrationPlan.bootstrapSql) {
		await client.query(sql);
	}

	const alreadyApplied = await getAppliedMigrationNames(client, migrationPlan);
	const pendingMigrations = migrationPlan.migrations.filter(
		(migration) => !alreadyApplied.has(migration.name)
	);
	const skippedNames = migrationPlan.migrations
		.filter((migration) => alreadyApplied.has(migration.name))
		.map((migration) => migration.name);

	if (applyOptions.dryRun === true) {
		return {
			migrationPlan,
			appliedNames: [],
			skippedNames,
			pendingNames: pendingMigrations.map((migration) => migration.name),
			appliedCount: 0,
			pendingCount: pendingMigrations.length,
			dryRun: true
		};
	}

	const run = async (
		activeClient: PostgreSQLRAGClient
	): Promise<PostgreSQLApplyMigrationsResult> => {
		const names = await executeMigrationSequence(
			activeClient,
			migrationPlan,
			pendingMigrations
		);

		return {
			migrationPlan,
			appliedNames: names,
			skippedNames,
			pendingNames: pendingMigrations.map((migration) => migration.name),
			appliedCount: names.length,
			pendingCount: pendingMigrations.length,
			dryRun: false
		};
	};

	if (
		typeof client.transaction === 'function' &&
		pendingMigrations.length > 0
	) {
		return client.transaction(async (transactionClient) =>
			run(transactionClient)
		);
	}

	return run(client);
};

export const applyPostgresSchemaPlan: typeof applyPostgresMigrations =
	applyPostgresMigrations;

export const createPgvectorStore = (
	options: PostgreSQLRAGOptions
): RAGVectorStore => {
	const vector = resolveVectorConfig(options ?? {});
	const schema = resolveSchemaConfig(options ?? {});
	const plan = createPostgresSchemaPlan(options ?? ({} as PostgreSQLRAGOptions));
	const getClient = resolveClientFactory(options ?? ({} as PostgreSQLRAGOptions));
	const diagnostics: PgvectorDiagnostics = {
		fallbackReason: undefined,
		lastInitError: undefined,
		lastQueryError: undefined,
		lastUpsertError: undefined,
		lastMigrationError: undefined
	};
	let initialized = false;
	let initPromise: Promise<void> | undefined;

	const ensureInitialized = async (): Promise<void> => {
		if (initialized) {
			return;
		}

		if (!initPromise) {
			initPromise = (async () => {
				try {
					const client = await getClient();
					await applyPostgresMigrations(options ?? {}, { client });
					initialized = true;
					diagnostics.lastInitError = undefined;
					diagnostics.lastMigrationError = undefined;
					diagnostics.fallbackReason = undefined;
				} catch (error) {
					initialized = false;
					const message =
						error instanceof Error ? error.message : String(error);
					diagnostics.lastInitError = message;
					diagnostics.lastMigrationError = message;
					diagnostics.fallbackReason = message;
					throw error;
				}
			})();
		}

		return initPromise;
	};

	const embed: RAGVectorStore['embed'] = async (input) => {
		if (typeof options.embedding === 'function') {
			const result = await options.embedding(input);

			return normalizeVector(result);
		}

		return normalizeVector([
			...createRAGVector(input.text, vector.dimensions)
		]);
	};

	const query = async (input: RAGQueryInput): Promise<RAGQueryResult[]> => {
		await ensureInitialized();
		const client = await getClient();
		const params: unknown[] = [];
		const qualifiedChunkTable = qualifiedTable(
			schema.schemaName,
			schema.chunkTableName
		);
		const operator = operatorForMetric(vector.distanceMetric);
		const vectorPlaceholder = makePlaceholder(
			params,
			vectorLiteral(normalizeVector(input.queryVector)),
			'vector'
		);
		const limitPlaceholder = makePlaceholder(params, input.topK);
		const whereParts: string[] = [];
		const filter = input.filter as Record<string, unknown> | undefined;

		if (filter?.chunkId !== undefined) {
			whereParts.push(
				`chunk_id = ${makePlaceholder(params, filter.chunkId)}`
			);
		}
		if (filter?.title !== undefined) {
			whereParts.push(`title = ${makePlaceholder(params, filter.title)}`);
		}
		if (filter?.source !== undefined) {
			whereParts.push(
				`source = ${makePlaceholder(params, filter.source)}`
			);
		}

		const metadataFilter = buildMetadataFilter(filter);
		if (metadataFilter) {
			whereParts.push(
				`metadata @> ${makePlaceholder(
					params,
					JSON.stringify(metadataFilter),
					'jsonb'
				)}`
			);
		}

		const whereSql =
			whereParts.length > 0 ? `WHERE ${whereParts.join(' AND ')}` : '';
		const sessionSql = plan.querySessionSql;
		const selectSql = `SELECT chunk_id, text, title, source, metadata, embedding ${operator} ${vectorPlaceholder} AS distance FROM ${qualifiedChunkTable} ${whereSql} ORDER BY distance ASC LIMIT ${limitPlaceholder}`;

		try {
			for (const sql of sessionSql) {
				await client.query(sql);
			}

			const result = await client.query<{
				chunk_id: string;
				text: string;
				title: string | null;
				source: string | null;
				metadata: unknown;
				distance: unknown;
			}>(selectSql, params);

			return result.rows.map((row) => ({
				chunkId: row.chunk_id,
				chunkText: row.text,
				title: row.title ?? undefined,
				source: row.source ?? undefined,
				metadata: parseMetadataValue(row.metadata),
				score: scoreFromDistance(
					Number(row.distance),
					vector.distanceMetric
				)
			}));
		} catch (error) {
			diagnostics.lastQueryError =
				error instanceof Error ? error.message : String(error);
			throw error;
		}
	};

	const upsert = async (input: RAGUpsertInput): Promise<void> => {
		await ensureInitialized();
		const client = await getClient();
		const qualifiedChunkTable = qualifiedTable(
			schema.schemaName,
			schema.chunkTableName
		);
		const sql = `INSERT INTO ${qualifiedChunkTable} (chunk_id, text, title, source, metadata, embedding, updated_at) VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, NOW()) ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, title = EXCLUDED.title, source = EXCLUDED.source, metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding, updated_at = NOW()`;

		try {
			for (const chunk of input.chunks) {
				const vectorValue =
					Array.isArray(chunk.embedding) && chunk.embedding.length > 0
						? normalizeVector(chunk.embedding)
						: await embed({ text: chunk.text });
				await client.query(sql, [
					chunk.chunkId,
					chunk.text,
					chunk.title ?? null,
					chunk.source ?? null,
					JSON.stringify(chunk.metadata ?? {}),
					vectorLiteral(vectorValue)
				]);
			}
		} catch (error) {
			diagnostics.lastUpsertError =
				error instanceof Error ? error.message : String(error);
			throw error;
		}
	};

	const clear = async (): Promise<void> => {
		await ensureInitialized();
		const client = await getClient();
		const qualifiedChunkTable = qualifiedTable(
			schema.schemaName,
			schema.chunkTableName
		);
		await client.query(`DELETE FROM ${qualifiedChunkTable}`);
	};

	return {
		embed,
		query,
		upsert,
		clear,
		getCapabilities: () => ({
			backend: 'postgres',
			persistence: 'external',
			nativeVectorSearch: true,
			serverSideFiltering: true,
			streamingIngestStatus: false
		}),
		getStatus: () =>
			createPgvectorStoreStatus({
				vector,
				schema,
				diagnostics,
				initialized
			})
	} as RAGVectorStore;
};

export const createPostgresRAGCollection = (
	options: PostgreSQLRAGOptions
): RAGCollection =>
	createRAGCollection({
		store: createPgvectorStore(options)
	});

export const createPostgresRAG = (
	options: PostgreSQLRAGOptions
): PostgreSQLRAG => {
	const store = createPgvectorStore(options);
	const collection = createRAGCollection({ store });
	const schemaPlan = createPostgresSchemaPlan(options);
	const migrationPlan = createPostgresMigrationPlan(options);

	return {
		store,
		collection,
		getStatus: () => store.getStatus?.(),
		getCapabilities: () => store.getCapabilities?.(),
		getSchemaPlan: () => schemaPlan,
		getMigrationPlan: () => migrationPlan,
		applyMigrations: (applyOptions) =>
			applyPostgresMigrations(options, applyOptions)
	};
};

export const createPostgreSQLRAG: typeof createPostgresRAG = createPostgresRAG;
