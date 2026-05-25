import type {
	RAGBackendCapabilities,
	RAGCollection,
	RAGQueryInput,
	RAGQueryResult,
	RAGUpsertInput,
	RAGVectorStore
} from '@absolutejs/rag';
import {
	createRAGCollection,
	createRAGVector,
	normalizeVector
} from '@absolutejs/rag';

export const ABSOLUTE_PINECONE_RAG_PACKAGE_NAME = '@absolutejs/rag-pinecone';

export const PINECONE_DISTANCE_METRICS = [
	'cosine',
	'euclidean',
	'dotproduct'
] as const;

export type PineconeDistanceMetric = 'cosine' | 'euclidean' | 'dotproduct';

export type PineconeIndexClient = {
	upsert: (records: unknown[]) => Promise<unknown>;
	query: (input: Record<string, unknown>) => Promise<{
		matches?: Array<{
			id: string;
			score?: number;
			values?: number[];
			metadata?: Record<string, unknown>;
		}>;
	}>;
	fetch: (ids: string[]) => Promise<{
		records?: Record<string, unknown>;
	}>;
	deleteMany: (input: string[] | Record<string, unknown>) => Promise<unknown>;
	deleteAll: () => Promise<unknown>;
	describeIndexStats: () => Promise<{
		totalRecordCount?: number;
		namespaces?: Record<string, { recordCount?: number }>;
	}>;
	namespace?: (namespace: string) => PineconeIndexClient;
};

export type PineconeRAGVectorConfig = {
	provider: 'pinecone';
	dimensions: number;
	distanceMetric?: PineconeDistanceMetric;
};

export type PineconeRAGOptions = {
	apiKey?: string;
	indexName?: string;
	indexHost?: string;
	namespace?: string;
	client?: PineconeIndexClient;
	vector: PineconeRAGVectorConfig;
	embedding?: RAGVectorStore['embed'];
};

export type PineconeRAG = {
	store: RAGVectorStore;
	collection: RAGCollection;
	getCapabilities: () => RAGBackendCapabilities | undefined;
};

export type PineconeServerlessSpec = {
	serverless: {
		cloud: 'aws' | 'gcp' | 'azure';
		region: string;
	};
};

export type PineconePodSpec = {
	pod: {
		environment: string;
		podType: string;
		pods?: number;
		replicas?: number;
		shards?: number;
		metadataConfig?: { indexed?: string[] };
		sourceCollection?: string;
	};
};

export type PineconeIndexSpec = PineconeServerlessSpec | PineconePodSpec;

export type PineconeIndexDescription = {
	name: string;
	dimension: number;
	metric: PineconeDistanceMetric;
	host?: string;
	spec?: PineconeIndexSpec;
	status?: { ready?: boolean; state?: string };
	deletionProtection?: 'enabled' | 'disabled';
};

export type DescribePineconeIndexOptions = {
	apiKey?: string;
	indexName: string;
};

export type EnsurePineconeIndexOptions = {
	apiKey?: string;
	indexName: string;
	dimensions: number;
	metric?: PineconeDistanceMetric;
	spec?: PineconeIndexSpec;
	deletionProtection?: 'enabled' | 'disabled';
	waitUntilReady?: boolean;
	waitTimeoutMs?: number;
	pollIntervalMs?: number;
};

export type EnsurePineconeIndexResult = {
	created: boolean;
	description: PineconeIndexDescription | undefined;
};

type ResolvedPineconeVectorConfig = {
	provider: 'pinecone';
	dimensions: number;
	distanceMetric: PineconeDistanceMetric;
};

type MetadataValue = string | number | boolean | string[];

type PineconeRecord = {
	id: string;
	values: number[];
	metadata: Record<string, MetadataValue>;
};

type PineconeQueryMatch = {
	id: string;
	score?: number;
	values?: number[];
	metadata?: Record<string, unknown>;
};

type PineconeFilter = Record<string, unknown>;

const PKG = ABSOLUTE_PINECONE_RAG_PACKAGE_NAME;
const DEFAULT_DIMENSIONS = 1536;
const DEFAULT_DISTANCE_METRIC: PineconeDistanceMetric = 'cosine';
const PINECONE_UPSERT_BATCH_SIZE = 100;
const PINECONE_FETCH_BATCH_SIZE = 1000;
const PINECONE_FILTERED_COUNT_TOPK = 10000;
const RESERVED_METADATA_KEYS = new Set(['chunkId', 'text', 'title', 'source']);

const isObjectRecord = (value: unknown): value is Record<string, unknown> =>
	Boolean(value) && typeof value === 'object' && !Array.isArray(value);

const isOperatorRecord = (value: unknown): value is Record<string, unknown> =>
	isObjectRecord(value) &&
	Object.keys(value).length > 0 &&
	Object.keys(value).every((key) => key.startsWith('$'));

const chunkArray = <T>(input: readonly T[], size: number): T[][] => {
	const out: T[][] = [];
	for (let i = 0; i < input.length; i += size) {
		out.push(input.slice(i, i + size));
	}

	return out;
};

const resolveVectorConfig = (
	options: Partial<PineconeRAGOptions>
): ResolvedPineconeVectorConfig => {
	const vector = options?.vector;
	if (!vector || vector.provider !== 'pinecone') {
		throw new Error(
			`${PKG}: Pinecone RAG requires vector.provider = "pinecone"`
		);
	}
	const dimensions = vector.dimensions ?? DEFAULT_DIMENSIONS;
	if (!Number.isInteger(dimensions) || dimensions <= 0) {
		throw new Error(
			`${PKG}: vector.dimensions must be a positive integer (received ${String(
				dimensions
			)})`
		);
	}
	const distanceMetric = vector.distanceMetric ?? DEFAULT_DISTANCE_METRIC;
	if (!PINECONE_DISTANCE_METRICS.includes(distanceMetric)) {
		throw new Error(
			`${PKG}: unsupported distanceMetric "${distanceMetric}". Allowed: ${PINECONE_DISTANCE_METRICS.join(
				', '
			)}`
		);
	}

	return { provider: 'pinecone', dimensions, distanceMetric };
};

type PineconeSDKModule = {
	Pinecone: new (config: { apiKey: string }) => {
		index: (name: string, host?: string) => PineconeIndexClient;
		createIndex: (input: Record<string, unknown>) => Promise<unknown>;
		describeIndex: (name: string) => Promise<PineconeIndexDescription>;
	};
};

let pineconeModulePromise: Promise<PineconeSDKModule> | undefined;
const loadPineconeSDK = (): Promise<PineconeSDKModule> => {
	if (!pineconeModulePromise) {
		pineconeModulePromise = (
			import('@pinecone-database/pinecone') as Promise<unknown> as Promise<PineconeSDKModule>
		).catch((error: unknown) => {
			pineconeModulePromise = undefined;
			throw new Error(
				`${PKG}: failed to load @pinecone-database/pinecone — install it as a dependency. (${
					error instanceof Error ? error.message : String(error)
				})`
			);
		});
	}

	return pineconeModulePromise;
};

const resolveIndexClientFactory = (
	options: PineconeRAGOptions
): (() => Promise<PineconeIndexClient>) => {
	const namespace =
		typeof options.namespace === 'string' && options.namespace.length > 0
			? options.namespace
			: undefined;

	let cached: PineconeIndexClient | undefined;

	const applyNamespace = (idx: PineconeIndexClient): PineconeIndexClient => {
		if (!namespace || typeof idx.namespace !== 'function') return idx;

		return idx.namespace(namespace);
	};

	if (options.client) {
		const injectedClient = options.client;

		return async () => {
			if (cached) return cached;
			cached = applyNamespace(injectedClient);

			return cached;
		};
	}

	const indexName = options.indexName;
	if (typeof indexName !== 'string' || indexName.length === 0) {
		throw new Error(
			`${PKG}: indexName is required when client is not provided`
		);
	}
	const apiKey = options.apiKey ?? process.env.PINECONE_API_KEY;
	if (!apiKey) {
		throw new Error(
			`${PKG}: missing Pinecone apiKey (pass options.apiKey or set PINECONE_API_KEY)`
		);
	}

	return async () => {
		if (cached) return cached;
		const { Pinecone } = await loadPineconeSDK();
		const pc = new Pinecone({ apiKey });
		const baseIndex = options.indexHost
			? pc.index(indexName, options.indexHost)
			: pc.index(indexName);
		cached = applyNamespace(baseIndex);

		return cached;
	};
};

const sanitizeMetadataValue = (value: unknown): MetadataValue | undefined => {
	if (value === null || value === undefined) return undefined;
	if (typeof value === 'string' || typeof value === 'boolean') return value;
	if (typeof value === 'number') {
		return Number.isFinite(value) ? value : undefined;
	}
	if (Array.isArray(value)) {
		const strings: string[] = [];
		for (const entry of value) {
			if (entry === null || entry === undefined) continue;
			if (typeof entry === 'string') {
				strings.push(entry);
				continue;
			}
			if (typeof entry === 'number' && Number.isFinite(entry)) {
				strings.push(String(entry));
				continue;
			}
			if (typeof entry === 'boolean') {
				strings.push(String(entry));
				continue;
			}
		}

		return strings;
	}
	if (typeof value === 'object') {
		try {
			return JSON.stringify(value);
		} catch {
			return undefined;
		}
	}

	return undefined;
};

const sanitizeMetadata = (
	metadata: unknown
): Record<string, MetadataValue> => {
	if (!isObjectRecord(metadata)) return {};
	const out: Record<string, MetadataValue> = {};
	for (const [key, value] of Object.entries(metadata)) {
		if (RESERVED_METADATA_KEYS.has(key)) continue;
		const sanitized = sanitizeMetadataValue(value);
		if (sanitized === undefined) continue;
		out[key] = sanitized;
	}

	return out;
};

const translateOperatorFilter = (
	field: string,
	operatorRecord: Record<string, unknown>
): PineconeFilter | undefined => {
	const direct: Record<string, unknown> = {};
	const expanded: PineconeFilter[] = [];
	for (const [op, value] of Object.entries(operatorRecord)) {
		switch (op) {
			case '$eq':
			case '$ne':
			case '$gt':
			case '$gte':
			case '$lt':
			case '$lte':
				direct[op] = value;
				break;
			case '$in':
				if (Array.isArray(value)) direct.$in = value;
				break;
			case '$exists':
				direct.$exists = Boolean(value);
				break;
			case '$contains':
				direct.$in = [value];
				break;
			case '$containsAny':
				if (Array.isArray(value)) direct.$in = value;
				break;
			case '$containsAll':
				if (Array.isArray(value)) {
					for (const entry of value) {
						expanded.push({ [field]: { $in: [entry] } });
					}
				}
				break;
			default:
				throw new Error(`${PKG}: unsupported filter operator "${op}"`);
		}
	}
	if (expanded.length === 0) {
		return Object.keys(direct).length > 0 ? { [field]: direct } : undefined;
	}
	if (Object.keys(direct).length === 0) {
		return expanded.length === 1 ? expanded[0] : { $and: expanded };
	}

	return { $and: [{ [field]: direct }, ...expanded] };
};

const translateFilter = (filter: unknown): PineconeFilter | undefined => {
	if (!isObjectRecord(filter) || Object.keys(filter).length === 0) {
		return undefined;
	}
	const clauses: PineconeFilter[] = [];
	for (const [key, value] of Object.entries(filter)) {
		if (key === '$and' || key === '$or') {
			if (!Array.isArray(value)) continue;
			const subs = value
				.map((entry) => translateFilter(entry))
				.filter(
					(entry): entry is PineconeFilter => entry !== undefined
				);
			if (subs.length > 0) clauses.push({ [key]: subs });
			continue;
		}
		if (key === '$not') {
			throw new Error(`${PKG}: $not is not supported by Pinecone`);
		}
		if (key.includes('.')) {
			throw new Error(
				`${PKG}: nested key paths ("${key}") are not supported by Pinecone (metadata is flat)`
			);
		}
		if (isOperatorRecord(value)) {
			const translated = translateOperatorFilter(key, value);
			if (translated) clauses.push(translated);
			continue;
		}
		clauses.push({ [key]: { $eq: value } });
	}
	if (clauses.length === 0) return undefined;
	if (clauses.length === 1) return clauses[0];

	return { $and: clauses };
};

const scoreForMetric = (
	rawScore: unknown,
	distanceMetric: PineconeDistanceMetric
): number => {
	if (typeof rawScore !== 'number' || !Number.isFinite(rawScore)) return 0;
	switch (distanceMetric) {
		case 'euclidean':
			return 1 / (1 + Math.abs(rawScore));
		case 'dotproduct':
		case 'cosine':
		default:
			return rawScore;
	}
};

const buildPineconeRecord = (
	chunk: RAGUpsertInput['chunks'][number],
	values: number[]
): PineconeRecord => {
	const metadata: Record<string, MetadataValue> = {
		...sanitizeMetadata(chunk.metadata),
		chunkId: chunk.chunkId,
		text: chunk.text
	};
	if (chunk.title) metadata.title = chunk.title;
	if (chunk.source) metadata.source = chunk.source;

	return { id: chunk.chunkId, values, metadata };
};

const extractQueryResult = (
	match: PineconeQueryMatch,
	distanceMetric: PineconeDistanceMetric
): RAGQueryResult => {
	const meta = isObjectRecord(match.metadata) ? match.metadata : {};
	const userMetadata: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(meta)) {
		if (RESERVED_METADATA_KEYS.has(key)) continue;
		userMetadata[key] = value;
	}
	const chunkId =
		typeof meta.chunkId === 'string' ? meta.chunkId : String(match.id);
	const chunkText = typeof meta.text === 'string' ? meta.text : '';

	return {
		chunkId,
		chunkText,
		title: typeof meta.title === 'string' ? meta.title : undefined,
		source: typeof meta.source === 'string' ? meta.source : undefined,
		metadata:
			Object.keys(userMetadata).length > 0 ? userMetadata : undefined,
		score: scoreForMetric(match.score, distanceMetric)
	};
};

export const createPineconeStore = (
	options: PineconeRAGOptions
): RAGVectorStore => {
	const vector = resolveVectorConfig(options ?? {});
	const namespace =
		typeof options.namespace === 'string' && options.namespace.length > 0
			? options.namespace
			: undefined;
	const getClient = resolveIndexClientFactory(options ?? {});

	const embed: RAGVectorStore['embed'] = async (input) => {
		if (typeof options.embedding === 'function') {
			const result = await options.embedding(input);

			return normalizeVector(result);
		}

		return normalizeVector([
			...createRAGVector(input.text, vector.dimensions)
		]);
	};

	const upsert = async (input: RAGUpsertInput): Promise<void> => {
		if (!input?.chunks || input.chunks.length === 0) return;
		const client = await getClient();
		const records = await Promise.all(
			input.chunks.map(async (chunk) => {
				const values =
					Array.isArray(chunk.embedding) && chunk.embedding.length > 0
						? normalizeVector(chunk.embedding)
						: await embed({ text: chunk.text });

				return buildPineconeRecord(chunk, values);
			})
		);
		for (const batch of chunkArray(records, PINECONE_UPSERT_BATCH_SIZE)) {
			await client.upsert(batch);
		}
	};

	const query = async (input: RAGQueryInput): Promise<RAGQueryResult[]> => {
		const client = await getClient();
		const filter = translateFilter(input.filter);
		const queryInput: Record<string, unknown> = {
			vector: normalizeVector(input.queryVector),
			topK: input.topK,
			includeMetadata: true,
			includeValues: false
		};
		if (filter) queryInput.filter = filter;
		const result = await client.query(queryInput);
		const matches = Array.isArray(result?.matches) ? result.matches : [];

		return matches.map((match) =>
			extractQueryResult(match, vector.distanceMetric)
		);
	};

	const count = async (
		input: { chunkIds?: string[]; filter?: Record<string, unknown> } = {}
	): Promise<number> => {
		const client = await getClient();
		if (Array.isArray(input.chunkIds) && input.chunkIds.length > 0) {
			let total = 0;
			for (const batch of chunkArray(
				input.chunkIds,
				PINECONE_FETCH_BATCH_SIZE
			)) {
				const response = await client.fetch(batch);
				total += Object.keys(response?.records ?? {}).length;
			}

			return total;
		}
		if (
			isObjectRecord(input.filter) &&
			Object.keys(input.filter).length > 0
		) {
			const filter = translateFilter(input.filter);
			const probeVector = new Array(vector.dimensions).fill(0);
			const queryInput: Record<string, unknown> = {
				vector: probeVector,
				topK: PINECONE_FILTERED_COUNT_TOPK,
				includeMetadata: false,
				includeValues: false
			};
			if (filter) queryInput.filter = filter;
			const result = await client.query(queryInput);

			return Array.isArray(result?.matches) ? result.matches.length : 0;
		}
		const stats = await client.describeIndexStats();
		if (namespace) {
			return stats?.namespaces?.[namespace]?.recordCount ?? 0;
		}

		return stats?.totalRecordCount ?? 0;
	};

	const remove = async (
		input: { chunkIds?: string[]; filter?: Record<string, unknown> } = {}
	): Promise<number> => {
		const client = await getClient();
		if (Array.isArray(input.chunkIds) && input.chunkIds.length > 0) {
			for (const batch of chunkArray(
				input.chunkIds,
				PINECONE_FETCH_BATCH_SIZE
			)) {
				try {
					await client.deleteMany(batch);
				} catch (error) {
					if (!isPineconeNotFound(error)) throw error;
				}
			}

			return input.chunkIds.length;
		}
		if (
			isObjectRecord(input.filter) &&
			Object.keys(input.filter).length > 0
		) {
			const filter = translateFilter(input.filter);
			if (!filter) return 0;
			const counted = await count({ filter: input.filter });
			try {
				await client.deleteMany({ filter });
			} catch (error) {
				if (!isPineconeNotFound(error)) throw error;
			}

			return counted;
		}

		return 0;
	};

	const clear = async (): Promise<void> => {
		const client = await getClient();
		try {
			await client.deleteAll();
		} catch (error) {
			if (!isPineconeNotFound(error)) throw error;
		}
	};

	return {
		embed,
		query,
		upsert,
		count,
		delete: remove,
		clear,
		getCapabilities: () => ({
			backend: 'custom',
			persistence: 'external',
			nativeVectorSearch: true,
			serverSideFiltering: true,
			streamingIngestStatus: false
		})
	} as RAGVectorStore;
};

const DEFAULT_INDEX_READY_TIMEOUT_MS = 120000;
const DEFAULT_INDEX_READY_POLL_MS = 1500;
const DEFAULT_SERVERLESS_SPEC: PineconeServerlessSpec = {
	serverless: { cloud: 'aws', region: 'us-east-1' }
};

type PineconeErrorLike = {
	status?: number;
	statusCode?: number;
	response?: { status?: number };
	name?: string;
	message?: string;
};

const isPineconeNotFound = (error: unknown): boolean => {
	if (!error) return false;
	const err = error as PineconeErrorLike;
	const status = err.status ?? err.statusCode ?? err.response?.status ?? undefined;
	if (status === 404) return true;
	const name = String(err.name ?? '');
	if (name === 'PineconeNotFoundError') return true;
	const message = String(err.message ?? '').toLowerCase();

	return (
		message.includes('not found') ||
		message.includes('does not exist') ||
		message.includes('404')
	);
};

const resolveProvisioningClient = async (
	options: DescribePineconeIndexOptions | EnsurePineconeIndexOptions
): Promise<InstanceType<PineconeSDKModule['Pinecone']>> => {
	const apiKey = options.apiKey ?? process.env.PINECONE_API_KEY;
	if (!apiKey) {
		throw new Error(
			`${PKG}: missing Pinecone apiKey (pass options.apiKey or set PINECONE_API_KEY)`
		);
	}
	if (
		typeof options.indexName !== 'string' ||
		options.indexName.length === 0
	) {
		throw new Error(`${PKG}: indexName is required`);
	}
	const { Pinecone } = await loadPineconeSDK();

	return new Pinecone({ apiKey });
};

const waitForIndexReady = async (
	pc: InstanceType<PineconeSDKModule['Pinecone']>,
	indexName: string,
	timeoutMs: number,
	pollIntervalMs: number
): Promise<PineconeIndexDescription> => {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		const description = await pc.describeIndex(indexName);
		const status = description?.status;
		const ready =
			status?.ready === true ||
			status?.state === 'Ready' ||
			status?.state === 'ScalingUp';
		if (ready) return description;
		if (Date.now() >= deadline) {
			throw new Error(
				`${PKG}: index "${indexName}" did not become ready within ${timeoutMs}ms (last state: ${
					status?.state ?? 'unknown'
				})`
			);
		}
		await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
	}
};

export const describePineconeIndex = async (
	options: DescribePineconeIndexOptions
): Promise<PineconeIndexDescription | undefined> => {
	const pc = await resolveProvisioningClient(options);
	try {
		return await pc.describeIndex(options.indexName);
	} catch (error) {
		if (isPineconeNotFound(error)) return undefined;
		throw error;
	}
};

export const ensurePineconeIndex = async (
	options: EnsurePineconeIndexOptions
): Promise<EnsurePineconeIndexResult> => {
	if (!Number.isInteger(options.dimensions) || options.dimensions <= 0) {
		throw new Error(
			`${PKG}: dimensions must be a positive integer (received ${String(
				options.dimensions
			)})`
		);
	}
	const metric = options.metric ?? 'cosine';
	if (!PINECONE_DISTANCE_METRICS.includes(metric)) {
		throw new Error(
			`${PKG}: unsupported metric "${metric}". Allowed: ${PINECONE_DISTANCE_METRICS.join(
				', '
			)}`
		);
	}
	const pc = await resolveProvisioningClient(options);
	const timeoutMs = options.waitTimeoutMs ?? DEFAULT_INDEX_READY_TIMEOUT_MS;
	const pollIntervalMs = options.pollIntervalMs ?? DEFAULT_INDEX_READY_POLL_MS;
	const shouldWait = options.waitUntilReady !== false;

	let existing: PineconeIndexDescription | undefined;
	try {
		existing = await pc.describeIndex(options.indexName);
	} catch (error) {
		if (!isPineconeNotFound(error)) throw error;
	}

	if (existing) {
		if (existing.dimension !== options.dimensions) {
			throw new Error(
				`${PKG}: index "${options.indexName}" already exists with dimension=${existing.dimension}, but ${options.dimensions} was requested`
			);
		}
		if (existing.metric && existing.metric !== metric) {
			throw new Error(
				`${PKG}: index "${options.indexName}" already exists with metric="${existing.metric}", but "${metric}" was requested`
			);
		}
		if (shouldWait && existing.status?.ready !== true) {
			const description = await waitForIndexReady(
				pc,
				options.indexName,
				timeoutMs,
				pollIntervalMs
			);

			return { created: false, description };
		}

		return { created: false, description: existing };
	}

	const spec = options.spec ?? DEFAULT_SERVERLESS_SPEC;
	await pc.createIndex({
		name: options.indexName,
		dimension: options.dimensions,
		metric,
		spec,
		...(options.deletionProtection
			? { deletionProtection: options.deletionProtection }
			: {})
	});

	if (!shouldWait) {
		let description: PineconeIndexDescription | undefined;
		try {
			description = await pc.describeIndex(options.indexName);
		} catch (error) {
			if (!isPineconeNotFound(error)) throw error;
		}

		return { created: true, description };
	}

	const description = await waitForIndexReady(
		pc,
		options.indexName,
		timeoutMs,
		pollIntervalMs
	);

	return { created: true, description };
};

export const createPineconeRAGCollection = (
	options: PineconeRAGOptions
): RAGCollection =>
	createRAGCollection({
		store: createPineconeStore(options)
	});

export const createPineconeRAG = (options: PineconeRAGOptions): PineconeRAG => {
	const store = createPineconeStore(options);
	const collection = createRAGCollection({ store });

	return {
		store,
		collection,
		getCapabilities: () => store.getCapabilities?.()
	};
};
