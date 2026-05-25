import type {
	RAGBackendCapabilities,
	RAGCollection,
	RAGVectorStore,
	RAGVectorStoreStatus,
	SQLiteRAGStoreOptions,
	SQLiteVecResolution
} from '@absolutejs/rag';
import {
	createRAGCollection,
	createSQLiteRAGStore as createCoreSQLiteRAGStore,
	ragPlugin,
	resolveAbsoluteSQLiteVec,
	resolveAbsoluteSQLiteVecExtensionPath
} from '@absolutejs/rag';

export const ABSOLUTE_SQLITE_RAG_PACKAGE_NAME = '@absolutejs/rag-sqlite';

export type SQLiteRAGCollectionOptions = {
	store?: RAGVectorStore;
	storeOptions?: SQLiteRAGStoreOptions;
};

export type SQLiteRAGOptions = {
	store?: RAGVectorStore;
	collection?: RAGCollection;
	storeOptions?: SQLiteRAGStoreOptions;
};

export type SQLiteRAGSupportSummary = {
	backendPackageName: typeof ABSOLUTE_SQLITE_RAG_PACKAGE_NAME;
	recommendedInstallCommand: string;
	resolution: SQLiteVecResolution;
	status?: RAGVectorStoreStatus;
	capabilities?: RAGBackendCapabilities;
	nativeRequested: boolean;
	nativeActive: boolean;
	actionableMessage: string;
};

export type SQLiteRAG = {
	store: RAGVectorStore;
	collection: RAGCollection;
	getStatus: () => RAGVectorStoreStatus | undefined;
	getCapabilities: () => RAGBackendCapabilities | undefined;
	getNativeSupport: () => SQLiteRAGSupportSummary;
};

const nativeMessageFromResolution = (
	resolution: SQLiteVecResolution | undefined
): string => {
	switch (resolution?.status) {
		case 'resolved':
			return 'Native sqlite vec support is installed and can be activated with native.mode="vec0".';
		case 'package_not_installed':
			return `Install ${ABSOLUTE_SQLITE_RAG_PACKAGE_NAME} and restart to let AbsoluteJS resolve the platform sqlite vec package automatically.`;
		case 'binary_missing':
			return 'The platform sqlite vec package was resolved but its native library file is missing.';
		case 'unsupported_platform':
			return `No sqlite vec platform package is defined for ${resolution.platformKey}. JSON fallback remains available.`;
		case 'package_invalid':
			return 'The installed sqlite vec package is invalid. Reinstall the backend package and restart.';
		case 'not_configured':
		default:
			return 'Native sqlite vec support is not configured. JSON fallback remains available.';
	}
};

export const createSQLiteRAGStore: typeof createCoreSQLiteRAGStore = (
	options = {}
) => createCoreSQLiteRAGStore(options);

export const createSQLiteRAGCollection = (
	options: SQLiteRAGCollectionOptions = {}
): RAGCollection => {
	const store =
		options.store ?? createSQLiteRAGStore(options.storeOptions ?? {});

	return createRAGCollection({ store });
};

export const createSQLiteRAG = (options: SQLiteRAGOptions = {}): SQLiteRAG => {
	const store =
		options.store ?? createSQLiteRAGStore(options.storeOptions ?? {});
	const collection = options.collection ?? createRAGCollection({ store });

	return {
		store,
		collection,
		getStatus: () => collection.getStatus?.() ?? store.getStatus?.(),
		getCapabilities: () =>
			collection.getCapabilities?.() ?? store.getCapabilities?.(),
		getNativeSupport: () => summarizeSQLiteRAGSupport(collection)
	};
};

export const createSQLiteRAGBackend: typeof createSQLiteRAG = createSQLiteRAG;

export const getSQLiteRAGNativeSupport: typeof resolveAbsoluteSQLiteVec = () =>
	resolveAbsoluteSQLiteVec();

export const summarizeSQLiteRAGSupport = (
	target?:
		| Pick<RAGCollection, 'getStatus' | 'getCapabilities'>
		| Pick<RAGVectorStore, 'getStatus' | 'getCapabilities'>
): SQLiteRAGSupportSummary => {
	const status = target?.getStatus?.();
	const capabilities = target?.getCapabilities?.();
	const nativeStatus = status?.native as
		| { resolution?: SQLiteVecResolution; requested?: boolean; active?: boolean }
		| undefined;
	const resolution = nativeStatus?.resolution ?? resolveAbsoluteSQLiteVec();
	const nativeRequested = nativeStatus?.requested ?? false;
	const nativeActive = nativeStatus?.active ?? false;

	let actionableMessage = nativeMessageFromResolution(resolution);

	if (nativeActive) {
		actionableMessage = 'Native sqlite vec0 is active for this store.';
	} else if (resolution?.status === 'resolved' && nativeRequested) {
		actionableMessage =
			'Native sqlite vec is installed, but this store is still running in fallback mode. Check the store diagnostics for load or query errors.';
	}

	return {
		backendPackageName: ABSOLUTE_SQLITE_RAG_PACKAGE_NAME,
		recommendedInstallCommand: `bun add ${ABSOLUTE_SQLITE_RAG_PACKAGE_NAME}`,
		resolution,
		status,
		capabilities,
		nativeRequested,
		nativeActive,
		actionableMessage
	};
};

export {
	createRAGCollection,
	ragPlugin,
	resolveAbsoluteSQLiteVec,
	resolveAbsoluteSQLiteVecExtensionPath
};

export type {
	RAGBackendCapabilities,
	RAGCollection,
	RAGVectorStore,
	RAGVectorStoreStatus,
	SQLiteRAGStoreOptions,
	SQLiteVecResolution
};
