import { createPostgresRAGStore } from '../src/createPostgresRAGStore';

const url = process.env.RAG_LEXICAL_TEST_URL;
if (!url) throw new Error('RAG_LEXICAL_TEST_URL is required (use an isolated test database)');
const db = new Bun.SQL(url);
const schema = `lexical_benchmark_${Date.now()}`;
let lastQuery: { sql: string; params: unknown[]; rows: number } | undefined;
const observed = { unsafe: async (sql: string, params: unknown[] = []) => {
  const result = await db.unsafe(sql, params);
  if (sql.startsWith('with search_query') || sql.startsWith('select chunk_id, text, title'))
    lastQuery = { sql, params, rows: result.length };
  return result;
} } as unknown as InstanceType<typeof Bun.SQL>;
const stores = {
  native: createPostgresRAGStore({ sql: observed, schemaName: schema, dimensions: 3, indexType: 'none' }),
  portable: createPostgresRAGStore({ sql: observed, schemaName: schema, dimensions: 3, indexType: 'none', lexicalMode: 'portable' }),
};
const reports: unknown[] = [];
const query = { query: 'quasar', topK: 10, filter: { tenant: 'a' } };
try {
  await stores.native.upsert({ chunks: [] });
  await stores.portable.upsert({ chunks: [] });
  for (const size of [1000, 10000, 100000]) {
    await db.unsafe(`truncate table "${schema}".rag_chunks`);
    await db.unsafe(`insert into "${schema}".rag_chunks (chunk_id, text, title, metadata, embedding)
      select i::text, case when i % 997 = 0 then 'quasar partnership evidence' else 'ordinary research document background' end,
        'Document ' || i, jsonb_build_object('tenant', case when i % 2 = 0 then 'a' else 'b' end), '[1,0,0]'::vector
      from generate_series(1, $1::int) i`, [size]);
    await db.unsafe(`vacuum analyze "${schema}".rag_chunks`);
    for (const [mode, store] of Object.entries(stores)) {
      const warm = await store.queryLexical!(query);
      if (warm.some(hit => hit.metadata?.tenant !== 'a' || !hit.chunkText.includes('quasar')))
        throw new Error('Scope or relevance fixture failed');
      const expected = Math.min(10, Math.floor(size / 1994));
      if (warm.length !== expected) throw new Error(`Coverage mismatch: ${mode} ${size}`);
      const timings: number[] = [];
      for (let trial = 0; trial < 10; trial++) {
        const started = performance.now();
        await store.queryLexical!(query);
        timings.push(performance.now() - started);
      }
      const sorted = [...timings].sort((a,b) => a-b);
      const captured = lastQuery!;
      const plan = await db.unsafe(`explain (analyze, buffers, format json) ${captured.sql}`, captured.params);
      reports.push({ size, mode, trials: timings, p50Ms: sorted[4], p95Ms: sorted[9], rowsTransferred: captured.rows,
        returned: warm.length, expected, plan: plan[0]['QUERY PLAN'] });
      console.log(JSON.stringify({ size, mode, p50Ms: sorted[4], p95Ms: sorted[9], rowsTransferred: captured.rows }));
    }
  }
  await Bun.write(process.argv[2] ?? 'postgres/benchmarks/lexical-results.json', JSON.stringify({
    generatedAt: new Date().toISOString(), protocol: 'Local PostgreSQL 15, deterministic synthetic corpus, one selective query, tenant filter, one warmup and ten trials per mode/size; excludes ingestion/index build and provider costs. This measures retrieval implementation, not web-search relevance.', reports,
  }, null, 2) + '\n');
} finally {
  await db.unsafe(`drop schema if exists "${schema}" cascade`);
  await db.close();
}
