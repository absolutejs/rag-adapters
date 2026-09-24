import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { createPostgresRAGStore } from "../src/createPostgresRAGStore";

const url = process.env.RAG_LEXICAL_TEST_URL;
const suite = url ? describe : describe.skip;
suite("native PostgreSQL lexical index", () => {
  const db = new Bun.SQL(url!);
  const schema = `lexical_${Date.now()}`;
  const store = createPostgresRAGStore({ sql: db, schemaName: schema, dimensions: 3, indexType: "none" });
  beforeAll(async () => {
    await store.upsert({ chunks: [
      { chunkId: "a", text: "partnership eligibility includes a trial", title: "Partner policy", metadata: { tenant: "a", topic: "alliances" }, embedding: [1, 0, 0] },
      { chunkId: "b", text: "partner partnership partnership eligibility", metadata: { tenant: "b" }, embedding: [1, 0, 0] },
      { chunkId: "c", text: "pricing overview", metadata: { tenant: "a", topic: "alliances" }, embedding: [1, 0, 0] },
      { chunkId: "d", text: "Release café O'Reilly documentation", metadata: { tenant: "a" }, embedding: [1, 0, 0] },
    ] });
  });
  afterAll(async () => { await db.unsafe(`drop schema if exists "${schema}" cascade`); await store.close?.(); });
  test("creates a GIN index and limits results within the tenant", async () => {
    const result = await store.queryLexical!({ query: "partner eligibility", topK: 1, filter: { tenant: "a" } });
    expect(result.map(x => x.chunkId)).toEqual(["a"]);
    const indexes = await db`select indexdef from pg_indexes where schemaname = ${schema}`;
    expect(indexes.some((x: { indexdef: string }) => x.indexdef.includes("USING gin"))).toBe(true);
  });
  test("retains any-term matching and indexes metadata values", async () => {
    expect((await store.queryLexical!({ query: "alliances unknownword", topK: 10, filter: { tenant: "a" } })).map(x => x.chunkId).sort()).toEqual(["a", "c"]);
  });
  test("handles punctuation, quotes, Unicode and empty queries without query syntax injection", async () => {
    expect((await store.queryLexical!({ query: "café O'Reilly", topK: 5, filter: { tenant: "a" } })).map(x => x.chunkId)).toContain("d");
    expect(await store.queryLexical!({ query: "!!!", topK: 5 })).toEqual([]);
    expect(await store.queryLexical!({ query: "", topK: 5 })).toEqual([]);
    expect(await store.queryLexical!({ query: "partner", topK: 0 })).toEqual([]);
  });
  test("rejects unsupported filters instead of silently dropping tenant scope", async () => {
    await expect(store.queryLexical!({ query: "partner", topK: 5, filter: { tenant: { $unknown: "a" } } })).rejects.toThrow("fully supported");
  });
  test("preserves fixture recall across native and portable modes", async () => {
    const portable = createPostgresRAGStore({ sql: db, schemaName: schema, dimensions: 3, indexType: "none", lexicalMode: "portable" });
    for (const query of ["partnership", "alliances", "pricing"]) {
      const input = { query, topK: 10, filter: { $and: [{ tenant: "a" }, { tenant: { $in: ["a", "b"] } }] } };
      const nativeIds = (await store.queryLexical!(input)).map(x => x.chunkId).sort();
      const portableIds = (await portable.queryLexical!(input)).map(x => x.chunkId).sort();
      expect(nativeIds).toEqual(portableIds);
    }
  });
  test("indexes an existing portable table on first native use", async () => {
    const options = { sql: db, schemaName: schema, tableName: "legacy", dimensions: 3, indexType: "none" as const };
    const portable = createPostgresRAGStore({ ...options, lexicalMode: "portable" });
    await portable.upsert({ chunks: [{ chunkId: "legacy", text: "migration evidence", embedding: [1, 0, 0] }] });
    const native = createPostgresRAGStore(options);
    expect((await native.queryLexical!({ query: "migration", topK: 1 }))[0]?.chunkId).toBe("legacy");
  });
  test("updates and deletions immediately change lexical results", async () => {
    await store.upsert({ chunks: [{ chunkId: "updated", text: "oldterm", metadata: { tenant: "a" }, embedding: [1, 0, 0] }] });
    expect((await store.queryLexical!({ query: "oldterm", topK: 5 })).length).toBe(1);
    await store.upsert({ chunks: [{ chunkId: "updated", text: "newterm", metadata: { tenant: "a" }, embedding: [1, 0, 0] }] });
    expect(await store.queryLexical!({ query: "oldterm", topK: 5 })).toEqual([]);
    expect((await store.queryLexical!({ query: "newterm", topK: 5 })).length).toBe(1);
    await store.delete?.({ chunkIds: ["updated"] });
    expect(await store.queryLexical!({ query: "newterm", topK: 5 })).toEqual([]);
  });
});
