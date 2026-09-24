# Changelog

## 0.1.0 — 2026-09-24

- Make indexed PostgreSQL lexical retrieval the default, using a weighted GIN expression index and bounded, parameterized, tenant-filtered queries.
- Add `lexicalMode: "portable"` to retain the previous in-memory scorer. Native tokenization, scores and ordering differ; native mode rejects unsupported filters instead of dropping them.
- Add real PostgreSQL CI tests and reproducible 1K/10K/100K-row measurements.
- Correct the plugin example to use the current `storeOptions` API and Bun.SQL driver.
