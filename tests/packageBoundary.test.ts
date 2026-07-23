import { describe, expect, test } from "bun:test";
import { readFile } from "node:fs/promises";
import { join } from "node:path";

const adapters = ["pinecone", "postgres", "sqlite"] as const;

describe("RAG adapter package boundaries", () => {
  for (const adapter of adapters) {
    test(`${adapter} uses the host's current RAG domain runtime`, async () => {
      const packageJson = JSON.parse(
        await readFile(
          join(import.meta.dir, "..", adapter, "package.json"),
          "utf8",
        ),
      ) as {
        dependencies?: Record<string, string>;
        devDependencies?: Record<string, string>;
        peerDependencies?: Record<string, string>;
      };

      expect(packageJson.dependencies?.["@absolutejs/rag"]).toBeUndefined();
      expect(packageJson.peerDependencies?.["@absolutejs/rag"]).toBe(
        ">=0.0.20 <0.1.0",
      );
      expect(packageJson.devDependencies?.["@absolutejs/rag"]).toBe("0.0.31");
    });
  }
});
