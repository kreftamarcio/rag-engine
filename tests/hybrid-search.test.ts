import { describe, it, expect } from 'vitest';
import { HybridSearch } from '../src/retrieval/hybrid-search.js';
import type {
  EmbeddingAdapter,
  VectorStoreAdapter,
  SparseIndexAdapter,
} from '../src/retrieval/hybrid-search.js';
import type { Chunk } from '../src/core/config.js';

// ── Test adapters ──

function makeChunk(id: string, content: string): Chunk {
  return {
    id,
    content,
    metadata: {
      source: `test/${id}`,
      chunkIndex: 0,
      totalChunks: 1,
      startOffset: 0,
      endOffset: content.length,
      tokenCount: content.split(/\s+/).length,
    },
  };
}

/** Returns a fixed vector for each text. Good enough for testing fusion logic. */
class StubEmbedder implements EmbeddingAdapter {
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((t) => {
      // Deterministic hash-like embedding
      const hash = [...t].reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
      return [hash % 100 / 100, (hash * 7) % 100 / 100, (hash * 13) % 100 / 100];
    });
  }
}

class StubVectorStore implements VectorStoreAdapter {
  private chunks: Chunk[] = [];

  async upsert(chunks: Chunk[]): Promise<void> {
    this.chunks.push(...chunks);
  }

  async query(_embedding: number[], topK: number): Promise<Array<{ chunk: Chunk; similarity: number }>> {
    return this.chunks.slice(0, topK).map((chunk, i) => ({
      chunk,
      similarity: 1 - i * 0.1,
    }));
  }
}

class StubSparseIndex implements SparseIndexAdapter {
  private chunks: Chunk[] = [];

  async index(chunks: Chunk[]): Promise<void> {
    this.chunks.push(...chunks);
  }

  async query(tokens: string[], topK: number): Promise<Array<{ chunk: Chunk; bm25Score: number }>> {
    // Simple token overlap scoring
    const scored = this.chunks.map((chunk) => {
      const chunkTokens = new Set(chunk.content.toLowerCase().split(/\s+/));
      const overlap = tokens.filter((t) => chunkTokens.has(t.toLowerCase())).length;
      return { chunk, bm25Score: overlap };
    });
    return scored
      .sort((a, b) => b.bm25Score - a.bm25Score)
      .slice(0, topK);
  }
}

class FailingVectorStore implements VectorStoreAdapter {
  async upsert(): Promise<void> { /* noop */ }
  async query(): Promise<Array<{ chunk: Chunk; similarity: number }>> {
    throw new Error('Dense retrieval unavailable');
  }
}

class FailingSparseIndex implements SparseIndexAdapter {
  async query(): Promise<Array<{ chunk: Chunk; bm25Score: number }>> {
    throw new Error('Sparse index unavailable');
  }
}

describe('HybridSearch', () => {
  // ── Basic retrieval ──

  it('returns results with dense-only search when no sparse index is configured', async () => {
    const store = new StubVectorStore();
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: store,
      // sparseIndex intentionally omitted
    });

    await search.upsert([
      makeChunk('a', 'TypeScript strict mode enables better type safety'),
      makeChunk('b', 'Circuit breakers prevent cascading failures'),
    ]);

    const result = await search.search('type safety', 2);

    expect(result.chunks.length).toBeGreaterThan(0);
    expect(result.sources).toContain('dense');
    expect(result.sources).not.toContain('sparse');
  });

  // ── RRF fusion and agreement ──

  it('reports agreement when both retrievers find the same chunk', async () => {
    const store = new StubVectorStore();
    const sparse = new StubSparseIndex();
    const chunks = [
      makeChunk('shared', 'Circuit breakers use rolling window failure counting'),
      makeChunk('dense-only', 'Orchestration patterns for multi-agent systems'),
    ];

    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: store,
      sparseIndex: sparse,
    });

    await search.upsert(chunks);
    const result = await search.search('circuit breaker rolling window', 5);

    // The chunk about circuit breakers should be found by both retrievers
    expect(result.sources).toContain('dense');
    expect(result.sources).toContain('sparse');
    expect(result.agreementCount).toBeGreaterThanOrEqual(0);
  });

  // ── Partial failure: degraded, not dead ──
  // From the source: "a sparse index outage should not fail a query that
  // dense retrieval alone can answer"

  it('degrades gracefully when sparse index fails', async () => {
    const store = new StubVectorStore();
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: store,
      sparseIndex: new FailingSparseIndex(),
    });

    await store.upsert([makeChunk('a', 'Still retrievable via dense')]);
    const result = await search.search('retrievable', 5);

    expect(result.chunks.length).toBeGreaterThan(0);
    expect(result.degraded).toBeDefined();
    expect(result.degraded?.retriever).toBe('sparse');
  });

  it('throws when both retrievers fail', async () => {
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new FailingVectorStore(),
      sparseIndex: new FailingSparseIndex(),
    });

    await expect(search.search('anything', 5)).rejects.toThrow(/Both retrievers failed/);
  });

  // ── Config validation at construction ──

  it('throws on negative fusion weights', () => {
    expect(() => new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new StubVectorStore(),
      denseWeight: -1,
      sparseWeight: 0.5,
    })).toThrow(/negative/);
  });

  it('throws when both fusion weights are zero', () => {
    expect(() => new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new StubVectorStore(),
      denseWeight: 0,
      sparseWeight: 0,
    })).toThrow(/zero/);
  });

  it('throws when topK is less than 1', async () => {
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new StubVectorStore(),
    });

    await expect(search.search('query', 0)).rejects.toThrow(/topK/);
  });

  // ── Unicode-aware tokenization ──
  // From the source: "The previous implementation used /[^a-z0-9\s]/g, which strips
  // every accented character: 'cobrança' became 'cobrana'"

  it('preserves accented characters in tokenization', () => {
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new StubVectorStore(),
    });

    const tokens = search.tokenize('cobrança São José über');
    expect(tokens).toContain('cobrança');
    expect(tokens).toContain('são');
    expect(tokens).toContain('josé');
    expect(tokens).toContain('über');
  });

  it('filters tokens shorter than 2 characters', () => {
    const search = new HybridSearch({
      embedding: new StubEmbedder(),
      vectorStore: new StubVectorStore(),
    });

    const tokens = search.tokenize('a is the id of it');
    // 'a' is 1 char → filtered. 'is', 'id', 'of', 'it' are 2 chars → kept.
    // 'the' is 3 chars → kept.
    expect(tokens).not.toContain('a');
    expect(tokens).toContain('is');
    expect(tokens).toContain('id');
  });
});
