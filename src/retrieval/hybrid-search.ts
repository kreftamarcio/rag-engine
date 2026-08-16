import type { Chunk } from '../core/engine.js';

export interface HybridSearchConfig {
  vectorStore: {
    provider: string;
    url: string;
    collection: string;
    apiKey?: string;
  };
  embedding: {
    provider: string;
    model: string;
    dimensions: number;
  };
  sparseWeight: number;
  denseWeight: number;
  fusionK: number;
}

interface ScoredChunk extends Chunk {
  score: number;
  source: 'dense' | 'sparse' | 'fused';
}

/**
 * Hybrid Search combines dense (vector similarity) and sparse (BM25) retrieval
 * using Reciprocal Rank Fusion (RRF) to produce a unified ranking.
 *
 * The key insight: dense retrieval captures semantic similarity,
 * while sparse retrieval captures exact keyword matches.
 * Combining both yields higher recall than either alone.
 */
export class HybridSearch {
  private readonly config: HybridSearchConfig;

  constructor(config: HybridSearchConfig) {
    this.config = config;
  }

  /**
   * Execute hybrid search: run dense and sparse in parallel, then fuse.
   */
  async search(query: string, topK: number): Promise<Chunk[]> {
    const [denseResults, sparseResults] = await Promise.all([
      this.denseSearch(query, topK * 2),
      this.sparseSearch(query, topK * 2),
    ]);

    const fused = this.reciprocalRankFusion(
      denseResults,
      sparseResults,
      topK,
    );

    return fused;
  }

  /**
   * Generate embeddings for an array of texts.
   */
  async embed(texts: string[]): Promise<number[][]> {
    // Batch embedding with provider SDK
    // OpenAI: POST /v1/embeddings with input array
    // Handles batching internally (max 2048 inputs per request)
    const batchSize = 512;
    const allEmbeddings: number[][] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const embeddings = await this.callEmbeddingAPI(batch);
      allEmbeddings.push(...embeddings);
    }

    return allEmbeddings;
  }

  /**
   * Upsert chunks with embeddings into the vector store.
   */
  async upsert(chunks: Chunk[]): Promise<void> {
    const batchSize = 100;

    for (let i = 0; i < chunks.length; i += batchSize) {
      const batch = chunks.slice(i, i + batchSize);
      await this.upsertBatch(batch);
    }
  }

  /**
   * Dense retrieval: vector similarity search using cosine distance.
   */
  private async denseSearch(query: string, topK: number): Promise<ScoredChunk[]> {
    const [queryEmbedding] = await this.embed([query]);

    // Query vector store with embedding
    // Returns chunks ordered by cosine similarity
    const results = await this.queryVectorStore(queryEmbedding!, topK);

    return results.map((r, idx) => ({
      ...r.chunk,
      score: r.similarity,
      source: 'dense' as const,
    }));
  }

  /**
   * Sparse retrieval: BM25 keyword matching.
   *
   * BM25 formula:
   * score(D,Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))
   *
   * Where:
   *   f(qi,D) = term frequency of qi in document D
   *   |D| = document length
   *   avgdl = average document length
   *   k1 = 1.2 (term frequency saturation)
   *   b = 0.75 (length normalization)
   */
  private async sparseSearch(query: string, topK: number): Promise<ScoredChunk[]> {
    // Tokenize query
    const queryTokens = this.tokenize(query);

    // Execute BM25 against sparse index
    const results = await this.queryBM25Index(queryTokens, topK);

    return results.map(r => ({
      ...r.chunk,
      score: r.bm25Score,
      source: 'sparse' as const,
    }));
  }

  /**
   * Reciprocal Rank Fusion (RRF)
   *
   * Combines rankings from multiple retrieval systems.
   * For each document d and retrieval system i:
   *   RRF_score(d) = Σ 1 / (k + rank_i(d))
   *
   * Properties:
   * - Rank-based (not score-based), so no normalization needed
   * - Documents ranked highly by both systems get boosted
   * - Parameter k (default 60) controls the impact of low-ranked documents
   */
  private reciprocalRankFusion(
    denseResults: ScoredChunk[],
    sparseResults: ScoredChunk[],
    topK: number,
  ): Chunk[] {
    const k = this.config.fusionK;
    const scores = new Map<string, { score: number; chunk: Chunk }>();

    // Score dense results
    for (let rank = 0; rank < denseResults.length; rank++) {
      const chunk = denseResults[rank]!;
      const rrf = this.config.denseWeight * (1 / (k + rank + 1));
      const existing = scores.get(chunk.id);

      if (existing) {
        existing.score += rrf;
      } else {
        scores.set(chunk.id, { score: rrf, chunk });
      }
    }

    // Score sparse results
    for (let rank = 0; rank < sparseResults.length; rank++) {
      const chunk = sparseResults[rank]!;
      const rrf = this.config.sparseWeight * (1 / (k + rank + 1));
      const existing = scores.get(chunk.id);

      if (existing) {
        existing.score += rrf;
      } else {
        scores.set(chunk.id, { score: rrf, chunk });
      }
    }

    // Sort by fused score and take topK
    return Array.from(scores.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(entry => entry.chunk);
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '')
      .split(/\s+/)
      .filter(token => token.length > 2);
  }

  // Abstract methods to be implemented by store-specific adapters
  private async callEmbeddingAPI(_texts: string[]): Promise<number[][]> {
    throw new Error('Embedding provider not configured');
  }

  private async queryVectorStore(
    _embedding: number[],
    _topK: number,
  ): Promise<Array<{ chunk: Chunk; similarity: number }>> {
    throw new Error('Vector store not configured');
  }

  private async queryBM25Index(
    _tokens: string[],
    _topK: number,
  ): Promise<Array<{ chunk: Chunk; bm25Score: number }>> {
    throw new Error('Sparse index not configured');
  }

  private async upsertBatch(_chunks: Chunk[]): Promise<void> {
    throw new Error('Vector store not configured');
  }
}
