import type { Chunk } from '../core/engine.js';

/**
 * Hybrid search: dense (vector) and sparse (BM25) retrieval fused by Reciprocal Rank
 * Fusion.
 *
 * The two retrievers fail in opposite directions. Dense retrieval captures semantic
 * similarity but misses exact terms: ask for error code E4021 and it returns
 * conceptually similar errors. Sparse retrieval matches exact terms but misses
 * paraphrase: ask about "login broken" and it will not find "authentication failure".
 * Fusing them yields higher recall than either alone.
 *
 * Store access is injected rather than subclassed. The previous design declared the
 * adapter methods private and threw from them, which made the documented extension
 * point impossible to use: a private method cannot be overridden from outside, and
 * subclassing gains nothing when the base implementation only throws.
 */

export interface EmbeddingAdapter {
  /** Embed a batch. Must return one vector per input, in the same order. */
  embed(texts: string[]): Promise<number[][]>;
}

export interface VectorStoreAdapter {
  query(
    embedding: number[],
    topK: number,
  ): Promise<Array<{ chunk: Chunk; similarity: number }>>;
  upsert(chunks: Chunk[]): Promise<void>;
}

export interface SparseIndexAdapter {
  /**
   * BM25 retrieval over tokenized query terms.
   *
   *   score(D,Q) = SUM IDF(qi) * ( f(qi,D) * (k1 + 1) )
   *                            / ( f(qi,D) + k1 * (1 - b + b * |D|/avgdl) )
   *
   * with k1 = 1.2 (term frequency saturation) and b = 0.75 (length normalization).
   */
  query(tokens: string[], topK: number): Promise<Array<{ chunk: Chunk; bm25Score: number }>>;
  index?(chunks: Chunk[]): Promise<void>;
}

export interface HybridSearchConfig {
  embedding: EmbeddingAdapter;
  vectorStore: VectorStoreAdapter;
  sparseIndex?: SparseIndexAdapter;
  /** Relative weight of dense results in fusion. */
  denseWeight?: number;
  /** Relative weight of sparse results in fusion. */
  sparseWeight?: number;
  /** RRF damping constant. Higher values flatten the contribution of low ranks. */
  fusionK?: number;
  /**
   * Candidates fetched per retriever, as a multiple of topK.
   *
   * Overfetching matters because fusion can only promote what was retrieved: a document
   * ranked 15th by dense and 3rd by sparse is invisible if only 10 were fetched. The
   * right multiple depends on how much the two retrievers disagree on your corpus.
   */
  overfetchFactor?: number;
  /** Embedding requests issued concurrently. Defaults to 3. */
  embeddingConcurrency?: number;
}

export interface SearchResult {
  chunks: Chunk[];
  /** Which retrievers contributed. Empty when both failed. */
  sources: Array<'dense' | 'sparse'>;
  /** Set when a retriever failed and the search continued without it. */
  degraded?: { retriever: 'dense' | 'sparse'; reason: string };
  /** Documents found by BOTH retrievers. High agreement is a strong relevance signal. */
  agreementCount: number;
}

interface ScoredChunk extends Chunk {
  score: number;
  source: 'dense' | 'sparse';
}

const DEFAULT_FUSION_K = 60;
const DEFAULT_OVERFETCH = 2;
const DEFAULT_EMBEDDING_CONCURRENCY = 3;
const EMBEDDING_BATCH_SIZE = 512;
const UPSERT_BATCH_SIZE = 100;

export class HybridSearch {
  private readonly embedding: EmbeddingAdapter;
  private readonly vectorStore: VectorStoreAdapter;
  private readonly sparseIndex: SparseIndexAdapter | undefined;
  private readonly denseWeight: number;
  private readonly sparseWeight: number;
  private readonly fusionK: number;
  private readonly overfetchFactor: number;
  private readonly embeddingConcurrency: number;

  constructor(config: HybridSearchConfig) {
    this.embedding = config.embedding;
    this.vectorStore = config.vectorStore;
    this.sparseIndex = config.sparseIndex;

    const denseWeight = config.denseWeight ?? 0.7;
    const sparseWeight = config.sparseWeight ?? 0.3;

    // Validated rather than accepted. Two zero weights produce a ranking driven purely
    // by Map insertion order, and a negative weight inverts relevance, both of which
    // look like poor retrieval quality rather than a configuration error.
    if (denseWeight < 0 || sparseWeight < 0) {
      throw new Error(
        `Fusion weights must be non-negative, received dense=${denseWeight} ` +
          `sparse=${sparseWeight}. A negative weight inverts relevance.`,
      );
    }

    if (denseWeight === 0 && sparseWeight === 0) {
      throw new Error(
        'Both fusion weights are zero, so every document would score identically and ' +
          'ranking would fall back to insertion order.',
      );
    }

    this.denseWeight = denseWeight;
    this.sparseWeight = sparseWeight;
    this.fusionK = config.fusionK ?? DEFAULT_FUSION_K;
    this.overfetchFactor = Math.max(1, config.overfetchFactor ?? DEFAULT_OVERFETCH);
    this.embeddingConcurrency = Math.max(
      1,
      config.embeddingConcurrency ?? DEFAULT_EMBEDDING_CONCURRENCY,
    );
  }

  /**
   * Run both retrievers and fuse.
   *
   * Uses allSettled rather than all: a sparse index outage should not fail a query that
   * dense retrieval alone can answer. The result reports which retriever was lost, so a
   * caller can distinguish degraded results from genuinely thin ones.
   */
  async search(query: string, topK: number): Promise<SearchResult> {
    if (topK < 1) {
      throw new Error(`topK must be at least 1, received ${topK}`);
    }

    const candidatePool = topK * this.overfetchFactor;

    const [dense, sparse] = await Promise.allSettled([
      this.denseSearch(query, candidatePool),
      this.sparseIndex
        ? this.sparseSearch(query, candidatePool)
        : Promise.resolve<ScoredChunk[]>([]),
    ]);

    const denseResults = dense.status === 'fulfilled' ? dense.value : [];
    const sparseResults = sparse.status === 'fulfilled' ? sparse.value : [];

    if (dense.status === 'rejected' && sparse.status === 'rejected') {
      throw new Error(
        'Both retrievers failed. Dense: ' +
          `${this.reasonOf(dense.reason)}. Sparse: ${this.reasonOf(sparse.reason)}.`,
      );
    }

    const sources: Array<'dense' | 'sparse'> = [];
    if (denseResults.length > 0) sources.push('dense');
    if (sparseResults.length > 0) sources.push('sparse');

    const fused = this.fuse(denseResults, sparseResults, topK);

    const degraded =
      dense.status === 'rejected'
        ? { retriever: 'dense' as const, reason: this.reasonOf(dense.reason) }
        : sparse.status === 'rejected'
          ? { retriever: 'sparse' as const, reason: this.reasonOf(sparse.reason) }
          : undefined;

    return {
      chunks: fused.chunks,
      sources,
      agreementCount: fused.agreementCount,
      ...(degraded ? { degraded } : {}),
    };
  }

  /**
   * Embed texts in concurrent batches.
   *
   * Previously sequential, which made 4000 texts eight serial round trips when they
   * could overlap. Order is preserved by writing into a pre-sized array rather than
   * appending, since batches complete out of order.
   */
  async embed(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    const batches: Array<{ start: number; texts: string[] }> = [];
    for (let i = 0; i < texts.length; i += EMBEDDING_BATCH_SIZE) {
      batches.push({ start: i, texts: texts.slice(i, i + EMBEDDING_BATCH_SIZE) });
    }

    const results = new Array<number[]>(texts.length);
    let cursor = 0;

    const worker = async (): Promise<void> => {
      for (;;) {
        const index = cursor++;
        const batch = batches[index];
        if (!batch) return;

        const embeddings = await this.embedding.embed(batch.texts);

        // A mismatch here silently misaligns every embedding after it, attaching each
        // vector to the wrong chunk. That corrupts the index rather than failing the
        // request, so it is a hard error.
        if (embeddings.length !== batch.texts.length) {
          throw new Error(
            `Embedding adapter returned ${embeddings.length} vectors for ` +
              `${batch.texts.length} inputs. Continuing would attach vectors to the ` +
              'wrong chunks and corrupt the index.',
          );
        }

        for (let j = 0; j < embeddings.length; j++) {
          results[batch.start + j] = embeddings[j]!;
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(this.embeddingConcurrency, batches.length) }, worker),
    );

    return results;
  }

  /**
   * Upsert chunks into both indexes.
   *
   * Sequential by design, unlike embedding. A vector store upsert is a write, and
   * concurrent writes to overlapping ids can land out of order, leaving the index
   * holding an older version of a chunk than the one last written.
   */
  async upsert(chunks: Chunk[]): Promise<void> {
    for (let i = 0; i < chunks.length; i += UPSERT_BATCH_SIZE) {
      const batch = chunks.slice(i, i + UPSERT_BATCH_SIZE);
      await this.vectorStore.upsert(batch);
      await this.sparseIndex?.index?.(batch);
    }
  }

  /**
   * Tokenize for sparse retrieval.
   *
   * Unicode-aware. The previous implementation used /[^a-z0-9\s]/g, which strips every
   * accented character: "cobrança" became "cobrana" and "São José" became "so jos".
   * Portuguese, Spanish, French and German sparse retrieval were silently broken, and
   * the symptom is poor relevance rather than an error, so nothing surfaces it.
   *
   * NFC normalization runs first so a precomposed "ç" and a "c" plus combining cedilla
   * produce the same token.
   */
  tokenize(text: string): string[] {
    return text
      .normalize('NFC')
      .toLowerCase()
      // Keep letters and numbers in any script; drop punctuation and symbols.
      .replace(/[^\p{L}\p{N}\s]/gu, ' ')
      .split(/\s+/)
      // Two characters rather than three: "id", "ip", "os" and "ui" are meaningful
      // technical terms, and a three-character floor discards all of them.
      .filter((token) => token.length >= 2);
  }

  private async denseSearch(query: string, topK: number): Promise<ScoredChunk[]> {
    const [queryEmbedding] = await this.embed([query]);

    if (!queryEmbedding) {
      throw new Error('Embedding adapter returned no vector for the query');
    }

    const results = await this.vectorStore.query(queryEmbedding, topK);

    return results.map((r) => ({ ...r.chunk, score: r.similarity, source: 'dense' }));
  }

  private async sparseSearch(query: string, topK: number): Promise<ScoredChunk[]> {
    if (!this.sparseIndex) return [];

    const tokens = this.tokenize(query);

    // Every term filtered out means BM25 has nothing to match. Returning early avoids
    // a pointless round trip and, more importantly, prevents an empty-token query from
    // being interpreted by an index as "match everything".
    if (tokens.length === 0) return [];

    const results = await this.sparseIndex.query(tokens, topK);

    return results.map((r) => ({ ...r.chunk, score: r.bm25Score, source: 'sparse' }));
  }

  /**
   * Reciprocal Rank Fusion.
   *
   *   RRF(d) = SUM over systems i of  w_i / (k + rank_i(d))
   *
   * Rank-based rather than score-based, and that choice is the point: dense retrieval
   * returns cosine similarities in [-1,1] while BM25 returns unbounded positive scores.
   * Normalizing them against each other requires distribution assumptions that break
   * across corpora, whereas ranks are directly comparable with no normalization.
   */
  private fuse(
    denseResults: readonly ScoredChunk[],
    sparseResults: readonly ScoredChunk[],
    topK: number,
  ): { chunks: Chunk[]; agreementCount: number } {
    const fused = new Map<
      string,
      { score: number; chunk: Chunk; systems: Set<'dense' | 'sparse'>; bestRank: number }
    >();

    const accumulate = (
      results: readonly ScoredChunk[],
      weight: number,
      system: 'dense' | 'sparse',
    ): void => {
      for (let rank = 0; rank < results.length; rank++) {
        const chunk = results[rank]!;
        const contribution = weight / (this.fusionK + rank + 1);
        const existing = fused.get(chunk.id);

        if (existing) {
          existing.score += contribution;
          existing.systems.add(system);
          existing.bestRank = Math.min(existing.bestRank, rank);
        } else {
          fused.set(chunk.id, {
            score: contribution,
            chunk,
            systems: new Set([system]),
            bestRank: rank,
          });
        }
      }
    };

    accumulate(denseResults, this.denseWeight, 'dense');
    accumulate(sparseResults, this.sparseWeight, 'sparse');

    const entries = [...fused.values()].sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      // Agreement first, then best rank. A document both retrievers found is a
      // stronger signal than one a single retriever ranked highly, and without an
      // explicit tie-break equal scores order by Map insertion, which is arbitrary.
      if (b.systems.size !== a.systems.size) return b.systems.size - a.systems.size;
      return a.bestRank - b.bestRank;
    });

    return {
      chunks: entries.slice(0, topK).map((entry) => entry.chunk),
      agreementCount: entries.filter((entry) => entry.systems.size > 1).length,
    };
  }

  private reasonOf(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
