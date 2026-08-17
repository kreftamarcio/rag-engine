/**
 * Cross-encoder reranker: reorders retrieval results by true relevance.
 *
 * Why a second stage exists at all: a bi-encoder embeds query and document
 * independently, so it cannot model term interaction between them. A cross-encoder
 * sees both together and scores actual relevance, but is far too slow to run over a
 * corpus. Running it over K retrieved candidates gives near-cross-encoder quality at
 * near-bi-encoder latency, because K is tiny relative to the corpus.
 *
 * The cost is real: reranking is usually the most expensive step in a RAG pipeline,
 * which is why batching and candidate caps are first-class here rather than an
 * afterthought.
 */

export interface RerankerConfig {
  /** Cross-encoder scoring function, injected for provider flexibility. */
  score: (query: string, documents: string[]) => Promise<number[]>;
  /** Results returned after reranking. */
  topK?: number;
  /**
   * Minimum RAW score to include.
   *
   * Applied before normalization, deliberately. A threshold on normalized scores is
   * meaningless: normalization maps the observed minimum to 0 and maximum to 1, so
   * `minScore: 0.5` keeps roughly the top half of whatever it is given, whether those
   * are ten excellent candidates or ten irrelevant ones.
   */
  minScore?: number;
  /** Rescale returned scores to [0,1] for presentation. Does not affect filtering. */
  normalize?: boolean;
  /**
   * Documents per scoring call. Defaults to 32.
   *
   * A cross-encoder handed 200 pairs in one request either exceeds a payload limit or
   * times out, and the failure arrives only after the whole batch has been paid for.
   */
  batchSize?: number;
  /** Concurrent scoring batches. Defaults to 2. */
  concurrency?: number;
  /**
   * Hard cap on candidates considered. Excess is dropped from the tail, which is the
   * least promising end of a retrieval-ordered list.
   */
  maxCandidates?: number;
}

export interface RankedDocument {
  content: string;
  /** Normalized score when `normalize` is on, raw otherwise. */
  score: number;
  /** Raw cross-encoder score, always present so a threshold stays interpretable. */
  rawScore: number;
  originalRank: number;
  metadata?: Record<string, unknown>;
}

export interface RerankerMetrics {
  candidatesReceived: number;
  candidatesScored: number;
  candidatesReturned: number;
  /** Candidates dropped by maxCandidates before scoring. */
  candidatesTruncated: number;
  /** Candidates dropped by minScore after scoring. */
  candidatesFiltered: number;
  /**
   * Mean absolute rank change among the RETURNED documents.
   *
   * Near zero means reranking agreed with retrieval and is not earning its cost. A
   * consistently low value is a signal to reconsider the second stage entirely.
   */
  rankDisplacement: number;
  /** Mean raw score of returned documents minus mean of all scored. */
  scoreImprovement: number;
  latencyMs: number;
  batchCount: number;
}

const DEFAULT_BATCH_SIZE = 32;
const DEFAULT_CONCURRENCY = 2;

export class Reranker {
  private readonly topK: number;
  private readonly minScore: number;
  private readonly normalize: boolean;
  private readonly batchSize: number;
  private readonly concurrency: number;
  private readonly maxCandidates: number | undefined;

  constructor(private readonly config: RerankerConfig) {
    this.topK = config.topK ?? 5;
    this.minScore = config.minScore ?? Number.NEGATIVE_INFINITY;
    this.normalize = config.normalize ?? true;
    this.batchSize = Math.max(1, config.batchSize ?? DEFAULT_BATCH_SIZE);
    this.concurrency = Math.max(1, config.concurrency ?? DEFAULT_CONCURRENCY);
    this.maxCandidates = config.maxCandidates;

    if (this.topK < 1) {
      throw new Error(`topK must be at least 1, received ${this.topK}`);
    }
  }

  async rerank(
    query: string,
    candidates: ReadonlyArray<{ content: string; metadata?: Record<string, unknown> }>,
  ): Promise<{ results: RankedDocument[]; metrics: RerankerMetrics }> {
    const startedAt = performance.now();

    if (candidates.length === 0) {
      return { results: [], metrics: this.emptyMetrics(startedAt) };
    }

    // Truncated from the tail: the input arrives in retrieval order, so the tail is
    // the least promising end. Truncating the head would discard the best candidates.
    const considered =
      this.maxCandidates !== undefined && candidates.length > this.maxCandidates
        ? candidates.slice(0, this.maxCandidates)
        : candidates;

    const truncated = candidates.length - considered.length;

    const { scores: rawScores, batchCount } = await this.scoreInBatches(
      query,
      considered.map((c) => c.content),
    );

    // Filtering uses RAW scores. Normalization is applied afterwards and only for
    // presentation, so a configured threshold means the same thing on every call.
    const kept: Array<{ candidate: (typeof considered)[number]; raw: number; rank: number }> = [];

    for (let i = 0; i < considered.length; i++) {
      const raw = rawScores[i] ?? Number.NEGATIVE_INFINITY;
      if (raw >= this.minScore) {
        kept.push({ candidate: considered[i]!, raw, rank: i });
      }
    }

    kept.sort((a, b) => {
      if (b.raw !== a.raw) return b.raw - a.raw;
      // Ties resolve to the original retrieval order, which is a real signal rather
      // than arbitrary. Without this, equal scores order non-deterministically and
      // the same query can return a different ranking on each call.
      return a.rank - b.rank;
    });

    const selected = kept.slice(0, this.topK);

    // Normalization is computed over the SELECTED window, so the returned scores span
    // [0,1] across what the caller actually receives.
    const normalizedScores = this.normalize
      ? this.normalizeScores(selected.map((s) => s.raw))
      : selected.map((s) => s.raw);

    const results: RankedDocument[] = selected.map((entry, index) => ({
      content: entry.candidate.content,
      score: normalizedScores[index] ?? entry.raw,
      rawScore: entry.raw,
      originalRank: entry.rank,
      ...(entry.candidate.metadata !== undefined
        ? { metadata: entry.candidate.metadata }
        : {}),
    }));

    return {
      results,
      metrics: {
        candidatesReceived: candidates.length,
        candidatesScored: considered.length,
        candidatesReturned: results.length,
        candidatesTruncated: truncated,
        candidatesFiltered: considered.length - kept.length,
        rankDisplacement: this.displacement(results),
        scoreImprovement: this.improvement(results, rawScores),
        latencyMs: performance.now() - startedAt,
        batchCount,
      },
    };
  }

  /**
   * Reciprocal Rank Fusion over several retrieval strategies.
   *
   *   RRF(d) = sum over systems i of  w_i / (k + rank_i(d))
   *
   * Rank-based rather than score-based on purpose: dense retrieval returns cosine
   * similarities in [-1,1] while BM25 returns unbounded positive scores, and
   * normalizing them against each other requires distribution assumptions that break
   * across corpora. Ranks are directly comparable with no normalization at all.
   */
  reciprocalRankFusion(
    resultSets: ReadonlyArray<
      ReadonlyArray<{ content: string; metadata?: Record<string, unknown> }>
    >,
    options: { k?: number; weights?: number[] } = {},
  ): Array<{
    content: string;
    metadata?: Record<string, unknown>;
    rrfScore: number;
    /** How many retrieval systems returned this document. */
    systemCount: number;
  }> {
    const k = options.k ?? 60;

    if (options.weights && options.weights.length !== resultSets.length) {
      throw new Error(
        `weights has ${options.weights.length} entries for ${resultSets.length} result ` +
          'sets. Every set needs a weight, or omit weights entirely.',
      );
    }

    const fused = new Map<
      string,
      {
        score: number;
        content: string;
        metadata?: Record<string, unknown>;
        systems: Set<number>;
      }
    >();

    for (let setIndex = 0; setIndex < resultSets.length; setIndex++) {
      const results = resultSets[setIndex]!;
      // Weighted so a precise retriever can outvote a noisy one. Unweighted fusion
      // gives a sparse retriever with high precision the same say as a dense one that
      // returns loosely related material.
      const weight = options.weights?.[setIndex] ?? 1;

      for (let rank = 0; rank < results.length; rank++) {
        const doc = results[rank]!;
        // Keyed on the full content, not a prefix. A 200-character prefix collapses
        // two chunks from the same document into one, and the survivor absorbs the
        // other's rank contribution, inflating its score while silently dropping a
        // distinct passage.
        const key = this.contentKey(doc.content);

        const existing = fused.get(key);
        const contribution = weight / (k + rank + 1);

        if (existing) {
          existing.score += contribution;
          existing.systems.add(setIndex);
        } else {
          fused.set(key, {
            score: contribution,
            content: doc.content,
            systems: new Set([setIndex]),
            ...(doc.metadata !== undefined ? { metadata: doc.metadata } : {}),
          });
        }
      }
    }

    return [...fused.values()]
      .sort((a, b) => {
        if (b.score !== a.score) return b.score - a.score;
        // Agreement across systems breaks ties: a document found by three retrievers
        // is a stronger signal than one found by a single retriever at a similar rank.
        return b.systems.size - a.systems.size;
      })
      .map((entry) => ({
        content: entry.content,
        rrfScore: entry.score,
        systemCount: entry.systems.size,
        ...(entry.metadata !== undefined ? { metadata: entry.metadata } : {}),
      }));
  }

  /**
   * Score in bounded batches with bounded concurrency.
   *
   * A scorer that returns the wrong number of values is treated as a hard error rather
   * than padded: silently filling with zeros would rank real documents as irrelevant
   * and the cause would be invisible.
   */
  private async scoreInBatches(
    query: string,
    documents: string[],
  ): Promise<{ scores: number[]; batchCount: number }> {
    const batches: Array<{ start: number; documents: string[] }> = [];

    for (let i = 0; i < documents.length; i += this.batchSize) {
      batches.push({ start: i, documents: documents.slice(i, i + this.batchSize) });
    }

    const scores = new Array<number>(documents.length);
    let cursor = 0;

    const worker = async (): Promise<void> => {
      for (;;) {
        const index = cursor++;
        const batch = batches[index];
        if (!batch) return;

        const batchScores = await this.config.score(query, batch.documents);

        if (batchScores.length !== batch.documents.length) {
          throw new Error(
            `Scorer returned ${batchScores.length} scores for ${batch.documents.length} ` +
              'documents. Padding the difference would rank real documents as irrelevant ' +
              'with no visible cause, so this is a hard error.',
          );
        }

        for (let j = 0; j < batchScores.length; j++) {
          scores[batch.start + j] = batchScores[j]!;
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(this.concurrency, batches.length) }, worker),
    );

    return { scores, batchCount: batches.length };
  }

  /**
   * Min-max rescale.
   *
   * All-equal scores return 1 rather than 0: the scorer found every document equally
   * relevant, and reporting 0 would imply it found them all irrelevant.
   */
  private normalizeScores(scores: number[]): number[] {
    if (scores.length === 0) return [];

    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min;

    if (range === 0) return scores.map(() => 1);
    return scores.map((s) => (s - min) / range);
  }

  /**
   * Mean absolute rank change among returned documents.
   *
   * Measured over the returned window rather than the whole sorted array, because that
   * is the only part the caller sees, and normalized by the window size so the figure
   * is comparable across calls with different candidate counts.
   */
  private displacement(results: readonly RankedDocument[]): number {
    if (results.length === 0) return 0;

    let total = 0;
    for (let i = 0; i < results.length; i++) {
      total += Math.abs(i - results[i]!.originalRank);
    }

    return total / results.length;
  }

  /**
   * Mean raw score of returned documents minus mean of everything scored.
   *
   * Both means divide by their own population size. The previous implementation divided
   * the returned sum by `topK` even when fewer documents were returned, and divided the
   * overall sum by a `total` that did not match the array it summed.
   */
  private improvement(results: readonly RankedDocument[], allScores: readonly number[]): number {
    if (results.length === 0 || allScores.length === 0) return 0;

    const returnedMean =
      results.reduce((sum, d) => sum + d.rawScore, 0) / results.length;

    const overallMean = allScores.reduce((sum, s) => sum + s, 0) / allScores.length;

    return returnedMean - overallMean;
  }

  /**
   * Content key for deduplication.
   *
   * FNV-1a over the whole string. Cheap, and unlike a prefix it cannot collapse two
   * passages that happen to share an opening.
   */
  private contentKey(content: string): string {
    let hash = 0x811c9dc5;

    for (let i = 0; i < content.length; i++) {
      hash ^= content.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }

    // Length is appended because a 32-bit hash collides on large corpora, and two
    // documents of different length are certainly distinct.
    return `${(hash >>> 0).toString(36)}:${content.length}`;
  }

  private emptyMetrics(startedAt: number): RerankerMetrics {
    return {
      candidatesReceived: 0,
      candidatesScored: 0,
      candidatesReturned: 0,
      candidatesTruncated: 0,
      candidatesFiltered: 0,
      rankDisplacement: 0,
      scoreImprovement: 0,
      latencyMs: performance.now() - startedAt,
      batchCount: 0,
    };
  }
}
