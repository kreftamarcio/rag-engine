/**
 * Cross-encoder Reranker: reorders retrieval results by relevance.
 *
 * Why reranking matters:
 *   Bi-encoder retrieval (embedding similarity) is fast but imprecise.
 *   Cross-encoders process query+document pairs jointly, producing
 *   much more accurate relevance scores at the cost of latency.
 *
 * Architecture:
 *   1. Retrieve K candidates via fast bi-encoder search
 *   2. Score each candidate with cross-encoder (query, document) pairs
 *   3. Return top-N reranked results
 *
 * This two-stage pipeline gives near-cross-encoder quality at
 * near-bi-encoder latency (since K << corpus size).
 */

export interface RerankerConfig {
  /** Cross-encoder scoring function (injected for provider flexibility) */
  score: (query: string, documents: string[]) => Promise<number[]>;
  /** Number of top results to return after reranking */
  topK?: number;
  /** Minimum relevance score to include (0-1) */
  minScore?: number;
  /** Enable score normalization to 0-1 range */
  normalize?: boolean;
}

export interface RankedDocument {
  content: string;
  score: number;
  originalRank: number;
  metadata?: Record<string, unknown>;
}

export interface RerankerMetrics {
  candidatesReceived: number;
  candidatesReturned: number;
  avgScoreImprovement: number;
  latencyMs: number;
  /** How much the order changed (Kendall tau distance) */
  rankDisplacement: number;
}

export class Reranker {
  private readonly topK: number;
  private readonly minScore: number;
  private readonly normalize: boolean;

  constructor(private readonly config: RerankerConfig) {
    this.topK = config.topK ?? 5;
    this.minScore = config.minScore ?? 0.0;
    this.normalize = config.normalize ?? true;
  }

  /**
   * Rerank a set of candidate documents against a query.
   * Returns the top-K most relevant documents with scores.
   */
  async rerank(
    query: string,
    candidates: Array<{ content: string; metadata?: Record<string, unknown> }>,
  ): Promise<{ results: RankedDocument[]; metrics: RerankerMetrics }> {
    const startTime = performance.now();

    if (candidates.length === 0) {
      return {
        results: [],
        metrics: this.emptyMetrics(startTime),
      };
    }

    // Score all candidates with cross-encoder
    const documents = candidates.map(c => c.content);
    const rawScores = await this.config.score(query, documents);

    // Normalize scores if enabled
    const scores = this.normalize ? this.normalizeScores(rawScores) : rawScores;

    // Create ranked documents with original position
    const ranked: RankedDocument[] = candidates.map((candidate, index) => ({
      content: candidate.content,
      score: scores[index] ?? 0,
      originalRank: index,
      metadata: candidate.metadata,
    }));

    // Sort by score descending
    ranked.sort((a, b) => b.score - a.score);

    // Apply filters
    const filtered = ranked
      .filter(doc => doc.score >= this.minScore)
      .slice(0, this.topK);

    // Calculate metrics
    const latencyMs = performance.now() - startTime;
    const rankDisplacement = this.calculateDisplacement(ranked);
    const avgScoreImprovement = this.calculateScoreImprovement(ranked, candidates.length);

    return {
      results: filtered,
      metrics: {
        candidatesReceived: candidates.length,
        candidatesReturned: filtered.length,
        avgScoreImprovement,
        latencyMs,
        rankDisplacement,
      },
    };
  }

  /**
   * Reciprocal Rank Fusion: merges results from multiple retrieval strategies.
   * Useful when combining dense + sparse search results before reranking.
   */
  reciprocalRankFusion(
    ...resultSets: Array<Array<{ content: string; metadata?: Record<string, unknown> }>>
  ): Array<{ content: string; metadata?: Record<string, unknown>; rrfScore: number }> {
    const k = 60; // Standard RRF constant
    const scoreMap = new Map<string, { score: number; content: string; metadata?: Record<string, unknown> }>();

    for (const results of resultSets) {
      for (let rank = 0; rank < results.length; rank++) {
        const doc = results[rank]!;
        const rrfScore = 1 / (k + rank + 1);
        const key = doc.content.slice(0, 200); // Dedup key

        const existing = scoreMap.get(key);
        if (existing) {
          existing.score += rrfScore;
        } else {
          scoreMap.set(key, { score: rrfScore, content: doc.content, metadata: doc.metadata });
        }
      }
    }

    return [...scoreMap.values()]
      .sort((a, b) => b.score - a.score)
      .map(({ score, content, metadata }) => ({ content, metadata, rrfScore: score }));
  }

  private normalizeScores(scores: number[]): number[] {
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const range = max - min;

    if (range === 0) return scores.map(() => 1);
    return scores.map(s => (s - min) / range);
  }

  private calculateDisplacement(ranked: RankedDocument[]): number {
    // Kendall tau-like metric: average position change
    let totalDisplacement = 0;
    for (let i = 0; i < ranked.length; i++) {
      totalDisplacement += Math.abs(i - ranked[i]!.originalRank);
    }
    return totalDisplacement / ranked.length;
  }

  private calculateScoreImprovement(ranked: RankedDocument[], total: number): number {
    if (ranked.length === 0) return 0;
    const topScoreAvg = ranked.slice(0, this.topK).reduce((sum, d) => sum + d.score, 0) / this.topK;
    const overallAvg = ranked.reduce((sum, d) => sum + d.score, 0) / total;
    return topScoreAvg - overallAvg;
  }

  private emptyMetrics(startTime: number): RerankerMetrics {
    return {
      candidatesReceived: 0,
      candidatesReturned: 0,
      avgScoreImprovement: 0,
      latencyMs: performance.now() - startTime,
      rankDisplacement: 0,
    };
  }
}
