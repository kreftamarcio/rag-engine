import type { Chunk } from '../core/engine.js';

/**
 * Hallucination scorer: verifies that a generated answer is grounded in its sources.
 *
 * Pipeline:
 *   1. Claim extraction    split the response into checkable factual statements
 *   2. Entailment check    for each claim, decide supported / contradicted / unmentioned
 *   3. Scoring             ratio of supported claims, with contradictions penalized
 *   4. Decision            accept / review / reject against a threshold
 *
 * Informed by the atomic-claim factuality literature: FActScore (Min et al., 2023) and
 * SAFE (Wei et al., 2024).
 *
 * A note on honesty: the built-in verifier is LEXICAL. It measures whether a claim's
 * vocabulary appears in the source, which is a proxy for entailment and not entailment
 * itself. It cannot detect a paraphrase with no shared vocabulary, and it cannot verify
 * a genuine logical inference. Supply an `entail` function for real NLI. The heuristic
 * exists so the pipeline degrades to something useful rather than to nothing.
 */

export type Verdict = 'supported' | 'contradicted' | 'unmentioned';

export interface Claim {
  text: string;
  verdict: Verdict;
  /** True only for 'supported'. Kept for callers that want the simple question. */
  supported: boolean;
  supportingChunkId?: string;
  /** Fraction of the claim's content found in the best-matching source, in [0,1]. */
  confidence: number;
}

export interface HallucinationConfig {
  enabled: boolean;
  /** Minimum grounding score to accept. */
  threshold: number;
  claimGranularity?: 'sentence' | 'atomic';
  /**
   * Fraction of a claim's content tokens that must appear in a source to count as
   * supported. Defaults to 0.6.
   *
   * This is CONTAINMENT, not similarity. See the comment on containment() for why
   * Jaccard is the wrong measure here.
   */
  containmentThreshold?: number;
  /**
   * Real entailment check. When provided, it decides the verdict and the lexical
   * heuristic is used only to preselect the most likely source chunk.
   */
  entail?: (claim: string, source: string) => Promise<Verdict>;
  /**
   * LLM-backed atomic decomposition. Required for claimGranularity: 'atomic'.
   *
   * Without it, requesting 'atomic' throws rather than silently downgrading, because a
   * caller who asked for atomic claims and received sentences gets a different score
   * and no indication why.
   */
  decompose?: (text: string) => Promise<string[]>;
}

export type ScoreOutcome =
  | { verifiable: true; groundingScore: number; claimCount: number }
  /** No sources, or no factual claims. Distinct from a score of 0. */
  | { verifiable: false; reason: 'no_sources' | 'no_claims' | 'empty_response' };

export interface Analysis {
  outcome: ScoreOutcome;
  claims: Claim[];
  unsupportedClaims: Claim[];
  /** Claims the source actively contradicts. The most serious category. */
  contradictedClaims: Claim[];
  recommendation: 'accept' | 'review' | 'reject' | 'unverifiable';
}

const DEFAULT_CONTAINMENT_THRESHOLD = 0.6;

/** Terms that flip polarity. A claim and a source that disagree on these contradict. */
const NEGATION_TOKENS = new Set([
  'not',
  'no',
  'never',
  'cannot',
  'without',
  'unsupported',
  'unavailable',
  'disabled',
  'nao',
  'nunca',
  'sem',
]);

export class HallucinationScorer {
  private readonly config: HallucinationConfig;
  private readonly containmentThreshold: number;

  constructor(config: HallucinationConfig) {
    this.config = config;
    this.containmentThreshold = config.containmentThreshold ?? DEFAULT_CONTAINMENT_THRESHOLD;

    if (config.threshold < 0 || config.threshold > 1) {
      throw new RangeError(`threshold must be within [0,1], received ${config.threshold}`);
    }

    // Fails at construction rather than at first call. An operator who selected atomic
    // granularity and silently received sentence granularity gets a different score
    // with no indication of why.
    if (config.claimGranularity === 'atomic' && !config.decompose) {
      throw new Error(
        "claimGranularity 'atomic' requires a decompose function. Without one this would " +
          "silently fall back to sentence granularity, producing a different score than " +
          'requested.',
      );
    }
  }

  /**
   * Grounding score for a response.
   *
   * Returns a discriminated outcome rather than a bare number, because "there was
   * nothing to verify" and "everything was fabricated" are opposite situations that a
   * single 0 conflates.
   */
  async score(response: string, sourceChunks: readonly Chunk[]): Promise<ScoreOutcome> {
    const analysis = await this.analyze(response, sourceChunks);
    return analysis.outcome;
  }

  /**
   * Full claim-level analysis.
   *
   * score() delegates here rather than duplicating the pipeline. Previously both methods
   * ran extraction and verification independently, so calling score() then analyze()
   * paid for everything twice.
   */
  async analyze(response: string, sourceChunks: readonly Chunk[]): Promise<Analysis> {
    if (!response.trim()) {
      return this.unverifiable('empty_response');
    }

    if (sourceChunks.length === 0) {
      // Not a score of 0. With no sources there is nothing to ground against, and
      // reporting total hallucination would be a claim the data cannot support.
      return this.unverifiable('no_sources');
    }

    const claimTexts = await this.extractClaims(response);

    if (claimTexts.length === 0) {
      return this.unverifiable('no_claims');
    }

    const claims = await this.verifyClaims(claimTexts, sourceChunks);

    const supported = claims.filter((c) => c.verdict === 'supported');
    const contradicted = claims.filter((c) => c.verdict === 'contradicted');

    // Contradictions are penalized beyond simply not counting as support. A source that
    // states the opposite is worse evidence than a source that is silent, so a
    // contradicted claim costs an additional half-claim against the score.
    const penalty = contradicted.length * 0.5;
    const rawScore = (supported.length - penalty) / claims.length;
    const groundingScore = Math.max(0, Math.round(rawScore * 100) / 100);

    return {
      outcome: { verifiable: true, groundingScore, claimCount: claims.length },
      claims,
      unsupportedClaims: claims.filter((c) => c.verdict !== 'supported'),
      contradictedClaims: contradicted,
      recommendation: this.recommend(groundingScore, contradicted.length),
    };
  }

  /**
   * Any contradiction forces at least 'review', regardless of score.
   *
   * A response can be 90% grounded and still assert the opposite of a source on the one
   * claim that matters. Averaging that away is how a confidently wrong answer ships.
   */
  private recommend(
    groundingScore: number,
    contradictionCount: number,
  ): Analysis['recommendation'] {
    if (contradictionCount > 0 && groundingScore >= this.config.threshold) {
      return 'review';
    }

    if (groundingScore >= this.config.threshold) return 'accept';
    if (groundingScore >= this.config.threshold * 0.7) return 'review';
    return 'reject';
  }

  private async extractClaims(text: string): Promise<string[]> {
    if (this.config.claimGranularity === 'atomic') {
      // Guaranteed present by the constructor check.
      const decomposed = await this.config.decompose!(text);
      const usable = decomposed.map((c) => c.trim()).filter((c) => c.length > 0);

      // An empty decomposition means the model failed, not that the text has no claims.
      // Falling back keeps the check running rather than silently reporting perfect
      // grounding for an unverified response.
      return usable.length > 0 ? usable : this.extractSentenceClaims(text);
    }

    return this.extractSentenceClaims(text);
  }

  /**
   * Sentence-level extraction.
   *
   * The abbreviation guard matters: splitting on every period turns "100 req/s. e.g.
   * burst" into fragments, and a fragment cannot be verified against anything.
   */
  private extractSentenceClaims(text: string): string[] {
    return text
      .replace(/\b(e\.g|i\.e|etc|vs|approx|no)\./gi, (match) => match.replace('.', '\u0001'))
      .split(/(?<=[.!?])\s+/)
      .map((s) => s.replace(/\u0001/g, '.').trim())
      // Only empty strings are dropped. The previous 10-character floor silently
      // excluded short factual sentences like "Rate limit is 100." from verification,
      // which is exactly the kind of specific claim worth checking.
      .filter((s) => s.length > 0)
      .filter((s) => this.isFactualStatement(s));
  }

  private async verifyClaims(
    claims: readonly string[],
    sourceChunks: readonly Chunk[],
  ): Promise<Claim[]> {
    return Promise.all(claims.map((claim) => this.verifyClaim(claim, sourceChunks)));
  }

  private async verifyClaim(claim: string, sourceChunks: readonly Chunk[]): Promise<Claim> {
    const claimTokens = this.tokenize(claim);

    let bestContainment = 0;
    let bestChunk: Chunk | undefined;

    for (const chunk of sourceChunks) {
      const score = this.containment(claimTokens, this.tokenize(chunk.content));

      if (score > bestContainment) {
        bestContainment = score;
        bestChunk = chunk;
      }
    }

    // A real NLI model decides the verdict when supplied. The lexical pass above still
    // runs, but only to pick which chunk to send, which keeps the entailment call to
    // one per claim instead of one per claim per chunk.
    if (this.config.entail && bestChunk) {
      const verdict = await this.config.entail(claim, bestChunk.content);

      return {
        text: claim,
        verdict,
        supported: verdict === 'supported',
        confidence: bestContainment,
        ...(verdict === 'supported' ? { supportingChunkId: bestChunk.id } : {}),
      };
    }

    const verdict = this.lexicalVerdict(claim, bestChunk, bestContainment);

    return {
      text: claim,
      verdict,
      supported: verdict === 'supported',
      confidence: bestContainment,
      ...(verdict === 'supported' && bestChunk ? { supportingChunkId: bestChunk.id } : {}),
    };
  }

  /**
   * Lexical verdict with contradiction detection.
   *
   * High overlap alone is not support. "The limit is 100 req/s" and "The limit is NOT
   * 100 req/s" share every content token, so pure overlap rates a direct contradiction
   * as perfectly grounded. That is the single most dangerous failure mode for a
   * grounding check, because it converts a caught error into a confident endorsement.
   *
   * Two signals distinguish them:
   *   - Polarity mismatch: one side negates and the other does not
   *   - Numeric mismatch: the same surrounding vocabulary with different figures
   */
  private lexicalVerdict(
    claim: string,
    bestChunk: Chunk | undefined,
    containment: number,
  ): Verdict {
    if (!bestChunk || containment < this.containmentThreshold) return 'unmentioned';

    if (this.polarityDiffers(claim, bestChunk.content)) return 'contradicted';
    if (this.numbersConflict(claim, bestChunk.content)) return 'contradicted';

    return 'supported';
  }

  /**
   * Containment, not Jaccard.
   *
   * Jaccard divides the intersection by the UNION, which makes it symmetric and
   * therefore wrong for this question. A 12-token claim compared against a 400-token
   * chunk has a union of roughly 400, so its Jaccard score caps near 0.03 and can never
   * clear a 0.3 threshold no matter how completely the source supports it.
   *
   * The actual question is "what fraction of this claim appears in the source", which is
   * |A intersect B| / |A|. That is asymmetric on purpose: a long source containing a
   * short claim should score 1.0.
   */
  private containment(claimTokens: Set<string>, sourceTokens: Set<string>): number {
    if (claimTokens.size === 0) return 0;

    let found = 0;
    for (const token of claimTokens) {
      if (sourceTokens.has(token)) found++;
    }

    return found / claimTokens.size;
  }

  private polarityDiffers(claim: string, source: string): boolean {
    const claimNegated = this.hasNegation(claim);
    const sourceNegated = this.hasNegation(source);

    // Only a mismatch matters. Both negated is agreement, and neither negated is
    // agreement.
    return claimNegated !== sourceNegated;
  }

  private hasNegation(text: string): boolean {
    return this.tokenizeRaw(text).some((token) => NEGATION_TOKENS.has(token));
  }

  /**
   * Detect conflicting figures.
   *
   * Only fires when the claim has numbers and the source has numbers and NONE of the
   * claim's appear. A claim citing a figure absent from an otherwise well-matched source
   * is a fabricated specific, which is the most common concrete hallucination.
   */
  private numbersConflict(claim: string, source: string): boolean {
    const claimNumbers = this.extractNumbers(claim);
    if (claimNumbers.length === 0) return false;

    const sourceNumbers = new Set(this.extractNumbers(source));
    if (sourceNumbers.size === 0) return false;

    return !claimNumbers.some((n) => sourceNumbers.has(n));
  }

  private extractNumbers(text: string): string[] {
    // Separators are stripped so "1,000" and "1000" compare equal, and a trailing
    // decimal zero is normalized so "100" matches "100.0".
    return (text.match(/\d[\d.,]*/g) ?? []).map((raw) => {
      const normalized = raw.replace(/[.,](?=\d{3}\b)/g, '').replace(/\.0+$/, '');
      return normalized;
    });
  }

  private isFactualStatement(sentence: string): boolean {
    const nonFactual = [
      /^(i think|in my opinion|maybe|perhaps|it seems|possibly)\b/i,
      /\?\s*$/,
      // A bare connective with no content. Anchored to the full string so a sentence
      // merely STARTING with "However" is still checked.
      /^(however|but|also|additionally|furthermore),?\s*$/i,
      // Meta-commentary about the sources rather than a claim drawn from them.
      /^(based on the (provided )?(context|documents|sources))\b.{0,20}$/i,
    ];

    return !nonFactual.some((pattern) => pattern.test(sentence.trim()));
  }

  /**
   * Unicode-aware tokenization.
   *
   * The previous implementation used /[^a-z0-9\s]/g, which strips accents: "cobrança"
   * became "cobrana" and no longer matched the source. Accented content scored as
   * ungrounded regardless of how well supported it actually was.
   */
  private tokenize(text: string): Set<string> {
    return new Set(this.tokenizeRaw(text));
  }

  private tokenizeRaw(text: string): string[] {
    return text
      .normalize('NFC')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 0);
  }

  private unverifiable(reason: 'no_sources' | 'no_claims' | 'empty_response'): Analysis {
    return {
      outcome: { verifiable: false, reason },
      claims: [],
      unsupportedClaims: [],
      contradictedClaims: [],
      recommendation: 'unverifiable',
    };
  }
}
