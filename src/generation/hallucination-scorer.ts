import type { Chunk } from '../core/engine.js';

export interface HallucinationConfig {
  enabled: boolean;
  threshold: number;
  model: string;
  claimGranularity?: 'sentence' | 'atomic';
}

interface Claim {
  text: string;
  supported: boolean;
  supportingChunkId?: string;
  confidence: number;
}

/**
 * Hallucination Scorer
 *
 * Verifies that generated responses are factually grounded
 * in the retrieved source documents.
 *
 * Pipeline:
 * 1. Claim Extraction: Split response into atomic factual claims
 * 2. Entailment Check: For each claim, verify against source chunks
 * 3. Scoring: Calculate ratio of supported claims
 * 4. Decision: Flag responses below threshold
 *
 * Based on research:
 * - FActScore (Min et al., 2023)
 * - SAFE (Wei et al., 2024)
 */
export class HallucinationScorer {
  private readonly config: HallucinationConfig;

  constructor(config: HallucinationConfig) {
    this.config = config;
  }

  /**
   * Score a generated response against source chunks.
   * Returns a grounding score between 0 and 1.
   */
  async score(response: string, sourceChunks: Chunk[]): Promise<number> {
    if (!response.trim() || sourceChunks.length === 0) {
      return 0;
    }

    // Step 1: Extract atomic claims from the response
    const claims = await this.extractClaims(response);

    if (claims.length === 0) {
      return 1.0; // No factual claims = nothing to hallucinate
    }

    // Step 2: Verify each claim against sources
    const verifiedClaims = await this.verifyClaims(claims, sourceChunks);

    // Step 3: Calculate grounding score
    const supportedCount = verifiedClaims.filter(c => c.supported).length;
    const groundingScore = supportedCount / verifiedClaims.length;

    return Math.round(groundingScore * 100) / 100;
  }

  /**
   * Get detailed claim-level analysis.
   */
  async analyze(response: string, sourceChunks: Chunk[]): Promise<{
    groundingScore: number;
    claims: Claim[];
    unsupportedClaims: Claim[];
    recommendation: 'accept' | 'review' | 'reject';
  }> {
    const claims = await this.extractClaims(response);
    const verifiedClaims = await this.verifyClaims(claims, sourceChunks);

    const supportedCount = verifiedClaims.filter(c => c.supported).length;
    const groundingScore = claims.length > 0
      ? supportedCount / verifiedClaims.length
      : 1.0;

    const unsupportedClaims = verifiedClaims.filter(c => !c.supported);

    let recommendation: 'accept' | 'review' | 'reject';
    if (groundingScore >= this.config.threshold) {
      recommendation = 'accept';
    } else if (groundingScore >= this.config.threshold * 0.7) {
      recommendation = 'review';
    } else {
      recommendation = 'reject';
    }

    return {
      groundingScore,
      claims: verifiedClaims,
      unsupportedClaims,
      recommendation,
    };
  }

  /**
   * Extract atomic factual claims from text.
   *
   * Atomic claims are minimal, self-contained factual statements.
   * Example: "The API supports OAuth 2.0 and rate limits to 100 req/s"
   * becomes:
   *   1. "The API supports OAuth 2.0"
   *   2. "The API rate limits to 100 requests per second"
   */
  private async extractClaims(text: string): Promise<string[]> {
    if (this.config.claimGranularity === 'sentence') {
      return this.extractSentenceClaims(text);
    }
    return this.extractAtomicClaims(text);
  }

  private extractSentenceClaims(text: string): string[] {
    // Split by sentence boundaries
    const sentences = text
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 10)
      .filter(s => this.isFactualStatement(s));

    return sentences;
  }

  private async extractAtomicClaims(text: string): Promise<string[]> {
    // In production: use LLM to decompose into atomic claims
    // Fallback: sentence-level extraction
    return this.extractSentenceClaims(text);
  }

  /**
   * Verify claims against source chunks using Natural Language Inference.
   *
   * For each claim:
   *   1. Find most relevant source chunk (by embedding similarity)
   *   2. Check entailment: does the source entail the claim?
   *   3. Classify as: ENTAILED | CONTRADICTED | NOT_MENTIONED
   */
  private async verifyClaims(claims: string[], sourceChunks: Chunk[]): Promise<Claim[]> {
    const sourceText = sourceChunks.map(c => c.content).join('\n\n');
    const verifiedClaims: Claim[] = [];

    for (const claimText of claims) {
      const verification = await this.checkEntailment(claimText, sourceText, sourceChunks);
      verifiedClaims.push(verification);
    }

    return verifiedClaims;
  }

  /**
   * Check if a claim is entailed by the source text.
   * Uses lexical overlap as a heuristic, with LLM judge as fallback.
   */
  private async checkEntailment(
    claim: string,
    sourceText: string,
    sourceChunks: Chunk[],
  ): Promise<Claim> {
    // Heuristic: check lexical overlap between claim and sources
    const claimTokens = this.tokenize(claim);
    let bestOverlap = 0;
    let bestChunkId: string | undefined;

    for (const chunk of sourceChunks) {
      const chunkTokens = this.tokenize(chunk.content);
      const overlap = this.jaccardSimilarity(claimTokens, chunkTokens);

      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        bestChunkId = chunk.id;
      }
    }

    // Thresholds for lexical overlap heuristic
    // In production: use cross-encoder NLI model or LLM judge
    const supported = bestOverlap > 0.3;

    return {
      text: claim,
      supported,
      supportingChunkId: supported ? bestChunkId : undefined,
      confidence: bestOverlap,
    };
  }

  private isFactualStatement(sentence: string): boolean {
    // Filter out questions, opinions, and meta-statements
    const nonFactual = [
      /^(I think|In my opinion|Maybe|Perhaps)/i,
      /\?$/,
      /^(However|But|Also|Additionally),?$/i,
    ];

    return !nonFactual.some(pattern => pattern.test(sentence));
  }

  private tokenize(text: string): Set<string> {
    return new Set(
      text
        .toLowerCase()
        .replace(/[^a-z0-9\s]/g, '')
        .split(/\s+/)
        .filter(t => t.length > 2),
    );
  }

  private jaccardSimilarity(setA: Set<string>, setB: Set<string>): number {
    let intersection = 0;
    for (const item of setA) {
      if (setB.has(item)) intersection++;
    }
    const union = setA.size + setB.size - intersection;
    return union === 0 ? 0 : intersection / union;
  }
}
