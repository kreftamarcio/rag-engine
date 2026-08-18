import { describe, it, expect } from 'vitest';
import { HallucinationScorer } from '../src/generation/hallucination-scorer.js';
import type { Chunk } from '../src/core/config.js';

function makeChunk(id: string, content: string): Chunk {
  return {
    id,
    content,
    metadata: {
      source: `test/${id}.md`,
      chunkIndex: 0,
      totalChunks: 1,
      startOffset: 0,
      endOffset: content.length,
      tokenCount: content.split(/\s+/).length,
    },
  };
}

function makeScorer(overrides: Partial<Parameters<typeof HallucinationScorer['prototype']['analyze']> extends never ? never : Record<string, unknown>> = {}) {
  return new HallucinationScorer({
    enabled: true,
    threshold: 0.8,
    claimGranularity: 'sentence',
    containmentThreshold: 0.6,
    ...overrides,
  } as any);
}

describe('HallucinationScorer', () => {
  // ── Discriminated output (not a bare number) ──

  it('returns verifiable:false with reason empty_response for blank input', async () => {
    const scorer = makeScorer();
    const result = await scorer.score('', [makeChunk('a', 'anything')]);
    expect(result).toEqual({ verifiable: false, reason: 'empty_response' });
  });

  it('returns verifiable:false with reason no_sources when chunks are empty', async () => {
    const scorer = makeScorer();
    const result = await scorer.score('The limit is 100.', []);
    expect(result).toEqual({ verifiable: false, reason: 'no_sources' });
  });

  it('returns verifiable:false with reason no_claims for non-factual text', async () => {
    const scorer = makeScorer();
    // Questions and hedged opinions are filtered by isFactualStatement
    const result = await scorer.score('Maybe? I think so?', [makeChunk('a', 'anything')]);
    expect(result).toEqual({ verifiable: false, reason: 'no_claims' });
  });

  // ── Containment, not Jaccard ──
  // Design decision from hallucination-scorer.ts:
  // "Jaccard divides by the union, so a 12-token claim against a 400-token
  //  chunk caps near 0.03 and can never clear a 0.3 threshold."

  it('scores a short claim against a long source as supported via containment', async () => {
    const scorer = makeScorer({ containmentThreshold: 0.6 });
    const shortClaim = 'The rate limit is 1000 requests per minute.';
    const longSource =
      'Configuration reference for the API gateway. ' +
      'The rate limit is 1000 requests per minute per API key. ' +
      'Exceeding this limit returns HTTP 429. ' +
      'Contact support for higher quotas. ' +
      'Rate limits are enforced at the edge, not at the origin. ' +
      'Burst allowance is 50 requests above the sustained limit.';

    const analysis = await scorer.analyze(shortClaim, [makeChunk('docs', longSource)]);

    // Containment of claim tokens in source should be high.
    // Jaccard would be ~0.05 here because the union is dominated by the source.
    expect(analysis.claims[0]?.verdict).toBe('supported');
    expect(analysis.outcome).toHaveProperty('verifiable', true);
  });

  // ── Polarity mismatch → contradicted ──

  it('detects polarity mismatch as contradiction', async () => {
    const scorer = makeScorer({ containmentThreshold: 0.3 });
    const claim = 'The feature is not available in the free tier.';
    const source = 'The feature is available in the free tier.';

    const analysis = await scorer.analyze(claim, [makeChunk('pricing', source)]);

    expect(analysis.contradictedClaims.length).toBeGreaterThanOrEqual(1);
    expect(analysis.contradictedClaims[0]?.verdict).toBe('contradicted');
  });

  // ── Numeric conflict → contradicted ──

  it('detects conflicting numbers as contradiction', async () => {
    const scorer = makeScorer({ containmentThreshold: 0.3 });
    const claim = 'The timeout is 30 seconds.';
    const source = 'The timeout is 60 seconds.';

    const analysis = await scorer.analyze(claim, [makeChunk('config', source)]);

    expect(analysis.contradictedClaims.length).toBeGreaterThanOrEqual(1);
  });

  // ── Any contradiction forces review ──
  // From the source: "A response can be 90% grounded and assert the opposite of a
  // source on the one claim that matters; averaging that away is how a confidently
  // wrong answer ships."

  it('forces review recommendation when any claim is contradicted, even with high score', async () => {
    const scorer = makeScorer({ threshold: 0.5, containmentThreshold: 0.3 });
    // 3 supported claims + 1 contradicted = score > 0.5 threshold,
    // but the contradiction should force at least 'review'
    const claims =
      'The API uses REST. Authentication is via API key. Rate limiting is enforced. The timeout is 30 seconds.';
    const source =
      'The API uses REST. Authentication is via API key. Rate limiting is enforced. The timeout is 60 seconds.';

    const analysis = await scorer.analyze(claims, [makeChunk('api', source)]);

    // Even if 3/4 claims pass, the contradicted one forces review
    if (analysis.contradictedClaims.length > 0) {
      expect(['review', 'reject']).toContain(analysis.recommendation);
      expect(analysis.recommendation).not.toBe('accept');
    }
  });

  // ── Config validation at construction ──

  it('throws on threshold outside [0,1]', () => {
    expect(() => new HallucinationScorer({
      enabled: true,
      threshold: 1.5,
      claimGranularity: 'sentence',
    })).toThrow();

    expect(() => new HallucinationScorer({
      enabled: true,
      threshold: -0.1,
      claimGranularity: 'sentence',
    })).toThrow();
  });

  it('throws when atomic granularity is requested without decompose function', () => {
    // From the source: "Fails at construction rather than at first call. An operator
    // who selected atomic granularity and silently received sentence granularity gets
    // a different score with no indication of why."
    expect(() => new HallucinationScorer({
      enabled: true,
      threshold: 0.8,
      claimGranularity: 'atomic',
      // decompose intentionally omitted
    })).toThrow();
  });

  // ── Sentence extraction filters non-factual ──

  it('filters questions and hedged opinions from claim extraction', async () => {
    const scorer = makeScorer();
    const mixed = 'The limit is 100. Is that enough? I think it depends. The API returns JSON.';
    const source = 'The limit is 100. The API returns JSON.';

    const analysis = await scorer.analyze(mixed, [makeChunk('docs', source)]);

    // Only factual statements should be extracted as claims
    const claimTexts = analysis.claims.map(c => c.text);
    for (const claim of claimTexts) {
      expect(claim).not.toMatch(/\?\s*$/);
      expect(claim).not.toMatch(/^I think/i);
    }
  });

  // ── NLI entail function is used when provided ──

  it('delegates to entail function when provided', async () => {
    let entailCalled = false;
    const scorer = new HallucinationScorer({
      enabled: true,
      threshold: 0.8,
      claimGranularity: 'sentence',
      containmentThreshold: 0.3,
      entail: async (_claim, _source) => {
        entailCalled = true;
        return 'supported';
      },
    });

    await scorer.analyze(
      'The sky is blue.',
      [makeChunk('facts', 'The sky is blue during clear weather.')],
    );

    expect(entailCalled).toBe(true);
  });

  // ── Unmentioned vs contradicted ──

  it('returns unmentioned when claim has no overlap with any source', async () => {
    const scorer = makeScorer({ containmentThreshold: 0.6 });
    const claim = 'PostgreSQL supports window functions.';
    const source = 'Redis is an in-memory key-value store used for caching.';

    const analysis = await scorer.analyze(claim, [makeChunk('redis', source)]);

    expect(analysis.claims[0]?.verdict).toBe('unmentioned');
  });
});
