/**
 * Example 02 — Full pipeline.
 *
 * Every stage enabled: hybrid retrieval (dense + BM25), cross-encoder reranking, citation
 * attribution, and grounding verification.
 *
 * Run:
 *   npx tsx examples/02-full-pipeline.ts
 *
 * Three things this example is built to demonstrate rather than assert:
 *
 *   1. Sparse retrieval earns its place. The query for error code "E4021" is one the
 *      lexical embedder handles poorly and BM25 handles well, because BM25 matches the
 *      exact token while a TF vector dilutes it among common words.
 *
 *   2. The grounding scorer catches a contradiction. The last check feeds it a claim that
 *      directly contradicts the sources, and the verdict comes back 'contradicted' rather
 *      than merely unsupported. A checker that only ever prints 'supported' teaches
 *      nothing.
 *
 *   3. Citation markers match the bibliography. Marker [1] refers to bibliography entry
 *      1, which is the property that makes a citation verifiable at all.
 */

import { RAGEngine, HallucinationScorer } from '../src/index.js';
import type { Document, Chunk } from '../src/index.js';
import {
  HashingEmbedder,
  MemoryVectorStore,
  MemoryBM25Index,
  ExtractiveGenerator,
  lexicalRerankerScore,
} from './in-memory-adapters.js';

const documents: Document[] = [
  {
    content: [
      'Error E4021 means the request payload exceeded the maximum size.',
      'The maximum payload size is 10 megabytes for all plans.',
      'Split large uploads into multiple requests to avoid this error.',
    ].join(' '),
    metadata: { source: 'docs/errors.md', title: 'Error Reference' },
  },
  {
    content: [
      'The default rate limit is 1000 requests per minute per key.',
      'Exceeding the limit returns status 429 with a Retry-After header.',
      'Enterprise plans raise the limit to 10000 requests per minute.',
    ].join(' '),
    metadata: { source: 'docs/rate-limits.md', title: 'Rate Limits' },
  },
  {
    content: [
      'Webhooks are signed with an HMAC SHA-256 signature.',
      'The signature appears in the X-Signature header of every delivery.',
      'Verify it before trusting the payload.',
    ].join(' '),
    metadata: { source: 'docs/webhooks.md', title: 'Webhooks' },
  },
];

async function main(): Promise<void> {
  const embedding = new HashingEmbedder();

  const engine = new RAGEngine({
    embedding,
    vectorStore: new MemoryVectorStore(),
    // Without this, an exact-token query like "E4021" relies entirely on dense retrieval.
    sparseIndex: new MemoryBM25Index(),
    generation: new ExtractiveGenerator(),

    chunking: { maxTokens: 40, overlap: 8 },

    retrieval: {
      // Sparse weighted higher than the usual 0.3 because the lexical embedder here is
      // weak, so BM25 is the more trustworthy of the two signals in this example.
      denseWeight: 0.4,
      sparseWeight: 0.6,
      overfetchFactor: 3,
    },

    reranker: {
      score: lexicalRerankerScore,
      topK: 3,
      // A raw threshold, applied before normalization. Normalized thresholds are
      // meaningless: they always keep roughly the top fraction regardless of relevance.
      minScore: 0.2,
      batchSize: 8,
    },

    citations: {
      embed: (texts) => embedding.embed(texts),
      // Low because the lexical embedder produces lower similarities than a real model.
      // A production threshold of 0.65 would cite nothing here.
      citationThreshold: 0.25,
      maxCitationsPerSentence: 2,
    },

    hallucination: {
      enabled: true,
      threshold: 0.8,
      claimGranularity: 'sentence',
      containmentThreshold: 0.6,
    },

    logger: {
      debug: () => undefined,
      info: (message, meta) => console.log(`[info] ${message}`, meta ?? ''),
      warn: (message) => console.warn(`[warn] ${message}`),
      error: (message) => console.error(`[error] ${message}`),
    },
  });

  await engine.ingest(documents);

  for (const question of ['What does error E4021 mean?', 'How are webhooks verified?']) {
    const result = await engine.query(question, { topK: 8, contextSize: 3 });

    console.log(`\n=== ${question} ===`);
    console.log(`answer:      ${result.answer || '(no context retrieved)'}`);
    console.log(`retrieved:   ${result.metadata.chunksRetrieved}`);
    console.log(`context:     ${result.metadata.chunksAfterReranking}`);
    console.log(`calls:       ${result.metadata.inferenceCalls}`);

    if (result.metadata.degraded) {
      // Printed because a degraded result is not a thin one, and conflating them sends
      // an operator looking for a relevance problem when a retriever is down.
      console.log(`degraded:    ${result.metadata.degraded.retriever} — ${result.metadata.degraded.reason}`);
    }

    console.log(`grounding:   ${JSON.stringify(result.grounding)}`);

    for (const citation of result.citations) {
      console.log(`  [${citation.marker}] ${citation.source} (${citation.relevance.toFixed(3)})`);
    }

    for (const entry of result.bibliography ?? []) {
      console.log(`  bib [${entry.marker}] ${entry.title ?? entry.sourceIndex}`);
    }
  }

  // ---------------------------------------------------------------------------
  // Contradiction detection, checked directly against the scorer.
  //
  // Run standalone rather than through query(), because the extractive generator copies
  // from the sources and therefore cannot produce a contradiction. Fabricating one is the
  // only way to show the detector working.
  // ---------------------------------------------------------------------------

  const scorer = new HallucinationScorer({
    enabled: true,
    threshold: 0.8,
    claimGranularity: 'sentence',
  });

  const sourceChunk: Chunk = {
    id: 'rate-limits#0',
    content: 'The default rate limit is 1000 requests per minute per key.',
    metadata: {
      source: 'docs/rate-limits.md',
      chunkIndex: 0,
      totalChunks: 1,
      startOffset: 0,
      endOffset: 59,
      tokenCount: 12,
    },
  };

  const cases = [
    // Supported: same figure, same polarity.
    'The default rate limit is 1000 requests per minute per key.',
    // Contradicted by polarity: identical vocabulary, negated.
    'The default rate limit is not 1000 requests per minute per key.',
    // Contradicted by figure: same vocabulary, different number.
    'The default rate limit is 5000 requests per minute per key.',
    // Unmentioned: the source says nothing about this.
    'Webhook signatures use HMAC SHA-512.',
  ];

  console.log('\n=== grounding verdicts ===');

  for (const claim of cases) {
    const analysis = await scorer.analyze(claim, [sourceChunk]);
    const verdict = analysis.claims[0]?.verdict ?? 'no claim extracted';

    console.log(`${verdict.padEnd(14)} ${analysis.recommendation.padEnd(13)} ${claim}`);
  }

  engine.dispose();
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
