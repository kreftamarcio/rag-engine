/**
 * Example 01 — Hello world.
 *
 * The smallest configuration that answers a question: an embedder, a vector store, a
 * generator, and a chunking policy. Reranking, citations, grounding checks and query
 * rewriting are all off.
 *
 * Run:
 *   npx tsx examples/01-hello-world.ts
 *
 * No API key, no database, no network.
 *
 * The last query is included on purpose and is EXPECTED to fail: it asks about "signing
 * in" while the document says "authentication". The lexical embedder shares no vocabulary
 * between them, so retrieval misses. That is the exact weakness a real embedding model
 * exists to fix, and seeing it in the output is more useful than reading about it.
 */

import { RAGEngine } from '../src/index.js';
import type { Document } from '../src/index.js';
import {
  HashingEmbedder,
  MemoryVectorStore,
  ExtractiveGenerator,
} from './in-memory-adapters.js';

const documents: Document[] = [
  {
    content: [
      'The API requires authentication on every request.',
      'Send your key in the Authorization header as a bearer token.',
      'Requests without a valid key are rejected with status 401.',
      'Keys are created in the dashboard under Settings, then API Keys.',
    ].join(' '),
    metadata: { source: 'docs/authentication.md', title: 'Authentication' },
  },
  {
    content: [
      'The default rate limit is 1000 requests per minute per key.',
      'Exceeding the limit returns status 429 with a Retry-After header.',
      'The header states how many seconds to wait before retrying.',
      'Enterprise plans raise the limit to 10000 requests per minute.',
    ].join(' '),
    metadata: { source: 'docs/rate-limits.md', title: 'Rate Limits' },
  },
];

async function main(): Promise<void> {
  const engine = new RAGEngine({
    embedding: new HashingEmbedder(),
    vectorStore: new MemoryVectorStore(),
    generation: new ExtractiveGenerator(),
    chunking: {
      // Small on purpose. With 512 tokens these short documents would each be a single
      // chunk, and the example would demonstrate nothing about chunking.
      maxTokens: 40,
      overlap: 8,
    },
  });

  const ingestion = await engine.ingest(documents);

  console.log('--- ingestion ---');
  console.log(`documents processed: ${ingestion.documentsProcessed}`);
  console.log(`chunks created:      ${ingestion.chunksCreated}`);
  console.log(`errors:              ${ingestion.errors.length}`);

  const questions = [
    'What is the rate limit?',
    'How do I authenticate?',
    // Expected to miss: "signing in" shares no vocabulary with "authentication".
    'How do I sign in?',
  ];

  for (const question of questions) {
    const result = await engine.query(question, { topK: 6, contextSize: 2 });

    console.log(`\n--- query: ${question} ---`);

    if (result.answer.length === 0) {
      // The engine returns an empty answer rather than generating without context. An
      // answer produced from no sources would have nothing grounding it at all.
      console.log('answer:   (no context retrieved, generation skipped)');
    } else {
      console.log(`answer:   ${result.answer}`);
    }

    console.log(`chunks:   ${result.metadata.chunksAfterReranking} of ${result.metadata.chunksRetrieved} retrieved`);
    console.log(`calls:    ${result.metadata.inferenceCalls} inference call(s)`);
    console.log(`grounding: ${result.grounding === null ? 'not configured' : JSON.stringify(result.grounding)}`);
  }

  // Frees the tiktoken encoder the chunker holds. Without this the process can hang on
  // exit, because the native handle keeps the event loop alive.
  engine.dispose();
}

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
