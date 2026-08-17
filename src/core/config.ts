/**
 * Configuration contract for the RAG pipeline.
 *
 * Every field here is derived from a component that exists in this repository. The
 * shape is deliberately adapter-based rather than provider-name based: HybridSearch
 * needs an object with an `embed` method, not the string "openai", and inventing a
 * string-to-provider registry that does not exist is how documentation starts lying.
 *
 * Consequence worth knowing: this library reads NO environment variables. Credentials
 * live in the adapters the caller constructs, so there is nothing for the library to
 * read from the environment and no `.env` surface of its own.
 */

import type {
  EmbeddingAdapter,
  VectorStoreAdapter,
  SparseIndexAdapter,
} from '../retrieval/hybrid-search.js';
import type { RerankerConfig } from '../retrieval/reranker.js';
import type { HallucinationConfig, ScoreOutcome } from '../generation/hallucination-scorer.js';
import type { CitationConfig, CitedResponse } from '../generation/citation-extractor.js';
import type { RecursiveChunkerConfig } from '../ingestion/chunkers/recursive.chunker.js';
import type { RewriteStrategy, ConversationTurn } from '../retrieval/query-rewriter.js';

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

export interface Document {
  content: string;
  metadata: {
    source: string;
    title?: string;
    author?: string;
    createdAt?: Date;
    [key: string]: unknown;
  };
}

export interface Chunk {
  id: string;
  content: string;
  embedding?: number[];
  metadata: {
    source: string;
    chunkIndex: number;
    totalChunks: number;
    parentId?: string;
    startOffset: number;
    endOffset: number;
    tokenCount: number;
    [key: string]: unknown;
  };
}

export interface Citation {
  /** Source identifier, taken from the originating chunk's metadata. */
  source: string;
  /** The passage that supports the sentence. */
  span: string;
  relevance: number;
  chunkId: string;
  /** Marker as it appears inline in the answer, matching the bibliography. */
  marker: number;
}

// ---------------------------------------------------------------------------
// Adapters
// ---------------------------------------------------------------------------

/**
 * Text generation.
 *
 * Injected rather than configured by provider name. The engine has no business knowing
 * which vendor answered, and a name string cannot carry credentials, retry policy or a
 * base URL.
 */
export interface GenerationAdapter {
  complete(prompt: string): Promise<string>;
}

/**
 * Minimal logger contract.
 *
 * Structural rather than a concrete dependency, so a caller can pass pino, console, or
 * nothing. `silentLogger` is the default so engine code can always call the logger
 * without null checks.
 */
export interface Logger {
  debug(message: string, meta?: Record<string, unknown>): void;
  info(message: string, meta?: Record<string, unknown>): void;
  warn(message: string, meta?: Record<string, unknown>): void;
  error(message: string, meta?: Record<string, unknown>): void;
}

export const silentLogger: Logger = {
  debug: () => undefined,
  info: () => undefined,
  warn: () => undefined,
  error: () => undefined,
};

// ---------------------------------------------------------------------------
// Engine configuration
// ---------------------------------------------------------------------------

export interface RAGConfig {
  /** Required. Embeds queries during retrieval and chunks during ingestion. */
  embedding: EmbeddingAdapter;
  /** Required. Dense retrieval and chunk persistence. */
  vectorStore: VectorStoreAdapter;
  /**
   * Optional. Without it, retrieval is dense-only.
   *
   * Omitting it is a real trade-off, not a minor one: dense retrieval alone misses
   * exact-term matches, so an error code or identifier will retrieve conceptually
   * similar documents rather than the one that names it.
   */
  sparseIndex?: SparseIndexAdapter;
  /** Required. Produces the answer from the assembled context. */
  generation: GenerationAdapter;

  chunking: RecursiveChunkerConfig;

  /** Fusion weights and overfetch. Defaults applied by HybridSearch. */
  retrieval?: {
    denseWeight?: number;
    sparseWeight?: number;
    fusionK?: number;
    overfetchFactor?: number;
    embeddingConcurrency?: number;
  };

  /**
   * Cross-encoder reranking. Omitted means retrieval order is used directly.
   *
   * Reranking is normally the most expensive step in the pipeline, so leaving it out is
   * a legitimate cost decision rather than a degraded configuration.
   */
  reranker?: RerankerConfig;

  /** Citation extraction. Requires its own embed function, which may differ from
   *  the retrieval embedder if a cheaper model is adequate for attribution. */
  citations?: CitationConfig;

  /** Grounding verification. */
  hallucination?: HallucinationConfig;

  /** Query rewriting. Each strategy costs one extra inference, so all are opt-in. */
  queryRewriting?: {
    maxVariations?: number;
    historyWindow?: number;
  };

  logger?: Logger;
}

export interface IngestOptions {
  /** Documents processed concurrently. Defaults to 10. */
  batchSize?: number;
}

export interface QueryOptions {
  /** Candidates retrieved before reranking. Defaults to 20. */
  topK?: number;
  /** Chunks passed to generation after reranking. Defaults to 5. */
  contextSize?: number;
  /** Generate query paraphrases. Costs one inference. */
  multiQuery?: boolean;
  /** Split a multi-hop question into sub-questions. Costs one inference. */
  decompose?: boolean;
  systemPrompt?: string;
  /** Conversation history, for follow-up resolution. */
  history?: readonly ConversationTurn[];
}

export interface QueryResult {
  answer: string;
  citations: Citation[];
  /** Present only when citations are configured. */
  bibliography?: CitedResponse['bibliography'];
  /**
   * Grounding verdict, or null when no scorer is configured.
   *
   * A discriminated outcome rather than a bare number, because "there was nothing to
   * verify" and "everything was fabricated" are opposite situations that a single 0
   * would conflate.
   */
  grounding: ScoreOutcome | null;
  /** Chunks actually sent to the model. */
  chunks: Chunk[];
  metadata: {
    queriesUsed: string[];
    rewriteStrategy: RewriteStrategy;
    chunksRetrieved: number;
    chunksAfterReranking: number;
    /** Inference calls spent, so cost is attributable per query. */
    inferenceCalls: number;
    /** Set when a retriever failed and the query continued without it. */
    degraded?: { retriever: 'dense' | 'sparse'; reason: string };
    latencyMs: number;
  };
}
