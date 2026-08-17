/**
 * @q1-digital/rag-engine
 *
 * RAG pipeline: hybrid retrieval, cross-encoder reranking, query rewriting, citation
 * attribution, and grounding verification.
 *
 * Adapters are injected, not configured by provider name. The library therefore reads no
 * environment variables of its own: credentials live in the adapters the caller builds.
 */

// ---------------------------------------------------------------------------
// Core
// ---------------------------------------------------------------------------

export { RAGEngine, RAGSession } from './core/engine.js';
export type { IngestReport } from './core/engine.js';

export { silentLogger } from './core/config.js';
export type {
  RAGConfig,
  QueryOptions,
  QueryResult,
  IngestOptions,
  Document,
  Chunk,
  Citation,
  GenerationAdapter,
  Logger,
} from './core/config.js';

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

export { RecursiveChunker } from './ingestion/chunkers/recursive.chunker.js';
export type {
  RecursiveChunkerConfig,
  TextChunk,
} from './ingestion/chunkers/recursive.chunker.js';

export { SemanticChunker } from './ingestion/chunkers/semantic.chunker.js';
export type {
  SemanticChunkerConfig,
  SemanticChunk,
} from './ingestion/chunkers/semantic.chunker.js';

// ---------------------------------------------------------------------------
// Retrieval
//
// The adapter interfaces are exported because a caller MUST implement them to construct
// the engine. Previously they were internal, so the primary class could not be built in
// typed code without redeclaring their shapes by hand.
// ---------------------------------------------------------------------------

export { HybridSearch } from './retrieval/hybrid-search.js';
export type {
  HybridSearchConfig,
  SearchResult,
  EmbeddingAdapter,
  VectorStoreAdapter,
  SparseIndexAdapter,
} from './retrieval/hybrid-search.js';

export { Reranker } from './retrieval/reranker.js';
export type {
  RerankerConfig,
  RankedDocument,
  RerankerMetrics,
} from './retrieval/reranker.js';

export { QueryRewriter } from './retrieval/query-rewriter.js';
export type {
  QueryRewriterConfig,
  RewriteResult,
  RewriteStrategy,
  ConversationTurn,
} from './retrieval/query-rewriter.js';

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

export { HallucinationScorer } from './generation/hallucination-scorer.js';
export type {
  HallucinationConfig,
  ScoreOutcome,
  Analysis,
  Claim,
  Verdict,
} from './generation/hallucination-scorer.js';

export { CitationExtractor } from './generation/citation-extractor.js';
export type {
  CitationConfig,
  Citation as SentenceCitation,
  CitedResponse,
} from './generation/citation-extractor.js';
