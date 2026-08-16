/**
 * @q1-digital/rag-engine
 *
 * Production-grade RAG pipeline: hybrid search, multi-strategy chunking,
 * cross-encoder reranking, citation extraction, and hallucination scoring.
 */

// Core
export { RAGEngine } from './core/engine.js';

// Ingestion
export { RecursiveChunker } from './ingestion/chunkers/recursive.chunker.js';
export { SemanticChunker } from './ingestion/chunkers/semantic.chunker.js';
export type { SemanticChunkerConfig, SemanticChunk } from './ingestion/chunkers/semantic.chunker.js';

// Retrieval
export { HybridSearch } from './retrieval/hybrid-search.js';
export { Reranker } from './retrieval/reranker.js';
export type { RerankerConfig, RankedDocument, RerankerMetrics } from './retrieval/reranker.js';

// Generation
export { HallucinationScorer } from './generation/hallucination-scorer.js';
export { CitationExtractor } from './generation/citation-extractor.js';
export type { CitationConfig, Citation, CitedResponse } from './generation/citation-extractor.js';
