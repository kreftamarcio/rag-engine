# rag-engine

> Production-grade Retrieval-Augmented Generation pipeline: hybrid search (dense + sparse), multi-strategy chunking, cross-encoder reranking, citation extraction, and hallucination scoring.

[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-blue.svg)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-20+-green.svg)](https://nodejs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE                        │
│                                                                 │
│  Document → Parser → Chunker → Embedder → Vector Store          │
│     │                  │                        │                │
│     │         ┌────────▼────────┐     ┌────────▼────────┐  │
│     │         │ Metadata Store │     │ Sparse Index   │  │
│     │         │ (PostgreSQL)  │     │ (BM25/SPLADE) │  │
│     │         └─────────────────┘     └────────────────┘  │
│     ▼                                                          │
│  ┌────────────────┐                                              │
│  │ Source Registry │                                              │
│  └────────────────┘                                              │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       RETRIEVAL PIPELINE                        │
│                                                                 │
│  Query → Rewriter → Hybrid Search → Reranker → Context Builder  │
│                         │                 │                      │
│                ┌────────▼───────┐  ┌─────▼────────────┐     │
│                │ Dense + Sparse │  │ Cross-Encoder  │     │
│                │ Fusion (RRF)   │  │ (ms-marco)     │     │
│                └────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                       GENERATION PIPELINE                       │
│                                                                 │
│  Context → Prompt Builder → LLM → Citation Linker → Validator   │
│                                        │              │         │
│                               ┌────────▼─────┐  ┌─────▼─────┐  │
│                               │ Source Map  │  │Hallucinate│  │
│                               │ Attribution │  │ Detector  │  │
│                               └──────────────┘  └───────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### Chunking Strategies

| Strategy | Use Case | Avg Chunk Size |
|----------|----------|----------------|
| `recursive` | General text, articles | 512 tokens |
| `semantic` | Technical docs, code | Variable (by topic) |
| `sliding-window` | Dense information | 256 tokens, 64 overlap |
| `sentence` | Q&A, FAQs | 3-5 sentences |
| `parent-child` | Hierarchical docs | Parent: 2048, Child: 256 |

### Retrieval Fusion

Hybrid search combines dense (vector similarity) and sparse (BM25/SPLADE) retrieval using Reciprocal Rank Fusion (RRF):

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

Where `k` is a constant (default: 60) and `rank_i(d)` is the rank of document `d` in the i-th retrieval system.

### Hallucination Scoring

Every generated response receives a grounding score:
- **Claim extraction**: Split response into atomic claims
- **Entailment check**: Verify each claim against source chunks
- **Score**: Ratio of supported claims to total claims
- **Threshold**: Responses below 0.85 are flagged for review

## Installation

```bash
npm install @q1-digital/rag-engine
```

## Quick Start

```typescript
import { RAGEngine } from '@q1-digital/rag-engine';

const rag = new RAGEngine({
  vectorStore: {
    provider: 'qdrant',
    url: process.env.QDRANT_URL!,
    collection: 'documents',
  },
  embedding: {
    provider: 'openai',
    model: 'text-embedding-3-large',
    dimensions: 3072,
  },
  chunking: {
    strategy: 'recursive',
    maxTokens: 512,
    overlap: 64,
  },
  reranker: {
    model: 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    topK: 5,
  },
  generation: {
    provider: 'anthropic',
    model: 'claude-sonnet-4-20250514',
    maxTokens: 2048,
  },
  hallucination: {
    enabled: true,
    threshold: 0.85,
    model: 'gpt-4o-mini', // lightweight judge
  },
});

// Ingest documents
await rag.ingest([
  { content: 'Full document text...', metadata: { source: 'docs/api.md', title: 'API Reference' } },
  { content: 'Another document...', metadata: { source: 'docs/guide.md', title: 'User Guide' } },
]);

// Query with citations
const result = await rag.query('How do I authenticate with the API?');

console.log(result.answer);       // Generated answer
console.log(result.citations);    // [{source: 'docs/api.md', span: '...', relevance: 0.94}]
console.log(result.groundingScore); // 0.92
console.log(result.chunks);       // Retrieved chunks used for generation
```

## Advanced Usage

### Multi-Query Retrieval

```typescript
// Decompose complex queries into sub-queries for better recall
const result = await rag.query('Compare REST and GraphQL authentication methods', {
  multiQuery: true,       // Generates 3 query variations
  queryDecomposition: true, // Splits into sub-questions
});
```

### Conversational RAG

```typescript
const session = rag.createSession();

const r1 = await session.query('What is the rate limit?');
// Uses: "What is the rate limit?"

const r2 = await session.query('How do I increase it?');
// Rewrites to: "How do I increase the API rate limit?"
// Context from r1 is carried forward
```

### Custom Chunking Pipeline

```typescript
import { ChunkingPipeline, RecursiveChunker, MetadataEnricher } from '@q1-digital/rag-engine';

const pipeline = new ChunkingPipeline([
  new RecursiveChunker({ maxTokens: 512, overlap: 64 }),
  new MetadataEnricher({ extractEntities: true, extractTopics: true }),
  new ParentChildLinker({ parentSize: 2048 }),
]);

const chunks = await pipeline.process(document);
```

## Configuration

```typescript
interface RAGConfig {
  vectorStore: VectorStoreConfig;
  embedding: EmbeddingConfig;
  chunking: ChunkingConfig;
  reranker?: RerankerConfig;
  generation: GenerationConfig;
  hallucination?: HallucinationConfig;
  cache?: CacheConfig;
}

interface VectorStoreConfig {
  provider: 'qdrant' | 'weaviate' | 'pinecone' | 'pgvector';
  url: string;
  collection: string;
  apiKey?: string;
}

interface ChunkingConfig {
  strategy: 'recursive' | 'semantic' | 'sliding-window' | 'sentence' | 'parent-child';
  maxTokens: number;
  overlap?: number;
  separators?: string[];
}

interface RerankerConfig {
  model: string;
  topK: number;
  threshold?: number;  // Minimum score to include
  batchSize?: number;
}

interface HallucinationConfig {
  enabled: boolean;
  threshold: number;   // 0-1, minimum grounding score
  model: string;       // Judge model
  claimGranularity?: 'sentence' | 'atomic';  // How to split claims
}
```

## Project Structure

```
src/
├── core/
│   ├── engine.ts                # Main RAG orchestrator
│   ├── config.ts                # Configuration validation
│   └── session.ts               # Conversational session state
├── ingestion/
│   ├── pipeline.ts              # Ingestion orchestrator
│   ├── parsers/
│   │   ├── pdf.parser.ts        # PDF extraction
│   │   ├── markdown.parser.ts   # Markdown parsing
│   │   ├── html.parser.ts       # HTML to text
│   │   └── docx.parser.ts       # DOCX extraction
│   ├── chunkers/
│   │   ├── recursive.chunker.ts
│   │   ├── semantic.chunker.ts
│   │   ├── sliding-window.chunker.ts
│   │   └── parent-child.chunker.ts
│   └── enrichers/
│       ├── metadata.enricher.ts # Entity/topic extraction
│       └── hierarchy.enricher.ts # Parent-child linking
├── retrieval/
│   ├── hybrid-search.ts         # Dense + sparse fusion
│   ├── dense.retriever.ts       # Vector similarity search
│   ├── sparse.retriever.ts      # BM25/SPLADE retrieval
│   ├── reranker.ts              # Cross-encoder reranking
│   ├── query-rewriter.ts        # Query expansion/decomposition
│   └── fusion.ts                # Reciprocal Rank Fusion
├── generation/
│   ├── prompt-builder.ts        # Context-aware prompt assembly
│   ├── citation-linker.ts       # Source attribution
│   └── hallucination-scorer.ts  # Grounding verification
├── stores/
│   ├── vector/
│   │   ├── qdrant.store.ts
│   │   ├── weaviate.store.ts
│   │   ├── pinecone.store.ts
│   │   └── pgvector.store.ts
│   └── metadata/
│       └── postgres.store.ts
└── index.ts                     # Public API exports
```

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@5 | % of relevant docs in top 5 | > 0.85 |
| MRR | Mean Reciprocal Rank | > 0.75 |
| NDCG@10 | Normalized Discounted Cumulative Gain | > 0.80 |
| Grounding Score | % claims supported by sources | > 0.85 |
| Latency P95 | End-to-end query time | < 2s |
| Citation Precision | Correct attributions / total | > 0.90 |

## Benchmarks

| Dataset | Recall@5 | MRR | Latency P95 |
|---------|----------|-----|-------------|
| MS MARCO | 0.89 | 0.78 | 1.2s |
| Natural Questions | 0.86 | 0.74 | 1.4s |
| HotpotQA (multi-hop) | 0.81 | 0.69 | 1.8s |
| Custom (internal) | 0.92 | 0.83 | 0.9s |

## Roadmap

- [ ] Adaptive chunking (auto-select strategy per document type)
- [ ] Streaming generation with incremental citation
- [ ] Multi-modal RAG (images, tables, charts)
- [ ] Knowledge graph integration for entity-based retrieval
- [ ] Online learning from user feedback
- [ ] Distributed ingestion with job queue

## License

MIT
