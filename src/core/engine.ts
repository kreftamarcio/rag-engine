import { z } from 'zod';
import type { RAGConfig, QueryOptions, QueryResult, IngestOptions } from './config.js';
import { HybridSearch } from '../retrieval/hybrid-search.js';
import { Reranker } from '../retrieval/reranker.js';
import { QueryRewriter } from '../retrieval/query-rewriter.js';
import { PromptBuilder } from '../generation/prompt-builder.js';
import { CitationLinker } from '../generation/citation-linker.js';
import { HallucinationScorer } from '../generation/hallucination-scorer.js';
import { ChunkingPipeline } from '../ingestion/pipeline.js';
import { Logger } from '../utils/logger.js';

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
  source: string;
  span: string;
  relevance: number;
  chunkId: string;
}

export class RAGEngine {
  private readonly hybridSearch: HybridSearch;
  private readonly reranker: Reranker | null;
  private readonly queryRewriter: QueryRewriter;
  private readonly promptBuilder: PromptBuilder;
  private readonly citationLinker: CitationLinker;
  private readonly hallucinationScorer: HallucinationScorer | null;
  private readonly chunkingPipeline: ChunkingPipeline;
  private readonly logger: Logger;

  constructor(private readonly config: RAGConfig) {
    this.logger = new Logger('rag-engine');

    this.hybridSearch = new HybridSearch({
      vectorStore: config.vectorStore,
      embedding: config.embedding,
      sparseWeight: 0.3,
      denseWeight: 0.7,
      fusionK: 60,
    });

    this.reranker = config.reranker
      ? new Reranker(config.reranker)
      : null;

    this.queryRewriter = new QueryRewriter({
      model: config.generation.model,
      provider: config.generation.provider,
    });

    this.promptBuilder = new PromptBuilder();
    this.citationLinker = new CitationLinker();

    this.hallucinationScorer = config.hallucination?.enabled
      ? new HallucinationScorer(config.hallucination)
      : null;

    this.chunkingPipeline = new ChunkingPipeline(config.chunking);

    this.logger.info('RAG Engine initialized', {
      vectorStore: config.vectorStore.provider,
      chunking: config.chunking.strategy,
      reranker: !!config.reranker,
      hallucination: !!config.hallucination?.enabled,
    });
  }

  /**
   * Ingest documents into the vector store.
   * Documents are chunked, embedded, and indexed.
   */
  async ingest(documents: Document[], options?: IngestOptions): Promise<{
    chunksCreated: number;
    documentsProcessed: number;
    errors: Array<{ source: string; error: string }>;
  }> {
    const results = {
      chunksCreated: 0,
      documentsProcessed: 0,
      errors: [] as Array<{ source: string; error: string }>,
    };

    const batchSize = options?.batchSize ?? 10;

    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);

      const batchResults = await Promise.allSettled(
        batch.map(async (doc) => {
          // Chunk the document
          const chunks = await this.chunkingPipeline.process(doc);

          // Generate embeddings
          const embeddings = await this.hybridSearch.embed(chunks.map(c => c.content));

          // Store in vector DB
          const enrichedChunks = chunks.map((chunk, idx) => ({
            ...chunk,
            embedding: embeddings[idx],
          }));

          await this.hybridSearch.upsert(enrichedChunks);

          return chunks.length;
        }),
      );

      for (let j = 0; j < batchResults.length; j++) {
        const result = batchResults[j]!;
        if (result.status === 'fulfilled') {
          results.chunksCreated += result.value;
          results.documentsProcessed++;
        } else {
          results.errors.push({
            source: batch[j]!.metadata.source,
            error: result.reason?.message ?? 'Unknown error',
          });
        }
      }
    }

    this.logger.info('Ingestion complete', results);
    return results;
  }

  /**
   * Query the RAG pipeline.
   * Rewrites query, retrieves chunks, reranks, generates answer with citations.
   */
  async query(question: string, options?: QueryOptions): Promise<QueryResult> {
    const startTime = performance.now();

    // Step 1: Query rewriting
    const queries = options?.multiQuery
      ? await this.queryRewriter.expand(question, 3)
      : [question];

    this.logger.debug('Query rewritten', { original: question, expanded: queries });

    // Step 2: Hybrid retrieval
    const topK = options?.topK ?? 20;
    const allChunks: Chunk[] = [];

    for (const q of queries) {
      const results = await this.hybridSearch.search(q, topK);
      allChunks.push(...results);
    }

    // Deduplicate by chunk ID
    const uniqueChunks = this.deduplicateChunks(allChunks);

    // Step 3: Reranking
    const rerankedChunks = this.reranker
      ? await this.reranker.rerank(question, uniqueChunks)
      : uniqueChunks.slice(0, options?.topK ?? 5);

    // Step 4: Build prompt with context
    const prompt = this.promptBuilder.build(question, rerankedChunks, {
      systemPrompt: options?.systemPrompt,
      citationFormat: 'inline',
    });

    // Step 5: Generate response
    const generatedAnswer = await this.generate(prompt);

    // Step 6: Extract citations
    const citations = this.citationLinker.extract(generatedAnswer, rerankedChunks);

    // Step 7: Hallucination scoring
    let groundingScore = 1.0;
    if (this.hallucinationScorer) {
      groundingScore = await this.hallucinationScorer.score(
        generatedAnswer,
        rerankedChunks,
      );
    }

    const latency = performance.now() - startTime;

    return {
      answer: generatedAnswer,
      citations,
      groundingScore,
      chunks: rerankedChunks,
      metadata: {
        queriesUsed: queries,
        chunksRetrieved: uniqueChunks.length,
        chunksAfterReranking: rerankedChunks.length,
        latencyMs: Math.round(latency),
      },
    };
  }

  /**
   * Create a conversational session with memory.
   */
  createSession(): RAGSession {
    return new RAGSession(this, this.queryRewriter);
  }

  private async generate(prompt: string): Promise<string> {
    // Provider-agnostic generation
    // Implementation delegates to configured LLM provider
    throw new Error('Generation not yet wired to provider');
  }

  private deduplicateChunks(chunks: Chunk[]): Chunk[] {
    const seen = new Set<string>();
    return chunks.filter(chunk => {
      if (seen.has(chunk.id)) return false;
      seen.add(chunk.id);
      return true;
    });
  }
}

class RAGSession {
  private history: Array<{ query: string; answer: string }> = [];

  constructor(
    private readonly engine: RAGEngine,
    private readonly rewriter: QueryRewriter,
  ) {}

  async query(question: string, options?: QueryOptions): Promise<QueryResult> {
    // Rewrite query with conversation context
    const contextualQuery = this.history.length > 0
      ? await this.rewriter.contextualize(question, this.history)
      : question;

    const result = await this.engine.query(contextualQuery, options);

    this.history.push({ query: question, answer: result.answer });

    // Keep history bounded
    if (this.history.length > 10) {
      this.history = this.history.slice(-5);
    }

    return result;
  }

  clearHistory(): void {
    this.history = [];
  }
}
