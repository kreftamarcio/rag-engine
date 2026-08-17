import { HybridSearch } from '../retrieval/hybrid-search.js';
import { Reranker } from '../retrieval/reranker.js';
import { QueryRewriter } from '../retrieval/query-rewriter.js';
import { CitationExtractor } from '../generation/citation-extractor.js';
import { HallucinationScorer } from '../generation/hallucination-scorer.js';
import { RecursiveChunker } from '../ingestion/chunkers/recursive.chunker.js';
import { silentLogger } from './config.js';
import type {
  Chunk,
  Citation,
  Document,
  IngestOptions,
  Logger,
  QueryOptions,
  QueryResult,
  RAGConfig,
} from './config.js';
import type { RewriteResult, RewriteStrategy } from '../retrieval/query-rewriter.js';

// Re-exported so `import type { Chunk } from '../core/engine.js'` keeps resolving.
// Four modules already depend on that path, and changing them is outside this batch.
export type { Document, Chunk, Citation };

export interface IngestReport {
  chunksCreated: number;
  documentsProcessed: number;
  errors: Array<{ source: string; error: string }>;
}

/**
 * RAG pipeline orchestrator.
 *
 * Composes the components in this repository and nothing else. Every optional stage is
 * genuinely optional: without a reranker, retrieval order is used directly; without a
 * scorer, `grounding` is null rather than a fabricated 1.0.
 */
export class RAGEngine {
  private readonly hybridSearch: HybridSearch;
  private readonly reranker: Reranker | null;
  private readonly queryRewriter: QueryRewriter;
  private readonly citations: CitationExtractor | null;
  private readonly hallucination: HallucinationScorer | null;
  private readonly chunker: RecursiveChunker;
  private readonly logger: Logger;

  constructor(private readonly config: RAGConfig) {
    this.logger = config.logger ?? silentLogger;

    this.hybridSearch = new HybridSearch({
      embedding: config.embedding,
      vectorStore: config.vectorStore,
      ...(config.sparseIndex ? { sparseIndex: config.sparseIndex } : {}),
      ...(config.retrieval ?? {}),
    });

    this.reranker = config.reranker ? new Reranker(config.reranker) : null;

    // The rewriter always exists, because contextualize() is needed by any session even
    // when no rewrite strategy is requested. Construction is free; the inference is not,
    // and no inference happens unless a strategy is explicitly enabled.
    this.queryRewriter = new QueryRewriter({
      complete: (prompt) => config.generation.complete(prompt),
      ...(config.queryRewriting ?? {}),
    });

    this.citations = config.citations ? new CitationExtractor(config.citations) : null;

    this.hallucination = config.hallucination?.enabled
      ? new HallucinationScorer(config.hallucination)
      : null;

    this.chunker = new RecursiveChunker(config.chunking);

    this.logger.info('RAG engine initialized', {
      sparseRetrieval: config.sparseIndex !== undefined,
      reranking: config.reranker !== undefined,
      citations: config.citations !== undefined,
      groundingChecks: config.hallucination?.enabled === true,
    });
  }

  /**
   * Chunk, embed and index documents.
   *
   * Uses allSettled per batch so one malformed document does not discard the batch. A
   * rejected document is reported by source rather than counted as processed, because
   * silently dropping it would leave the index incomplete with no signal.
   */
  async ingest(documents: readonly Document[], options?: IngestOptions): Promise<IngestReport> {
    const report: IngestReport = { chunksCreated: 0, documentsProcessed: 0, errors: [] };
    const batchSize = Math.max(1, options?.batchSize ?? 10);

    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);

      const settled = await Promise.allSettled(
        batch.map(async (document) => {
          const chunks = this.toChunks(document);
          if (chunks.length === 0) return 0;

          const embeddings = await this.hybridSearch.embed(chunks.map((c) => c.content));

          await this.hybridSearch.upsert(
            chunks.map((chunk, index) => ({ ...chunk, embedding: embeddings[index]! })),
          );

          return chunks.length;
        }),
      );

      for (let j = 0; j < settled.length; j++) {
        const result = settled[j]!;

        if (result.status === 'fulfilled') {
          report.chunksCreated += result.value;
          report.documentsProcessed++;
        } else {
          report.errors.push({
            source: batch[j]!.metadata.source,
            error: result.reason instanceof Error ? result.reason.message : String(result.reason),
          });
        }
      }
    }

    this.logger.info('Ingestion complete', { ...report, errors: report.errors.length });
    return report;
  }

  /**
   * Retrieve, rerank, generate, attribute, verify.
   *
   * Inference calls are counted and returned. Query rewriting, generation and grounding
   * each cost at least one call, so a query that looks cheap can be three, and without
   * the count that is invisible until the invoice arrives.
   */
  async query(question: string, options?: QueryOptions): Promise<QueryResult> {
    const startedAt = performance.now();

    const rewrite = await this.rewriteQuery(question, options);
    let inferenceCalls = rewrite.inferenceCalls;

    const topK = options?.topK ?? 20;
    const contextSize = options?.contextSize ?? 5;

    const retrieved: Chunk[] = [];
    let degraded: QueryResult['metadata']['degraded'];

    for (const query of rewrite.queries) {
      const result = await this.hybridSearch.search(query, topK);
      retrieved.push(...result.chunks);

      // Retained from the first degraded retriever rather than the last, so the report
      // names the failure the caller is most likely to act on.
      if (result.degraded && !degraded) degraded = result.degraded;
    }

    const unique = this.deduplicate(retrieved);

    const context = this.reranker
      ? await this.rerank(question, unique, contextSize)
      : unique.slice(0, contextSize);

    // No context means retrieval found nothing. Generating anyway would produce an
    // answer with no grounding whatsoever, which is the exact failure this pipeline
    // exists to prevent, so it returns empty rather than inventing.
    if (context.length === 0) {
      return this.emptyResult(rewrite, inferenceCalls, unique.length, startedAt, degraded);
    }

    const answer = await this.config.generation.complete(
      this.buildPrompt(question, context, options?.systemPrompt),
    );
    inferenceCalls++;

    const attributed = await this.attribute(answer, context);

    const grounding = this.hallucination
      ? await this.hallucination.score(answer, context)
      : null;

    return {
      answer: attributed.annotatedAnswer,
      citations: attributed.citations,
      ...(attributed.bibliography ? { bibliography: attributed.bibliography } : {}),
      grounding,
      chunks: context,
      metadata: {
        queriesUsed: rewrite.queries,
        rewriteStrategy: rewrite.strategy,
        chunksRetrieved: unique.length,
        chunksAfterReranking: context.length,
        inferenceCalls,
        ...(degraded ? { degraded } : {}),
        latencyMs: Math.round(performance.now() - startedAt),
      },
    };
  }

  /** Conversational session with follow-up resolution. */
  createSession(): RAGSession {
    return new RAGSession(this);
  }

  /** Frees the tokenizer held by the chunker. */
  dispose(): void {
    this.chunker.dispose();
  }

  /**
   * Select a rewrite strategy and apply it.
   *
   * Contextualization takes precedence over decomposition: an unresolved pronoun makes
   * every downstream step operate on the wrong question, so resolving it first is not a
   * preference but an ordering requirement.
   */
  private async rewriteQuery(question: string, options?: QueryOptions): Promise<RewriteResult> {
    if (options?.history && options.history.length > 0) {
      return this.queryRewriter.contextualize(question, options.history);
    }
    if (options?.decompose) {
      return this.queryRewriter.decompose(question);
    }
    if (options?.multiQuery) {
      return this.queryRewriter.expand(question);
    }

    return { queries: [question], strategy: 'none', original: question, inferenceCalls: 0 };
  }

  /**
   * Rerank and map back to chunks.
   *
   * RankedDocument carries content but no chunk id, so the ranked results are mapped
   * back through `originalRank`, which indexes the candidate array that was passed in.
   * Matching on content instead would collapse duplicate passages and lose metadata.
   */
  private async rerank(question: string, candidates: Chunk[], contextSize: number): Promise<Chunk[]> {
    const { results, metrics } = await this.reranker!.rerank(
      question,
      candidates.map((chunk) => ({ content: chunk.content, metadata: chunk.metadata })),
    );

    // Displacement near zero means reranking agreed with retrieval and is not earning
    // its cost, which is worth surfacing since reranking is the most expensive stage.
    this.logger.debug('Reranked', {
      returned: metrics.candidatesReturned,
      filtered: metrics.candidatesFiltered,
      displacement: metrics.rankDisplacement,
    });

    return results
      .map((document) => candidates[document.originalRank])
      .filter((chunk): chunk is Chunk => chunk !== undefined)
      .slice(0, contextSize);
  }

  /**
   * Attribute sentences to chunks.
   *
   * When no citation extractor is configured the answer is returned unannotated with an
   * empty citation list. Fabricating citations from retrieval rank would attach
   * authority to sentences nothing verified.
   */
  private async attribute(
    answer: string,
    context: readonly Chunk[],
  ): Promise<{
    annotatedAnswer: string;
    citations: Citation[];
    bibliography?: Awaited<ReturnType<CitationExtractor['extract']>>['bibliography'];
  }> {
    if (!this.citations) {
      return { annotatedAnswer: answer, citations: [] };
    }

    const cited = await this.citations.extract(
      answer,
      context.map((chunk) => ({
        content: chunk.content,
        ...(typeof chunk.metadata.title === 'string' ? { title: chunk.metadata.title } : {}),
      })),
    );

    const citations: Citation[] = [];

    for (const citation of cited.citations) {
      for (const source of citation.sources) {
        const chunk = context[source.sourceIndex];
        if (!chunk) continue;

        citations.push({
          source: chunk.metadata.source,
          span: source.passage,
          relevance: source.similarity,
          chunkId: chunk.id,
          marker: source.marker,
        });
      }
    }

    if (cited.uncitedSentences.length > 0) {
      this.logger.debug('Sentences without a citation', {
        count: cited.uncitedSentences.length,
        coverage: cited.coverageRatio,
      });
    }

    return {
      annotatedAnswer: cited.annotatedText,
      citations,
      bibliography: cited.bibliography,
    };
  }

  /**
   * Assemble the generation prompt.
   *
   * Chunks are numbered so the model can reference them, and the instruction to admit
   * insufficient context is explicit: without it, a model handed weak context produces a
   * confident answer from parametric memory, which is precisely what a RAG pipeline is
   * supposed to prevent.
   */
  private buildPrompt(question: string, context: readonly Chunk[], systemPrompt?: string): string {
    const sources = context
      .map((chunk, index) => `[${index + 1}] (${chunk.metadata.source})\n${chunk.content}`)
      .join('\n\n');

    return [
      systemPrompt ?? 'Answer using only the provided sources.',
      '',
      'Rules:',
      '- Use only information present in the sources below.',
      '- If the sources do not contain the answer, say so explicitly.',
      '- Do not add facts from prior knowledge.',
      '',
      'Sources:',
      sources,
      '',
      `Question: ${question}`,
    ].join('\n');
  }

  /**
   * Convert a document into chunks.
   *
   * `totalChunks` is filled after chunking rather than during, because it is not known
   * until the split finishes, and a chunk claiming "1 of 1" mid-loop would be wrong for
   * every document that splits.
   */
  private toChunks(document: Document): Chunk[] {
    const pieces = this.chunker.chunk(document.content);

    return pieces.map((piece) => ({
      id: `${document.metadata.source}#${piece.chunkIndex}`,
      content: piece.content,
      metadata: {
        ...document.metadata,
        source: document.metadata.source,
        chunkIndex: piece.chunkIndex,
        totalChunks: pieces.length,
        startOffset: piece.startOffset,
        endOffset: piece.endOffset,
        tokenCount: piece.tokenCount,
      },
    }));
  }

  private deduplicate(chunks: readonly Chunk[]): Chunk[] {
    const seen = new Set<string>();

    return chunks.filter((chunk) => {
      if (seen.has(chunk.id)) return false;
      seen.add(chunk.id);
      return true;
    });
  }

  private emptyResult(
    rewrite: RewriteResult,
    inferenceCalls: number,
    retrieved: number,
    startedAt: number,
    degraded: QueryResult['metadata']['degraded'],
  ): QueryResult {
    this.logger.warn('Retrieval returned no context; not generating an answer');

    return {
      answer: '',
      citations: [],
      // Explicitly unverifiable rather than a grounding score of 0. There was nothing to
      // verify, which is a different statement from "the answer was fabricated".
      grounding: this.hallucination ? { verifiable: false, reason: 'no_sources' } : null,
      chunks: [],
      metadata: {
        queriesUsed: rewrite.queries,
        rewriteStrategy: rewrite.strategy,
        chunksRetrieved: retrieved,
        chunksAfterReranking: 0,
        inferenceCalls,
        ...(degraded ? { degraded } : {}),
        latencyMs: Math.round(performance.now() - startedAt),
      },
    };
  }
}

/**
 * Conversational session.
 *
 * Exported, unlike the previous version where the class was declared but never exported,
 * making the documented conversational API unreachable from the package.
 *
 * History is bounded because a pronoun almost always refers to something recent, and an
 * unbounded transcript grows the rewrite prompt quadratically for no gain in resolution.
 */
export class RAGSession {
  private history: Array<{ query: string; answer: string }> = [];

  constructor(
    private readonly engine: RAGEngine,
    private readonly maxTurns = 10,
  ) {}

  async query(question: string, options?: QueryOptions): Promise<QueryResult> {
    const result = await this.engine.query(question, { ...options, history: this.history });

    // Only successful turns are recorded. An empty answer would poison the next
    // rewrite, which resolves references against what was previously said.
    if (result.answer.length > 0) {
      this.history.push({ query: question, answer: result.answer });

      if (this.history.length > this.maxTurns) {
        this.history = this.history.slice(-Math.ceil(this.maxTurns / 2));
      }
    }

    return result;
  }

  getHistory(): ReadonlyArray<{ query: string; answer: string }> {
    return this.history;
  }

  clearHistory(): void {
    this.history = [];
  }
}

export type { RewriteStrategy };
