/**
 * Query rewriting.
 *
 * The question a user asks is rarely the question that retrieves well. Three
 * distinct failure modes need three distinct rewrites:
 *
 *   Vocabulary mismatch  A user writes "login is broken", the documentation says
 *                        "authentication failure". Dense retrieval narrows the gap
 *                        but does not close it, and sparse retrieval misses entirely.
 *                        -> expansion
 *
 *   Multi-hop question   "Compare REST and GraphQL auth" needs both sides retrieved
 *                        separately. A single embedding lands between the two topics
 *                        and retrieves neither well.
 *                        -> decomposition
 *
 *   Follow-up turn       "How do I increase it?" retrieves nothing, because the
 *                        referent lives in the previous turn.
 *                        -> contextualization
 *
 * Every rewrite costs an inference, so each is opt-in rather than always on. A
 * pipeline that rewrites unconditionally pays for three extra calls on a question
 * that needed none.
 */

export interface QueryRewriterConfig {
  /** Provider-agnostic completion function. Injected, never constructed here. */
  complete: (prompt: string) => Promise<string>;
  /** Cap on generated variations, so a verbose model cannot inflate cost. */
  maxVariations?: number;
  /** Turns of history considered when contextualizing. Defaults to 4. */
  historyWindow?: number;
}

export interface RewriteResult {
  /** Queries to retrieve with. Always contains at least the original. */
  queries: string[];
  strategy: RewriteStrategy;
  /** The input, retained so a trace can show what changed. */
  original: string;
  /** Inference calls consumed. Cost attribution needs this. */
  inferenceCalls: number;
  /** Set when a rewrite was attempted and rejected. */
  fallbackReason?: string;
}

export type RewriteStrategy =
  | 'none'
  | 'expansion'
  | 'decomposition'
  | 'hyde'
  | 'contextualization';

export interface ConversationTurn {
  query: string;
  answer: string;
}

const DEFAULT_MAX_VARIATIONS = 3;
const DEFAULT_HISTORY_WINDOW = 4;

/** Below this, a "query" is not specific enough to rewrite usefully. */
const MIN_REWRITABLE_LENGTH = 8;

export class QueryRewriter {
  private readonly complete: (prompt: string) => Promise<string>;
  private readonly maxVariations: number;
  private readonly historyWindow: number;

  constructor(config: QueryRewriterConfig) {
    this.complete = config.complete;
    this.maxVariations = Math.max(1, config.maxVariations ?? DEFAULT_MAX_VARIATIONS);
    this.historyWindow = Math.max(1, config.historyWindow ?? DEFAULT_HISTORY_WINDOW);
  }

  /**
   * Generate paraphrases that use different vocabulary for the same intent.
   *
   * The original is always kept and always first. A rewrite can lose a term that
   * mattered, and dropping the user's own phrasing means a query that would have
   * worked is replaced by one that might not.
   */
  async expand(query: string): Promise<RewriteResult> {
    if (query.trim().length < MIN_REWRITABLE_LENGTH) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 0,
        fallbackReason: 'Query too short to paraphrase meaningfully',
      };
    }

    const prompt = [
      'Rewrite the following search query in different words, preserving its exact intent.',
      `Produce at most ${this.maxVariations} variations, one per line, no numbering.`,
      'Use synonyms and alternative technical phrasing a document might have used.',
      'Do not answer the query. Do not add explanation.',
      '',
      `Query: ${query}`,
    ].join('\n');

    const raw = await this.complete(prompt);
    const variations = this.parseLines(raw)
      .filter((v) => v.toLowerCase() !== query.toLowerCase())
      .slice(0, this.maxVariations);

    if (variations.length === 0) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 1,
        fallbackReason: 'Model returned no usable variations',
      };
    }

    return {
      queries: [query, ...variations],
      strategy: 'expansion',
      original: query,
      inferenceCalls: 1,
    };
  }

  /**
   * Split a multi-hop question into independently retrievable sub-questions.
   *
   * The original is deliberately EXCLUDED here, unlike expansion. A compound query
   * embeds to a point between its topics and retrieves neither well, so keeping it
   * would spend a retrieval slot on the query that motivated the decomposition.
   */
  async decompose(query: string): Promise<RewriteResult> {
    const prompt = [
      'Break the following question into independent sub-questions, each answerable',
      'on its own from a single document.',
      `Produce at most ${this.maxVariations} sub-questions, one per line, no numbering.`,
      'If the question is already atomic, return it unchanged as a single line.',
      'Do not answer them.',
      '',
      `Question: ${query}`,
    ].join('\n');

    const raw = await this.complete(prompt);
    const parts = this.parseLines(raw).slice(0, this.maxVariations);

    // One part means the model judged the question atomic. Reporting 'none' rather
    // than 'decomposition' keeps the trace honest about what actually happened.
    if (parts.length <= 1) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 1,
        fallbackReason: 'Question is already atomic',
      };
    }

    return {
      queries: parts,
      strategy: 'decomposition',
      original: query,
      inferenceCalls: 1,
    };
  }

  /**
   * HyDE: Hypothetical Document Embeddings.
   *
   * Generates a plausible ANSWER and embeds that instead of the question. It works
   * because a question and its answer live in different regions of embedding space:
   * documents are answers, so a hypothetical answer is nearer to them than the
   * question is.
   *
   * The trade-off is real and must not be hidden. HyDE hallucinates confidently, and
   * a wrong hypothesis retrieves wrong documents while looking like a normal result.
   * It is therefore right for conceptual questions where the SHAPE of an answer is
   * predictable, and wrong for factual lookup where a specific value is needed and a
   * fabricated one poisons retrieval.
   *
   * Reference: Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance
   * Labels" (2022).
   */
  async hyde(query: string, options: { keepOriginal?: boolean } = {}): Promise<RewriteResult> {
    const prompt = [
      'Write a short passage that would answer the following question, as if excerpted',
      'from technical documentation. Two or three sentences.',
      'Accuracy is not required: the passage is used only to locate similar real',
      'documents. Match the vocabulary and register documentation would use.',
      '',
      `Question: ${query}`,
    ].join('\n');

    const hypothetical = (await this.complete(prompt)).trim();

    if (hypothetical.length === 0) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 1,
        fallbackReason: 'Model produced an empty hypothetical document',
      };
    }

    // Keeping the original hedges the hallucination risk: if the hypothesis is
    // wrong, the literal question still retrieves something relevant.
    return {
      queries: options.keepOriginal === false ? [hypothetical] : [query, hypothetical],
      strategy: 'hyde',
      original: query,
      inferenceCalls: 1,
    };
  }

  /**
   * Resolve a follow-up into a self-contained query.
   *
   * Only the last N turns are used. Full history grows the prompt without improving
   * resolution, because a pronoun almost always refers to something recent, and
   * paying quadratically for context that does not help is the cost defect that
   * hides best in a conversational pipeline.
   */
  async contextualize(query: string, history: readonly ConversationTurn[]): Promise<RewriteResult> {
    if (history.length === 0) {
      return { queries: [query], strategy: 'none', original: query, inferenceCalls: 0 };
    }

    // Cheap check first. A query with no referring expression needs no inference,
    // and most do not.
    if (!this.looksLikeFollowUp(query)) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 0,
        fallbackReason: 'No referring expression detected',
      };
    }

    const recent = history.slice(-this.historyWindow);
    const transcript = recent
      .map((turn, i) => `Turn ${i + 1}\nUser: ${turn.query}\nAssistant: ${turn.answer}`)
      .join('\n\n');

    const prompt = [
      'Rewrite the final user question so it is fully self-contained, resolving every',
      'pronoun and implicit reference using the conversation above.',
      'Return only the rewritten question.',
      'If it is already self-contained, return it unchanged.',
      '',
      transcript,
      '',
      `Final question: ${query}`,
    ].join('\n');

    const rewritten = (await this.complete(prompt)).trim();

    // A rewrite that shrinks the query has almost certainly dropped information
    // rather than resolved a reference, so the original is kept.
    if (rewritten.length === 0 || rewritten.length < query.length) {
      return {
        queries: [query],
        strategy: 'none',
        original: query,
        inferenceCalls: 1,
        fallbackReason: 'Rewrite was shorter than the original, suggesting lost information',
      };
    }

    return {
      queries: [rewritten],
      strategy: 'contextualization',
      original: query,
      inferenceCalls: 1,
    };
  }

  /**
   * Pick a strategy from the query's shape, then apply it.
   *
   * Heuristic and deliberately conservative: it returns 'none' unless there is a
   * positive signal, because the default of not rewriting costs nothing and a
   * misapplied rewrite costs an inference plus retrieval quality.
   */
  async auto(
    query: string,
    options: { history?: readonly ConversationTurn[]; allowHyde?: boolean } = {},
  ): Promise<RewriteResult> {
    if (options.history && options.history.length > 0 && this.looksLikeFollowUp(query)) {
      return this.contextualize(query, options.history);
    }

    if (this.looksMultiHop(query)) {
      return this.decompose(query);
    }

    // Gated: HyDE is only appropriate for conceptual questions, and the caller knows
    // whether its corpus is conceptual or factual. Enabling it by default would
    // silently degrade factual lookup.
    if (options.allowHyde && this.looksConceptual(query)) {
      return this.hyde(query);
    }

    return { queries: [query], strategy: 'none', original: query, inferenceCalls: 0 };
  }

  /**
   * Merge results from several queries, deduplicating by id.
   *
   * Position within each result list is preserved as a rank, because the caller
   * fuses on rank (RRF) and a merged list that discards per-query ordering destroys
   * the input that fusion needs.
   */
  static mergeResults<T extends { id: string }>(
    resultsPerQuery: ReadonlyArray<readonly T[]>,
  ): Array<{ item: T; ranks: number[]; queryCount: number }> {
    const merged = new Map<string, { item: T; ranks: number[] }>();

    for (const results of resultsPerQuery) {
      for (let rank = 0; rank < results.length; rank++) {
        const item = results[rank]!;
        const existing = merged.get(item.id);

        if (existing) {
          existing.ranks.push(rank + 1);
        } else {
          merged.set(item.id, { item, ranks: [rank + 1] });
        }
      }
    }

    return [...merged.values()]
      .map((entry) => ({ ...entry, queryCount: entry.ranks.length }))
      // Documents retrieved by more queries first: agreement across paraphrases is
      // a stronger relevance signal than a single high rank.
      .sort((a, b) => {
        if (b.queryCount !== a.queryCount) return b.queryCount - a.queryCount;
        return Math.min(...a.ranks) - Math.min(...b.ranks);
      });
  }

  /** Pronouns and elliptical openers that cannot resolve without prior context. */
  private looksLikeFollowUp(query: string): boolean {
    const lower = query.toLowerCase().trim();

    const referring = [
      /\b(it|that|this|those|these|they|them)\b/,
      /^(and|but|so|also|what about|how about|why not)\b/,
      /\b(the same|instead|again|the above|the former|the latter)\b/,
    ];

    return referring.some((pattern) => pattern.test(lower));
  }

  private looksMultiHop(query: string): boolean {
    const lower = query.toLowerCase();

    const comparative = [
      /\b(compare|versus|vs\.?|difference between|better than)\b/,
      /\b(both|either)\b.*\band\b/,
      /\band then\b/,
    ];

    // Two question marks means two questions, regardless of phrasing.
    if ((query.match(/\?/g) ?? []).length > 1) return true;

    return comparative.some((pattern) => pattern.test(lower));
  }

  private looksConceptual(query: string): boolean {
    const lower = query.toLowerCase();

    // "How does X work" has a predictable answer shape, which is what HyDE needs.
    const conceptual = [/^(how does|how do|why does|why do|what is the difference)/, /\bexplain\b/, /\bconcept\b/];

    // An identifier, code, version or error string means a specific value is wanted,
    // and a fabricated one would retrieve the wrong document.
    const factual = [/\b[A-Z]{2,}-?\d+\b/, /\bv?\d+\.\d+/, /\berror\b/, /\bcode\b/, /[`_]/];

    if (factual.some((pattern) => pattern.test(query))) return false;
    return conceptual.some((pattern) => pattern.test(lower));
  }

  /**
   * Split model output into clean lines.
   *
   * Strips list markers and quotes, because models add them despite instructions and
   * a leading "1. " becomes part of the embedded text if left in place.
   */
  private parseLines(raw: string): string[] {
    return raw
      .split('\n')
      .map((line) =>
        line
          .trim()
          .replace(/^[-*\u2022]\s*/, '')
          .replace(/^\d+[.)]\s*/, '')
          .replace(/^["']|["']$/g, '')
          .trim(),
      )
      .filter((line) => line.length >= MIN_REWRITABLE_LENGTH);
  }
}
