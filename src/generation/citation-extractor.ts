/**
 * Citation extractor: links generated sentences to the source passages that support
 * them.
 *
 * Attribution is not the same as grounding. The hallucination scorer asks "is this
 * claim supported at all"; this module asks "which passage supports it", which is what
 * a reader needs to verify an answer themselves.
 *
 * The threshold matters more than it appears. Too low and every sentence gets a
 * citation, including fabricated ones, which is worse than no citations because it
 * launders invention as sourced fact. A missing citation is visible; a wrong one is not.
 */

export interface CitationConfig {
  /** Embedding function. Must return one vector per input, in order. */
  embed: (texts: string[]) => Promise<number[][]>;
  /** Minimum cosine similarity to create a citation link. Defaults to 0.65. */
  citationThreshold?: number;
  /** Maximum sources cited per sentence. Defaults to 3. */
  maxCitationsPerSentence?: number;
  style?: 'numeric' | 'footnote';
  /** Texts per embedding request. Defaults to 256. */
  batchSize?: number;
  /** Concurrent embedding requests. Defaults to 3. */
  concurrency?: number;
  /** Characters per source passage before splitting. Defaults to 300. */
  passageLength?: number;
}

export interface Citation {
  sentenceIndex: number;
  sentence: string;
  sources: Array<{
    /** Index into the original sources array. */
    sourceIndex: number;
    /** Marker number as it appears inline. Matches the bibliography. */
    marker: number;
    passage: string;
    similarity: number;
    title?: string;
  }>;
}

export interface CitedResponse {
  /** Original answer with inline markers inserted. Formatting preserved. */
  annotatedText: string;
  citations: Citation[];
  bibliography: Array<{
    /** Marker number used inline. */
    marker: number;
    sourceIndex: number;
    title?: string;
    excerpt: string;
  }>;
  /** Fraction of factual sentences carrying at least one citation. */
  coverageRatio: number;
  /**
   * Sentences with no citation above threshold.
   *
   * Reported separately because an uncited sentence is not automatically a problem: a
   * transition or a restatement of the question has nothing to cite. Distinguishing
   * those from unsupported assertions is the caller's job, and they need the list.
   */
  uncitedSentences: Array<{ index: number; sentence: string; bestSimilarity: number }>;
}

interface SentenceSpan {
  text: string;
  /** Offset in the ORIGINAL answer, so markers can be inserted without reflowing. */
  start: number;
  end: number;
}

const DEFAULT_THRESHOLD = 0.65;
const DEFAULT_MAX_CITATIONS = 3;
const DEFAULT_BATCH_SIZE = 256;
const DEFAULT_CONCURRENCY = 3;
const DEFAULT_PASSAGE_LENGTH = 300;

export class CitationExtractor {
  private readonly embed: CitationConfig['embed'];
  private readonly threshold: number;
  private readonly maxCitations: number;
  private readonly batchSize: number;
  private readonly concurrency: number;
  private readonly passageLength: number;

  constructor(config: CitationConfig) {
    this.embed = config.embed;
    this.threshold = config.citationThreshold ?? DEFAULT_THRESHOLD;
    this.maxCitations = config.maxCitationsPerSentence ?? DEFAULT_MAX_CITATIONS;
    this.batchSize = Math.max(1, config.batchSize ?? DEFAULT_BATCH_SIZE);
    this.concurrency = Math.max(1, config.concurrency ?? DEFAULT_CONCURRENCY);
    this.passageLength = Math.max(50, config.passageLength ?? DEFAULT_PASSAGE_LENGTH);

    if (this.threshold < 0 || this.threshold > 1) {
      throw new RangeError(
        `citationThreshold must be within [0,1], received ${this.threshold}`,
      );
    }
  }

  async extract(
    answer: string,
    sources: ReadonlyArray<{ content: string; title?: string }>,
  ): Promise<CitedResponse> {
    const spans = this.splitSentences(answer);

    const passages = sources.flatMap((source, sourceIndex) =>
      this.splitPassages(source.content).map((passage) => ({
        passage,
        sourceIndex,
        ...(source.title !== undefined ? { title: source.title } : {}),
      })),
    );

    if (spans.length === 0 || passages.length === 0) {
      return {
        annotatedText: answer,
        citations: [],
        bibliography: [],
        coverageRatio: 0,
        uncitedSentences: spans.map((span, index) => ({
          index,
          sentence: span.text,
          bestSimilarity: 0,
        })),
      };
    }

    const texts = [...spans.map((s) => s.text), ...passages.map((p) => p.passage)];
    const embeddings = await this.embedInBatches(texts);

    const sentenceEmbeddings = embeddings.slice(0, spans.length);
    const passageEmbeddings = embeddings.slice(spans.length);

    // Matching happens first, numbering second. Markers cannot be assigned during
    // matching because a marker must reflect position in the FINAL bibliography, which
    // is not known until every sentence has been matched.
    const rawMatches: Array<
      Array<{ sourceIndex: number; passage: string; similarity: number; title?: string }>
    > = [];

    const bestSimilarities: number[] = [];

    for (let i = 0; i < spans.length; i++) {
      const sentenceEmbedding = sentenceEmbeddings[i]!;
      const matches: Array<{
        sourceIndex: number;
        passage: string;
        similarity: number;
        title?: string;
      }> = [];

      let best = 0;

      for (let j = 0; j < passages.length; j++) {
        const similarity = this.cosineSimilarity(sentenceEmbedding, passageEmbeddings[j]!);
        if (similarity > best) best = similarity;

        if (similarity >= this.threshold) {
          const candidate = passages[j]!;
          matches.push({
            sourceIndex: candidate.sourceIndex,
            passage: candidate.passage,
            similarity,
            ...(candidate.title !== undefined ? { title: candidate.title } : {}),
          });
        }
      }

      bestSimilarities[i] = best;

      // Deduplicated by source before truncation. Three passages from one document are
      // one citation, and counting them as three fills the per-sentence budget with a
      // single source while excluding others that genuinely support the sentence.
      const bySource = new Map<number, (typeof matches)[number]>();
      for (const match of matches.sort((a, b) => b.similarity - a.similarity)) {
        if (!bySource.has(match.sourceIndex)) bySource.set(match.sourceIndex, match);
      }

      rawMatches[i] = [...bySource.values()].slice(0, this.maxCitations);
    }

    // Markers assigned in order of first appearance, which is what a reader expects:
    // the first citation in the text is [1].
    const markerBySource = new Map<number, number>();
    for (const matches of rawMatches) {
      for (const match of matches ?? []) {
        if (!markerBySource.has(match.sourceIndex)) {
          markerBySource.set(match.sourceIndex, markerBySource.size + 1);
        }
      }
    }

    const citations: Citation[] = [];
    const uncitedSentences: CitedResponse['uncitedSentences'] = [];

    for (let i = 0; i < spans.length; i++) {
      const matches = rawMatches[i] ?? [];

      if (matches.length === 0) {
        uncitedSentences.push({
          index: i,
          sentence: spans[i]!.text,
          bestSimilarity: Math.round((bestSimilarities[i] ?? 0) * 1000) / 1000,
        });
        continue;
      }

      citations.push({
        sentenceIndex: i,
        sentence: spans[i]!.text,
        sources: matches.map((match) => ({
          ...match,
          marker: markerBySource.get(match.sourceIndex)!,
        })),
      });
    }

    const bibliography = [...markerBySource.entries()]
      .sort((a, b) => a[1] - b[1])
      .map(([sourceIndex, marker]) => {
        // Not a non-null assertion: sourceIndex is derived from the passage list, and a
        // defensive lookup costs nothing while an out-of-range index would otherwise
        // throw a TypeError with no context.
        const source = sources[sourceIndex];
        const content = source?.content ?? '';

        return {
          marker,
          sourceIndex,
          ...(source?.title !== undefined ? { title: source.title } : {}),
          excerpt: content.length > 200 ? `${content.slice(0, 200)}...` : content,
        };
      });

    return {
      annotatedText: this.annotate(answer, spans, citations),
      citations,
      bibliography,
      coverageRatio: spans.length > 0 ? citations.length / spans.length : 0,
      uncitedSentences,
    };
  }

  /**
   * Insert markers into the original text at recorded offsets.
   *
   * The previous implementation joined the sentence array with a single space, which
   * destroyed every newline, paragraph break, list marker and code block in the answer.
   * A correctly formatted response came back as one wall of text.
   *
   * Insertions are applied from the END backwards, so earlier offsets stay valid as the
   * string grows.
   */
  private annotate(
    original: string,
    spans: readonly SentenceSpan[],
    citations: readonly Citation[],
  ): string {
    const byIndex = new Map(citations.map((c) => [c.sentenceIndex, c]));

    const insertions: Array<{ at: number; text: string }> = [];

    for (let i = 0; i < spans.length; i++) {
      const citation = byIndex.get(i);
      if (!citation) continue;

      const markers = [...new Set(citation.sources.map((s) => s.marker))]
        .sort((a, b) => a - b)
        .map((m) => `[${m}]`)
        .join('');

      insertions.push({ at: spans[i]!.end, text: ` ${markers}` });
    }

    let result = original;
    for (const insertion of insertions.sort((a, b) => b.at - a.at)) {
      result = result.slice(0, insertion.at) + insertion.text + result.slice(insertion.at);
    }

    return result;
  }

  /**
   * Split into sentences while recording offsets in the original string.
   *
   * Offsets are what make formatting-preserving annotation possible. Abbreviations are
   * masked before splitting so "e.g." does not produce a fragment that matches nothing
   * and dilutes the coverage ratio.
   */
  private splitSentences(text: string): SentenceSpan[] {
    const masked = text.replace(
      /\b(e\.g|i\.e|etc|vs|approx|no|fig|al)\./gi,
      (match) => match.replace('.', '\u0001'),
    );

    const spans: SentenceSpan[] = [];
    const boundary = /(?<=[.!?])\s+/g;

    let start = 0;
    let match: RegExpExecArray | null;

    while ((match = boundary.exec(masked)) !== null) {
      const end = match.index + 1;
      this.pushSpan(spans, text, start, end);
      start = boundary.lastIndex;
    }

    this.pushSpan(spans, text, start, text.length);

    return spans;
  }

  /**
   * Record a span, trimming surrounding whitespace from the recorded bounds.
   *
   * No length floor. The previous 5-character filter dropped short sentences from
   * citation, and because the output was rebuilt from the filtered array they vanished
   * from the answer entirely.
   */
  private pushSpan(spans: SentenceSpan[], original: string, start: number, end: number): void {
    let from = start;
    let to = end;

    while (from < to && /\s/.test(original[from]!)) from++;
    while (to > from && /\s/.test(original[to - 1]!)) to--;

    if (to <= from) return;

    spans.push({ text: original.slice(from, to), start: from, end: to });
  }

  private splitPassages(text: string): string[] {
    const sentences = text.split(/(?<=[.!?])\s+/);
    const passages: string[] = [];
    let current = '';

    for (const sentence of sentences) {
      if (current && (current.length + sentence.length + 1) > this.passageLength) {
        passages.push(current.trim());
        current = sentence;
      } else {
        current += (current ? ' ' : '') + sentence;
      }
    }

    if (current.trim()) passages.push(current.trim());

    return passages;
  }

  /**
   * Embed in bounded batches with bounded concurrency.
   *
   * A long answer over ten documents produces hundreds of texts, which in a single
   * request exceeds a provider's input limit and fails after the whole payload has been
   * paid for. Order is preserved by writing into a pre-sized array, since batches
   * complete out of order.
   */
  private async embedInBatches(texts: string[]): Promise<number[][]> {
    const batches: Array<{ start: number; texts: string[] }> = [];

    for (let i = 0; i < texts.length; i += this.batchSize) {
      batches.push({ start: i, texts: texts.slice(i, i + this.batchSize) });
    }

    const results = new Array<number[]>(texts.length);
    let cursor = 0;

    const worker = async (): Promise<void> => {
      for (;;) {
        const index = cursor++;
        const batch = batches[index];
        if (!batch) return;

        const embeddings = await this.embed(batch.texts);

        // A count mismatch misaligns every subsequent vector, attributing each sentence
        // to the wrong passage. That produces confident wrong citations rather than a
        // visible failure, so it is a hard error.
        if (embeddings.length !== batch.texts.length) {
          throw new Error(
            `Embedder returned ${embeddings.length} vectors for ${batch.texts.length} ` +
              'inputs. Continuing would attribute sentences to the wrong passages.',
          );
        }

        for (let j = 0; j < embeddings.length; j++) {
          results[batch.start + j] = embeddings[j]!;
        }
      }
    };

    await Promise.all(
      Array.from({ length: Math.min(this.concurrency, batches.length) }, worker),
    );

    return results;
  }

  /**
   * Cosine similarity.
   *
   * Dimensionality is checked rather than assumed. The previous loop bound came from the
   * first vector alone, so mismatched dimensions silently produced a partial dot product
   * and a plausible-looking wrong similarity.
   */
  private cosineSimilarity(a: readonly number[], b: readonly number[]): number {
    if (a.length !== b.length) {
      throw new Error(
        `Embedding dimensionality mismatch: ${a.length} vs ${b.length}. Comparing them ` +
          'would yield a partial dot product and a plausible but meaningless similarity.',
      );
    }

    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      const x = a[i]!;
      const y = b[i]!;
      dot += x * y;
      normA += x * x;
      normB += y * y;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dot / denominator;
  }
}
