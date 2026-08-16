/**
 * Citation Extractor: links generated claims to source passages.
 *
 * When the LLM generates a response from retrieved context, this module
 * identifies which specific source passages support each sentence.
 * Enables inline citations like [1], [2] in the generated output.
 *
 * Algorithm:
 *   1. Split answer into sentences
 *   2. For each sentence, find best matching passage(s) via embedding similarity
 *   3. Only cite if similarity exceeds threshold (avoids false attributions)
 *   4. Format citations in configurable style
 */

export interface CitationConfig {
  /** Embedding function for similarity matching */
  embed: (texts: string[]) => Promise<number[][]>;
  /** Minimum similarity to create a citation link */
  citationThreshold?: number;
  /** Maximum sources to cite per sentence */
  maxCitationsPerSentence?: number;
  /** Citation style */
  style?: 'numeric' | 'author-year' | 'footnote';
}

export interface Citation {
  sentenceIndex: number;
  sentence: string;
  sources: Array<{
    index: number;
    passage: string;
    similarity: number;
    title?: string;
  }>;
}

export interface CitedResponse {
  /** Original answer with inline citations inserted */
  annotatedText: string;
  /** Structured citation data */
  citations: Citation[];
  /** Unique sources referenced */
  bibliography: Array<{
    index: number;
    title?: string;
    excerpt: string;
  }>;
  /** Percentage of sentences with at least one citation */
  coverageRatio: number;
}

const DEFAULTS = {
  citationThreshold: 0.65,
  maxCitationsPerSentence: 3,
  style: 'numeric' as const,
};

export class CitationExtractor {
  private readonly config: Required<Omit<CitationConfig, 'embed'>> & CitationConfig;

  constructor(config: CitationConfig) {
    this.config = { ...DEFAULTS, ...config };
  }

  /**
   * Extract citations from a generated answer given the source documents.
   */
  async extract(
    answer: string,
    sources: Array<{ content: string; title?: string }>,
  ): Promise<CitedResponse> {
    const sentences = this.splitSentences(answer);
    const sourcePassages = sources.flatMap((source, srcIdx) =>
      this.splitPassages(source.content).map(passage => ({
        passage,
        sourceIndex: srcIdx,
        title: source.title,
      })),
    );

    if (sentences.length === 0 || sourcePassages.length === 0) {
      return { annotatedText: answer, citations: [], bibliography: [], coverageRatio: 0 };
    }

    // Embed sentences and passages in batch
    const allTexts = [...sentences, ...sourcePassages.map(p => p.passage)];
    const embeddings = await this.config.embed(allTexts);

    const sentenceEmbeddings = embeddings.slice(0, sentences.length);
    const passageEmbeddings = embeddings.slice(sentences.length);

    // Find citations for each sentence
    const citations: Citation[] = [];
    const usedSources = new Set<number>();

    for (let i = 0; i < sentences.length; i++) {
      const sentenceEmb = sentenceEmbeddings[i]!;
      const matches: Citation['sources'] = [];

      for (let j = 0; j < sourcePassages.length; j++) {
        const sim = this.cosineSimilarity(sentenceEmb, passageEmbeddings[j]!);
        if (sim >= this.config.citationThreshold) {
          matches.push({
            index: sourcePassages[j]!.sourceIndex,
            passage: sourcePassages[j]!.passage,
            similarity: sim,
            title: sourcePassages[j]!.title,
          });
        }
      }

      // Sort by similarity and take top N
      matches.sort((a, b) => b.similarity - a.similarity);
      const topMatches = matches.slice(0, this.config.maxCitationsPerSentence);

      if (topMatches.length > 0) {
        citations.push({
          sentenceIndex: i,
          sentence: sentences[i]!,
          sources: topMatches,
        });
        topMatches.forEach(m => usedSources.add(m.index));
      }
    }

    // Build annotated text with inline citations
    const annotatedText = this.buildAnnotatedText(sentences, citations);

    // Build bibliography
    const bibliography = [...usedSources].sort().map(idx => ({
      index: idx,
      title: sources[idx]?.title,
      excerpt: sources[idx]!.content.slice(0, 200) + '...',
    }));

    const coverageRatio = citations.length / sentences.length;

    return { annotatedText, citations, bibliography, coverageRatio };
  }

  private buildAnnotatedText(sentences: string[], citations: Citation[]): string {
    const citationMap = new Map(citations.map(c => [c.sentenceIndex, c]));

    return sentences.map((sentence, idx) => {
      const citation = citationMap.get(idx);
      if (!citation) return sentence;

      const refs = [...new Set(citation.sources.map(s => s.index))]
        .map(i => `[${i + 1}]`)
        .join('');

      return `${sentence} ${refs}`;
    }).join(' ');
  }

  private splitSentences(text: string): string[] {
    return text
      .split(/(?<=[.!?])\s+/)
      .map(s => s.trim())
      .filter(s => s.length > 5);
  }

  private splitPassages(text: string, maxLength = 300): string[] {
    const sentences = text.split(/(?<=[.!?])\s+/);
    const passages: string[] = [];
    let current = '';

    for (const sentence of sentences) {
      if ((current + ' ' + sentence).length > maxLength && current) {
        passages.push(current.trim());
        current = sentence;
      } else {
        current += (current ? ' ' : '') + sentence;
      }
    }
    if (current.trim()) passages.push(current.trim());
    return passages;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i]! * b[i]!;
      normA += a[i]! * a[i]!;
      normB += b[i]! * b[i]!;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }
}
