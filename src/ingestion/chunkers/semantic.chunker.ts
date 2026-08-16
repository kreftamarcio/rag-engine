/**
 * Semantic Chunker: splits text at natural topic boundaries
 * using embedding similarity between adjacent segments.
 *
 * Unlike fixed-size or recursive chunking, semantic chunking produces
 * chunks that respect topic coherence. This improves retrieval precision
 * because each chunk is semantically self-contained.
 *
 * Algorithm:
 *   1. Split text into sentences
 *   2. Create sliding window of N sentences
 *   3. Embed each window
 *   4. Calculate cosine similarity between adjacent windows
 *   5. Split at points where similarity drops below threshold
 *   6. Merge small segments that fall under minimum chunk size
 *
 * Trade-off: Requires embedding calls during ingestion (slower, costlier)
 * but produces significantly better retrieval quality.
 */

export interface SemanticChunkerConfig {
  /** Embedding function (injected for provider flexibility) */
  embed: (texts: string[]) => Promise<number[][]>;
  /** Similarity threshold below which to split (0-1, lower = more splits) */
  similarityThreshold?: number;
  /** Sliding window size in sentences */
  windowSize?: number;
  /** Minimum chunk size in characters */
  minChunkSize?: number;
  /** Maximum chunk size in characters */
  maxChunkSize?: number;
  /** Overlap sentences between chunks for context continuity */
  overlapSentences?: number;
}

export interface SemanticChunk {
  content: string;
  index: number;
  metadata: {
    startSentence: number;
    endSentence: number;
    avgSimilarity: number;
    charCount: number;
    tokenEstimate: number;
  };
}

const DEFAULT_CONFIG = {
  similarityThreshold: 0.45,
  windowSize: 3,
  minChunkSize: 200,
  maxChunkSize: 2000,
  overlapSentences: 1,
};

export class SemanticChunker {
  private readonly config: Required<Pick<SemanticChunkerConfig, keyof typeof DEFAULT_CONFIG>> & SemanticChunkerConfig;

  constructor(config: SemanticChunkerConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async chunk(text: string): Promise<SemanticChunk[]> {
    const sentences = this.splitSentences(text);

    if (sentences.length <= this.config.windowSize) {
      return [{
        content: text,
        index: 0,
        metadata: {
          startSentence: 0,
          endSentence: sentences.length - 1,
          avgSimilarity: 1,
          charCount: text.length,
          tokenEstimate: this.estimateTokens(text),
        },
      }];
    }

    // Create windows
    const windows = this.createWindows(sentences);

    // Embed all windows in batch (efficient)
    const embeddings = await this.config.embed(windows);

    // Find split points by comparing adjacent embeddings
    const splitPoints = this.findSplitPoints(embeddings);

    // Create chunks from split points
    const rawChunks = this.createChunksFromSplits(sentences, splitPoints);

    // Merge chunks that are too small
    const mergedChunks = this.mergeSmallChunks(rawChunks);

    // Split chunks that are too large
    return this.splitLargeChunks(mergedChunks);
  }

  private splitSentences(text: string): string[] {
    // Regex handles common sentence boundaries including abbreviations
    const sentenceRegex = /(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])|(?<=[.!?])\s*\n/g;
    const sentences = text.split(sentenceRegex).filter(s => s.trim().length > 0);
    return sentences.map(s => s.trim());
  }

  private createWindows(sentences: string[]): string[] {
    const windows: string[] = [];
    for (let i = 0; i <= sentences.length - this.config.windowSize; i++) {
      const window = sentences.slice(i, i + this.config.windowSize).join(' ');
      windows.push(window);
    }
    return windows;
  }

  private findSplitPoints(embeddings: number[][]): number[] {
    const similarities: number[] = [];
    const splitPoints: number[] = [];

    // Calculate cosine similarity between adjacent windows
    for (let i = 0; i < embeddings.length - 1; i++) {
      similarities.push(this.cosineSimilarity(embeddings[i]!, embeddings[i + 1]!));
    }

    // Find valleys (local minima below threshold)
    for (let i = 1; i < similarities.length - 1; i++) {
      const isValley = similarities[i]! < similarities[i - 1]! &&
                       similarities[i]! < similarities[i + 1]!;
      const belowThreshold = similarities[i]! < this.config.similarityThreshold;

      if (isValley && belowThreshold) {
        splitPoints.push(i + this.config.windowSize - 1);
      }
    }

    return splitPoints;
  }

  private createChunksFromSplits(sentences: string[], splitPoints: number[]): SemanticChunk[] {
    const chunks: SemanticChunk[] = [];
    let start = 0;

    const allSplits = [...splitPoints, sentences.length];

    for (let i = 0; i < allSplits.length; i++) {
      const end = allSplits[i]!;
      const chunkSentences = sentences.slice(start, end);
      const content = chunkSentences.join(' ');

      chunks.push({
        content,
        index: i,
        metadata: {
          startSentence: start,
          endSentence: end - 1,
          avgSimilarity: 0, // Will be calculated after
          charCount: content.length,
          tokenEstimate: this.estimateTokens(content),
        },
      });

      // Overlap: start next chunk a few sentences back
      start = Math.max(start + 1, end - this.config.overlapSentences);
    }

    return chunks;
  }

  private mergeSmallChunks(chunks: SemanticChunk[]): SemanticChunk[] {
    const merged: SemanticChunk[] = [];

    for (const chunk of chunks) {
      if (merged.length > 0 && chunk.content.length < this.config.minChunkSize) {
        // Merge with previous chunk
        const prev = merged[merged.length - 1]!;
        prev.content += ' ' + chunk.content;
        prev.metadata.endSentence = chunk.metadata.endSentence;
        prev.metadata.charCount = prev.content.length;
        prev.metadata.tokenEstimate = this.estimateTokens(prev.content);
      } else {
        merged.push({ ...chunk });
      }
    }

    return merged;
  }

  private splitLargeChunks(chunks: SemanticChunk[]): SemanticChunk[] {
    const result: SemanticChunk[] = [];
    let index = 0;

    for (const chunk of chunks) {
      if (chunk.content.length <= this.config.maxChunkSize) {
        result.push({ ...chunk, index: index++ });
      } else {
        // Split at sentence boundaries within maxChunkSize
        const sentences = this.splitSentences(chunk.content);
        let current = '';

        for (const sentence of sentences) {
          if ((current + ' ' + sentence).length > this.config.maxChunkSize && current.length > 0) {
            result.push({
              content: current.trim(),
              index: index++,
              metadata: {
                startSentence: chunk.metadata.startSentence,
                endSentence: chunk.metadata.endSentence,
                avgSimilarity: chunk.metadata.avgSimilarity,
                charCount: current.trim().length,
                tokenEstimate: this.estimateTokens(current.trim()),
              },
            });
            current = sentence;
          } else {
            current += (current ? ' ' : '') + sentence;
          }
        }

        if (current.trim()) {
          result.push({
            content: current.trim(),
            index: index++,
            metadata: {
              startSentence: chunk.metadata.startSentence,
              endSentence: chunk.metadata.endSentence,
              avgSimilarity: chunk.metadata.avgSimilarity,
              charCount: current.trim().length,
              tokenEstimate: this.estimateTokens(current.trim()),
            },
          });
        }
      }
    }

    return result;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i]! * b[i]!;
      normA += a[i]! * a[i]!;
      normB += b[i]! * b[i]!;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0) return 0;
    return dotProduct / denominator;
  }

  private estimateTokens(text: string): number {
    // Rough estimate: ~4 chars per token for English
    return Math.ceil(text.length / 4);
  }
}
