import { encoding_for_model } from 'tiktoken';

export interface RecursiveChunkerConfig {
  maxTokens: number;
  overlap: number;
  separators?: string[];
  model?: string;
}

export interface TextChunk {
  content: string;
  startOffset: number;
  endOffset: number;
  tokenCount: number;
  chunkIndex: number;
}

/**
 * Recursive Character Text Splitter
 *
 * Splits text hierarchically using a list of separators,
 * trying to keep semantically related text together.
 *
 * Default separator hierarchy:
 *   1. "\n\n" (paragraphs)
 *   2. "\n" (lines)
 *   3. ". " (sentences)
 *   4. " " (words)
 *   5. "" (characters)
 *
 * Algorithm:
 *   1. Try splitting by the first separator
 *   2. If any resulting chunk > maxTokens, recursively split
 *      that chunk using the next separator in the hierarchy
 *   3. Merge small adjacent chunks up to maxTokens
 *   4. Add overlap between consecutive chunks
 */
export class RecursiveChunker {
  private readonly config: RecursiveChunkerConfig;
  private readonly separators: string[];
  private readonly encoder: ReturnType<typeof encoding_for_model>;

  constructor(config: RecursiveChunkerConfig) {
    this.config = config;
    this.separators = config.separators ?? [
      '\n\n',  // Paragraph breaks
      '\n',    // Line breaks
      '. ',    // Sentence boundaries
      ', ',    // Clause boundaries
      ' ',     // Words
      '',      // Characters (last resort)
    ];
    this.encoder = encoding_for_model('gpt-4o');
  }

  /**
   * Split text into chunks respecting token limits.
   */
  chunk(text: string): TextChunk[] {
    const rawChunks = this.splitRecursively(text, 0);
    const withOverlap = this.addOverlap(rawChunks, text);
    return withOverlap;
  }

  private splitRecursively(text: string, separatorIndex: number): TextChunk[] {
    if (separatorIndex >= this.separators.length) {
      // Last resort: hard truncate
      return this.hardSplit(text);
    }

    const separator = this.separators[separatorIndex]!;
    const parts = separator === ''
      ? [...text]  // Split by character
      : text.split(separator);

    const chunks: TextChunk[] = [];
    let currentChunk = '';
    let currentOffset = 0;

    for (const part of parts) {
      const candidate = currentChunk
        ? currentChunk + separator + part
        : part;

      const tokenCount = this.countTokens(candidate);

      if (tokenCount <= this.config.maxTokens) {
        currentChunk = candidate;
      } else {
        // Current chunk is ready
        if (currentChunk) {
          chunks.push(this.createChunk(currentChunk, currentOffset, chunks.length));
          currentOffset += currentChunk.length + separator.length;
        }

        // Check if the part itself exceeds limit
        if (this.countTokens(part) > this.config.maxTokens) {
          // Recursively split with next separator
          const subChunks = this.splitRecursively(part, separatorIndex + 1);
          for (const sub of subChunks) {
            chunks.push({
              ...sub,
              chunkIndex: chunks.length,
              startOffset: currentOffset + sub.startOffset,
              endOffset: currentOffset + sub.endOffset,
            });
          }
          currentOffset += part.length + separator.length;
          currentChunk = '';
        } else {
          currentChunk = part;
        }
      }
    }

    // Don't forget the last chunk
    if (currentChunk) {
      chunks.push(this.createChunk(currentChunk, currentOffset, chunks.length));
    }

    return chunks;
  }

  /**
   * Add overlap between consecutive chunks.
   * Takes the last N tokens from the previous chunk and prepends to current.
   */
  private addOverlap(chunks: TextChunk[], _originalText: string): TextChunk[] {
    if (this.config.overlap === 0 || chunks.length <= 1) {
      return chunks;
    }

    const result: TextChunk[] = [chunks[0]!];

    for (let i = 1; i < chunks.length; i++) {
      const prevChunk = chunks[i - 1]!;
      const currentChunk = chunks[i]!;

      // Get overlap text from end of previous chunk
      const overlapText = this.getLastNTokens(prevChunk.content, this.config.overlap);

      if (overlapText) {
        result.push({
          content: overlapText + ' ' + currentChunk.content,
          startOffset: prevChunk.endOffset - overlapText.length,
          endOffset: currentChunk.endOffset,
          tokenCount: this.countTokens(overlapText + ' ' + currentChunk.content),
          chunkIndex: i,
        });
      } else {
        result.push(currentChunk);
      }
    }

    return result;
  }

  private hardSplit(text: string): TextChunk[] {
    const chunks: TextChunk[] = [];
    const tokens = this.encoder.encode(text);

    for (let i = 0; i < tokens.length; i += this.config.maxTokens) {
      const slice = tokens.slice(i, i + this.config.maxTokens);
      const content = this.encoder.decode(slice);
      const decoded = new TextDecoder().decode(content);

      chunks.push({
        content: decoded,
        startOffset: i,
        endOffset: i + slice.length,
        tokenCount: slice.length,
        chunkIndex: chunks.length,
      });
    }

    return chunks;
  }

  private createChunk(content: string, offset: number, index: number): TextChunk {
    return {
      content: content.trim(),
      startOffset: offset,
      endOffset: offset + content.length,
      tokenCount: this.countTokens(content),
      chunkIndex: index,
    };
  }

  private countTokens(text: string): number {
    return this.encoder.encode(text).length;
  }

  private getLastNTokens(text: string, n: number): string {
    const tokens = this.encoder.encode(text);
    if (tokens.length <= n) return text;

    const slice = tokens.slice(-n);
    const decoded = this.encoder.decode(slice);
    return new TextDecoder().decode(decoded);
  }

  dispose(): void {
    this.encoder.free();
  }
}
