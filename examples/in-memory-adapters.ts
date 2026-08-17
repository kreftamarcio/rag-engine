/**
 * In-memory adapters for the examples.
 *
 * These exist so every example in this directory RUNS with no API key, no vector
 * database and no network. That is the difference between an example and a snippet: a
 * reader can execute it and compare the output against what the README claims.
 *
 * They are honest stand-ins, not production components:
 *
 *   - The embedder is LEXICAL. It builds a term-frequency vector with the hashing trick,
 *     so cosine similarity reflects vocabulary overlap and nothing more. It will not
 *     match "login is broken" to "authentication failure", which is precisely the
 *     weakness a real embedding model exists to fix.
 *   - The vector store is brute force. Correct, and O(n) per query.
 *   - The generator is EXTRACTIVE. It selects sentences from the provided context rather
 *     than writing new ones, so grounding scores come out high by construction. That is
 *     intentional: the example verifies the pipeline is wired correctly, not that a model
 *     is any good.
 *
 * The BM25 index, by contrast, is a genuine implementation of the formula the README
 * documents. There is no reason to fake that one.
 */

import type {
  EmbeddingAdapter,
  VectorStoreAdapter,
  SparseIndexAdapter,
  GenerationAdapter,
  Chunk,
} from '../src/index.js';

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

/**
 * Unicode-aware tokenizer, matching the one HybridSearch uses.
 *
 * An ASCII-only tokenizer would strip accents and silently break every non-English
 * example, and the symptom would look like poor retrieval rather than a bug.
 */
function tokenize(text: string): string[] {
  return text
    .normalize('NFC')
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .split(/\s+/)
    .filter((token) => token.length >= 2);
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/**
 * Term-frequency embedder using the hashing trick.
 *
 * Each token is hashed into one of `dimensions` buckets and its count incremented, then
 * the vector is L2-normalized so cosine similarity reduces to a dot product.
 *
 * Collisions are real and unmitigated: two unrelated terms landing in the same bucket
 * become indistinguishable. With 256 dimensions and a small corpus that is acceptable
 * for a demo, and it is the honest trade-off of the hashing trick.
 */
export class HashingEmbedder implements EmbeddingAdapter {
  constructor(private readonly dimensions = 256) {}

  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => this.vectorize(text));
  }

  private vectorize(text: string): number[] {
    const vector = new Array<number>(this.dimensions).fill(0);

    for (const token of tokenize(text)) {
      vector[this.hash(token) % this.dimensions]! += 1;
    }

    const magnitude = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));

    // An empty or fully-filtered text yields a zero vector. Returning it as-is is
    // correct: cosine against a zero vector is defined as 0 downstream, whereas dividing
    // here would produce NaN and poison every comparison.
    if (magnitude === 0) return vector;

    return vector.map((value) => value / magnitude);
  }

  /** FNV-1a. Deterministic across runs, which is what makes the examples reproducible. */
  private hash(token: string): number {
    let hash = 0x811c9dc5;

    for (let i = 0; i < token.length; i++) {
      hash ^= token.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193);
    }

    return hash >>> 0;
  }
}

// ---------------------------------------------------------------------------
// Vector store
// ---------------------------------------------------------------------------

/** Brute-force cosine over an in-memory map. Upsert is by chunk id. */
export class MemoryVectorStore implements VectorStoreAdapter {
  private readonly chunks = new Map<string, Chunk>();

  async upsert(chunks: Chunk[]): Promise<void> {
    for (const chunk of chunks) {
      if (!chunk.embedding) {
        throw new Error(
          `Chunk "${chunk.id}" has no embedding. Storing it would make it permanently ` +
            'unretrievable by dense search while appearing to be indexed.',
        );
      }
      this.chunks.set(chunk.id, chunk);
    }
  }

  async query(
    embedding: number[],
    topK: number,
  ): Promise<Array<{ chunk: Chunk; similarity: number }>> {
    return [...this.chunks.values()]
      .map((chunk) => ({
        chunk,
        similarity: cosine(embedding, chunk.embedding ?? []),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK);
  }

  get size(): number {
    return this.chunks.size;
  }
}

function cosine(a: readonly number[], b: readonly number[]): number {
  // Length mismatch means one side was embedded by a different model. Returning 0 is
  // safer than comparing a prefix, which would yield a plausible but meaningless number.
  if (a.length === 0 || a.length !== b.length) return 0;

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! * a[i]!;
    normB += b[i]! * b[i]!;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dot / denominator;
}

// ---------------------------------------------------------------------------
// Sparse index
// ---------------------------------------------------------------------------

interface IndexedDocument {
  chunk: Chunk;
  termFrequencies: Map<string, number>;
  length: number;
}

/**
 * Okapi BM25.
 *
 *   score(D,Q) = SUM IDF(qi) * ( f(qi,D) * (k1 + 1) )
 *                            / ( f(qi,D) + k1 * (1 - b + b * |D|/avgdl) )
 *
 *   IDF(qi) = ln( 1 + (N - n(qi) + 0.5) / (n(qi) + 0.5) )
 *
 * This is a real implementation, not a stub. The 0.5 smoothing in IDF and the outer
 * `1 +` matter: without them, a term appearing in more than half the corpus produces a
 * NEGATIVE idf, and a document containing it scores worse than one that does not.
 */
export class MemoryBM25Index implements SparseIndexAdapter {
  private readonly documents = new Map<string, IndexedDocument>();
  private readonly documentFrequency = new Map<string, number>();

  constructor(
    private readonly k1 = 1.2,
    private readonly b = 0.75,
  ) {}

  async index(chunks: Chunk[]): Promise<void> {
    for (const chunk of chunks) {
      // Re-indexing an existing id must first retract its contribution to document
      // frequency, otherwise idf drifts every time the same chunk is written again.
      const existing = this.documents.get(chunk.id);
      if (existing) {
        for (const term of existing.termFrequencies.keys()) {
          const count = (this.documentFrequency.get(term) ?? 1) - 1;
          if (count <= 0) this.documentFrequency.delete(term);
          else this.documentFrequency.set(term, count);
        }
      }

      const tokens = tokenize(chunk.content);
      const termFrequencies = new Map<string, number>();

      for (const token of tokens) {
        termFrequencies.set(token, (termFrequencies.get(token) ?? 0) + 1);
      }

      for (const term of termFrequencies.keys()) {
        this.documentFrequency.set(term, (this.documentFrequency.get(term) ?? 0) + 1);
      }

      this.documents.set(chunk.id, { chunk, termFrequencies, length: tokens.length });
    }
  }

  async query(
    tokens: string[],
    topK: number,
  ): Promise<Array<{ chunk: Chunk; bm25Score: number }>> {
    if (this.documents.size === 0) return [];

    const totalDocuments = this.documents.size;
    const averageLength =
      [...this.documents.values()].reduce((sum, doc) => sum + doc.length, 0) / totalDocuments;

    const scored: Array<{ chunk: Chunk; bm25Score: number }> = [];

    for (const document of this.documents.values()) {
      let score = 0;

      for (const term of tokens) {
        const frequency = document.termFrequencies.get(term);
        if (!frequency) continue;

        const documentsWithTerm = this.documentFrequency.get(term) ?? 0;

        const idf = Math.log(
          1 + (totalDocuments - documentsWithTerm + 0.5) / (documentsWithTerm + 0.5),
        );

        const denominator =
          frequency + this.k1 * (1 - this.b + (this.b * document.length) / averageLength);

        score += idf * ((frequency * (this.k1 + 1)) / denominator);
      }

      // Only documents that matched at least one term are candidates. Returning zeros
      // would fill the fusion input with noise that still consumes rank positions.
      if (score > 0) scored.push({ chunk: document.chunk, bm25Score: score });
    }

    return scored.sort((a, b) => b.bm25Score - a.bm25Score).slice(0, topK);
  }

  get size(): number {
    return this.documents.size;
  }
}

// ---------------------------------------------------------------------------
// Reranker scorer
// ---------------------------------------------------------------------------

/**
 * Term-overlap scorer standing in for a cross-encoder.
 *
 * Returns the fraction of query terms present in the document, which is containment
 * rather than similarity. A real cross-encoder models term interaction between query and
 * document; this cannot, and will rank a keyword-stuffed passage above a well-written one.
 */
export function lexicalRerankerScore(query: string, documents: string[]): Promise<number[]> {
  const queryTerms = new Set(tokenize(query));

  if (queryTerms.size === 0) return Promise.resolve(documents.map(() => 0));

  return Promise.resolve(
    documents.map((document) => {
      const documentTerms = new Set(tokenize(document));
      let found = 0;

      for (const term of queryTerms) {
        if (documentTerms.has(term)) found++;
      }

      return found / queryTerms.size;
    }),
  );
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

/**
 * Extractive generator.
 *
 * Selects the best-matching sentences from the prompt's Sources block instead of writing
 * new text. Two consequences worth stating:
 *
 *   1. Grounding scores are high BY CONSTRUCTION, because every sentence is copied from a
 *      source. The example therefore verifies wiring, not model quality.
 *   2. It answers query-rewriting prompts poorly, since those ask for paraphrases rather
 *      than extraction. Examples that enable multiQuery will show that plainly, which is
 *      more useful than hiding it.
 */
export class ExtractiveGenerator implements GenerationAdapter {
  constructor(private readonly maxSentences = 2) {}

  async complete(prompt: string): Promise<string> {
    const question = this.extractQuestion(prompt);
    const sources = this.extractSources(prompt);

    if (sources.length === 0 || question.length === 0) {
      // The explicit refusal matters: an extractive generator with nothing to extract
      // must say so rather than emit an empty string that reads as a valid answer.
      return 'The provided sources do not contain enough information to answer.';
    }

    const queryTerms = new Set(tokenize(question));

    const ranked = sources
      .flatMap((source) => this.splitSentences(source))
      .map((sentence) => {
        const terms = new Set(tokenize(sentence));
        let overlap = 0;
        for (const term of queryTerms) {
          if (terms.has(term)) overlap++;
        }
        return { sentence, overlap };
      })
      .filter((entry) => entry.overlap > 0)
      .sort((a, b) => b.overlap - a.overlap)
      .slice(0, this.maxSentences);

    if (ranked.length === 0) {
      return 'The provided sources do not contain enough information to answer.';
    }

    return ranked.map((entry) => entry.sentence).join(' ');
  }

  private extractQuestion(prompt: string): string {
    const match = /Question:\s*(.+)$/m.exec(prompt);
    return match?.[1]?.trim() ?? '';
  }

  /** Parses the numbered Sources block the engine's prompt builder emits. */
  private extractSources(prompt: string): string[] {
    const sourcesIndex = prompt.indexOf('Sources:');
    if (sourcesIndex === -1) return [];

    const questionIndex = prompt.lastIndexOf('Question:');
    const block = prompt.slice(
      sourcesIndex + 'Sources:'.length,
      questionIndex > sourcesIndex ? questionIndex : undefined,
    );

    return block
      .split(/\n(?=\[\d+\])/)
      .map((entry) => entry.replace(/^\[\d+\]\s*\([^)]*\)\s*/, '').trim())
      .filter((entry) => entry.length > 0);
  }

  private splitSentences(text: string): string[] {
    return text
      .split(/(?<=[.!?])\s+/)
      .map((sentence) => sentence.trim())
      .filter((sentence) => sentence.length > 0);
  }
}
