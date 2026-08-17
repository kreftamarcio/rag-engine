---
title: Examples
audience: iniciante
last_verified: 2026-08-16
status: canonical
---

# Examples

Every example here runs with **no API key, no database and no network**. The adapters in
[`in-memory-adapters.ts`](in-memory-adapters.ts) are real working implementations, so you
can execute an example and compare its output against what this page claims.

## Prerequisites

```bash
npm install
```

Node 20 or later. `tsx` comes from devDependencies, so no global install is needed.

---

## 01 — Hello world

The smallest configuration that answers a question: embedder, vector store, generator,
chunking policy. Reranking, citations, grounding and query rewriting are all off.

```bash
npx tsx examples/01-hello-world.ts
```

**What the output contains**

- An ingestion summary: 2 documents processed, several chunks created, 0 errors. The chunk
  count is above 2 because `maxTokens` is set to 40, which splits both documents.
- An answer for *"What is the rate limit?"* quoting the rate-limit sentences.
- An answer for *"How do I authenticate?"* quoting the authentication sentences.
- **No answer** for *"How do I sign in?"*, printed as `(no context retrieved, generation
  skipped)`.
- `grounding: not configured` on every query, because no scorer was passed.

**Why the third query fails, on purpose**

The documents say *authentication*; the question says *sign in*. They share no vocabulary,
and the example's embedder is lexical: it builds a term-frequency vector, so cosine
similarity reflects word overlap and nothing more. A real embedding model bridges that gap.
Seeing the miss in the output is more useful than reading a warning about it.

Note what the engine does **not** do here: it returns an empty answer rather than generating
from zero context. An answer with no sources has nothing grounding it.

---

## 02 — Full pipeline

Every stage enabled: hybrid retrieval with BM25, cross-encoder reranking, citation
attribution, grounding verification.

```bash
npx tsx examples/02-full-pipeline.ts
```

**What the output contains**

- An `[info]` line from the injected logger listing which stages are active.
- For *"What does error E4021 mean?"*: an answer about payload size, retrieved and context
  counts, `inferenceCalls`, a grounding verdict, and citation lines shaped
  `[1] docs/errors.md (0.xxx)` followed by matching `bib [1]` entries.
- For *"How are webhooks verified?"*: an answer about the HMAC SHA-256 signature.
- A `grounding verdicts` table with four rows.

**The verdict table is the point of this example**

| Claim | Expected verdict |
|---|---|
| `The default rate limit is 1000 requests per minute per key.` | `supported` |
| `The default rate limit is **not** 1000 requests per minute per key.` | `contradicted` |
| `The default rate limit is **5000** requests per minute per key.` | `contradicted` |
| `Webhook signatures use HMAC SHA-512.` | `unmentioned` |

Rows 2 and 3 are what separates this scorer from lexical overlap. Both share nearly every
word with the source, so an overlap-only checker rates them as well-grounded. Row 2 is
caught by polarity mismatch, row 3 by numeric mismatch.

Contradiction is checked directly against `HallucinationScorer` rather than through
`query()`, because the extractive generator copies sentences from the sources and therefore
**cannot** produce a contradiction. Fabricating one by hand is the only honest way to show
the detector working.

**Why sparse retrieval is weighted higher here**

`denseWeight: 0.4, sparseWeight: 0.6`, inverted from the usual default. The example's
embedder is weak, and BM25 matches the exact token `e4021` while a term-frequency vector
dilutes it among common words. In production with a real embedding model the usual ratio is
the other way round.

**Why the thresholds look low**

`citationThreshold: 0.25` and `reranker.minScore: 0.2`. The lexical embedder produces lower
similarities than a real model, so a production threshold of `0.65` would cite nothing at
all. Treat these numbers as calibrated to the fake embedder, not as recommendations.

---

## The adapters

[`in-memory-adapters.ts`](in-memory-adapters.ts) implements all four interfaces. Copy it as
a starting point, but know what each one is:

| Adapter | Honest description |
|---|---|
| `HashingEmbedder` | Term-frequency vector via the hashing trick. Real cosine over word overlap, zero semantic generalization. Hash collisions are unmitigated. |
| `MemoryVectorStore` | Brute-force cosine over a `Map`. Correct, and O(n) per query. |
| `MemoryBM25Index` | **Genuine Okapi BM25**, `k1 = 1.2`, `b = 0.75`, with IDF smoothing. Retracts document frequency on re-index so IDF does not drift. |
| `lexicalRerankerScore` | Query-term containment. A real cross-encoder models term interaction; this cannot, and will rank a keyword-stuffed passage above a well-written one. |
| `ExtractiveGenerator` | Selects sentences from the context instead of writing new ones. Grounding is high **by construction**, so it verifies wiring rather than model quality. |

---

## Common failures

| Symptom | Cause |
|---|---|
| `Cannot find module '../src/index.js'` | Run from the repository root, not from inside `examples/` |
| Process hangs after the last line | `engine.dispose()` was not called; the tiktoken native handle keeps the event loop alive |
| Every query returns no context | `chunking.maxTokens` larger than your documents, so nothing splits and the single chunk matches poorly. Lower it. |
| Zero citations with real embeddings | `citationThreshold` still at the example's `0.25`, or conversely at `0.65` while using the lexical embedder |
| `Chunk "x" has no embedding` | `upsert` called without embedding first. `RAGEngine.ingest` handles this; a hand-rolled path must not skip it. |

---

## Verifying a run

A successful run of example 02 prints, in order: one `[info]` line, two query blocks each
with a non-empty answer, and a four-row verdict table containing exactly one `supported`,
two `contradicted` and one `unmentioned`.

If the verdict table shows `supported` on rows 2 or 3, contradiction detection has
regressed and that is a P0 defect, not a tuning issue.

---

## Not covered by these examples

- **Real providers.** No OpenAI, Anthropic, Qdrant or pgvector adapter ships in this
  repository. See [Not implemented](../README.md#not-implemented).
- **Typechecking.** `examples/` sits outside the `tsconfig.json` include set, so
  `npm run typecheck` skips it. Running the examples is the check.
- **`SemanticChunker`.** Exported and usable directly, but `RAGEngine` always uses
  `RecursiveChunker`, so no example exercises it.
- **HyDE.** `QueryRewriter.hyde()` exists and is exported, but `query()` never selects it.
