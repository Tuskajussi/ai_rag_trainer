# First Vector Store in Node

This project is a small, from-scratch vector store built in plain Node.js.

It is intentionally simple:

- It chunks documents into overlapping word windows.
- It converts text into vectors using either a local hashing embedder (default, no API key needed) or OpenAI embeddings (`text-embedding-3-small` / `text-embedding-3-large`).
- It supports BM25 lexical retrieval over the same chunks.
- It stores vectors, chunk text, and metadata in a JSON file.
- It runs nearest-neighbor search with cosine similarity.

This is a teaching system, not a production semantic search engine.

## Train your first model

If you want the smallest real training loop in this repo, run:

```bash
npm run train:simple
```

That example lives in [src/trainSimpleModel.js](/C:/git/ai_trainer/src/trainSimpleModel.js) and is explained in [docs/train-simple-model.md](/C:/git/ai_trainer/docs/train-simple-model.md).

It trains a tiny linear model from scratch using:

- weights
- bias
- mean squared error
- gradient descent

This is the best first step before moving to logistic regression or neural nets.

## Why this is useful

If you do not yet understand vector databases, this project lets you see the whole pipeline:

1. Raw text goes in.
2. Text becomes chunks.
3. Chunks become vectors.
4. Vectors are stored with metadata.
5. A query becomes a vector.
6. Similar vectors are returned as matches.

Once this clicks, you can swap the toy embedder for a real embedding model and the JSON file for Pinecone, Postgres + pgvector, Qdrant, or OpenAI vector stores.

## Files

- [src/chunker.js](/C:/git/ai_trainer/src/chunker.js)
- [src/embedder.js](/C:/git/ai_trainer/src/embedder.js)
- [src/vectorStore.js](/C:/git/ai_trainer/src/vectorStore.js)
- [src/demo.js](/C:/git/ai_trainer/src/demo.js)
- [src/reranker.js](/C:/git/ai_trainer/src/reranker.js)
- [src/search.js](/C:/git/ai_trainer/src/search.js)
- [src/server.js](/C:/git/ai_trainer/src/server.js)
- [public/index.html](/C:/git/ai_trainer/public/index.html)
- [public/app.js](/C:/git/ai_trainer/public/app.js)
- [public/styles.css](/C:/git/ai_trainer/public/styles.css)

## Run it

```bash
npm run demo
```

That command will:

- build a small index from sample documents
- save it to `data/vector-store.json`
- run a few example searches

After that, try your own queries:

```bash
npm run search -- "how long do refunds take"
```

Or filter by metadata:

```bash
npm run search -- "can I use SSO" product=enterprise
```

You can also choose a retrieval mode:

```bash
npm run search -- mode=vector "Mille alustoille prototyyppi tehtiin?"
npm run search -- mode=bm25 "Mille alustoille prototyyppi tehtiin?"
npm run search -- mode=hybrid "Mille alustoille prototyyppi tehtiin?"
npm run search -- mode=hybrid topK=5 vectorK=3 bm25K=2 "Mille alustoille prototyyppi tehtiin?"
npm run search -- mode=rerank topK=3 vectorK=3 bm25K=2 candidateTopK=5 "Mille alustoille prototyyppi tehtiin?"
npm run search -- mode=rerank topK=5 vectorK=10 bm25K=10 candidateTopK=10 finalVectorMin=2 finalBm25Min=1 "Multer"
npm run search -- mode=answer topK=5 vectorK=10 bm25K=10 candidateTopK=10 finalVectorMin=2 finalBm25Min=1 "Miten valokuvat prosessoidaan nakkilistassa?"
```

For `mode=answer`, set `OPENAI_API_KEY` first. The CLI will:

- retrieve and rerank a chunk set
- send only those chunks to the OpenAI Responses API
- print a grounded answer
- print a `Sources` section built from the cited chunk IDs

You can put the key in a local `.env` file:

```dotenv
OPENAI_API_KEY=your-key
```

Or set it directly in PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
npm run search -- mode=answer topK=5 vectorK=10 bm25K=10 candidateTopK=10 "Miten valokuvat prosessoidaan nakkilistassa?"
```

## Run the local UI

Start the browser workbench:

```bash
npm run ui
```

Then open:

```txt
http://127.0.0.1:4173
```

The UI lets you:

- ask questions in `vector`, `bm25`, `hybrid`, `rerank`, or `answer` mode
- inspect timings for each pipeline step
- inspect returned chunks, metadata, and ranking signals
- see grounded `Sources` for answer mode
- reload the current `data/vector-store.json` without restarting the server

The browser UI uses the same query pipeline as the CLI, so the results should match `npm run search`.

## Index a real PDF from `sources`

Put a PDF in [sources](/C:/git/ai_trainer/sources) and optionally add a [meta.txt](/C:/git/ai_trainer/sources/meta.txt) file with tab-separated metadata rows.

Then run:

```bash
npm run index:sources
```

Or build a real embedding index with OpenAI:

```bash
npm run index:sources:openai
```

That uses `text-embedding-3-small` by default. You can also override it:

```bash
npm run index:sources -- embedder=openai embeddingModel=text-embedding-3-large
npm run index:sources -- embedder=openai embeddingModel=text-embedding-3-small dimensions=1024
```

That command will:

- extract text from each PDF in `sources`
- parse shared metadata from `meta.txt`
- attach normalized metadata fields such as `sourceTitle`, `sourceAuthor`, `sourceLanguage`, and `sourceSubjects`
- chunk and index the extracted text into `data/vector-store.json`

Example searches:

```bash
npm run search -- "Mille alustoille prototyyppi tehtiin?"
npm run search -- "Kuka on työn tekijä?" sourceAuthor="Mäki, Jussi"
npm run search -- mode=bm25 "Multer"
npm run search -- mode=rerank topK=3 vectorK=3 bm25K=2 candidateTopK=5 "Multer"
```

The search output now shows which embedder was used for the loaded store. If the store was indexed with OpenAI embeddings, query-time vector search will call the OpenAI embeddings API to embed the user query before comparing it to stored chunk vectors.

## Evaluate the pipeline

A small thesis eval set lives at [evals/thesis-eval.json](/C:/git/ai_trainer/evals/thesis-eval.json).

Run retrieval-focused evals:

```bash
npm run eval -- mode=vector
npm run eval -- mode=bm25
npm run eval -- mode=hybrid
npm run eval -- mode=rerank
npm run eval -- mode=compare
```

Run end-to-end answer evals:

```bash
npm run eval -- mode=answer
npm run eval -- mode=all
```

The eval runner reports:

- `Hit@1`, `Hit@3`, `Hit@5`, and `MRR@K` for retrieval when available
- average timings across the eval set
- in `mode=compare`, side-by-side retrieval metrics for `vector`, `bm25`, `hybrid`, and `rerank`
- for `mode=answer`, a heuristic answer score based on:
  - whether a relevant source chunk was cited
  - whether required answer terms were present

## What is "fake" here

The embedder is local and deterministic. It uses token hashing, not a real semantic embedding model.

That means:

- it is good for learning the mechanics
- it can match shared terms and similar word shapes
- it is not a substitute for real semantic embeddings

## Good next step

After you understand this version, the next upgrade is:

1. keep the same `VectorStore` shape
2. replace `HashingEmbedder` with a real embedding API
3. keep the same search and metadata ideas

That is usually the easiest path from "I don't get vector databases" to "I can build one."
