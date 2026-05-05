import { mkdir, readFile, writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import path from "node:path";

import { chunkText } from "./chunker.js";
import { createEmbedderFromConfig, serializeEmbedder } from "./embedderFactory.js";
import { HashingEmbedder } from "./embedder.js";
import { countTermFrequencies, tokenize } from "./tokenize.js";

const BM25_DEFAULTS = {
  b: 0.75,
  k1: 1.2
};

function dotProduct(left, right) {
  let score = 0;

  for (let i = 0; i < left.length; i += 1) {
    score += left[i] * right[i];
  }

  return score;
}

function matchesFilter(metadata, filter) {
  return Object.entries(filter).every(([key, value]) => {
    const candidate = metadata[key];

    if (Array.isArray(candidate)) {
      return candidate.includes(value);
    }

    return candidate === value;
  });
}

function buildSearchResult(record, score, extra = {}) {
  return {
    id: record.id,
    documentId: record.documentId,
    chunkIndex: record.chunkIndex,
    text: record.text,
    metadata: record.metadata,
    score,
    ...extra
  };
}

function buildMergedCandidate(result) {
  return buildSearchResult(result, 0, {
    bm25Rank: null,
    bm25Score: null,
    hybridScore: 0,
    selectedBy: [],
    vectorRank: null,
    vectorScore: null
  });
}

function computeBm25Stats(records) {
  const documentFrequencies = Object.create(null);
  let totalLength = 0;

  for (const record of records) {
    totalLength += record.length;

    for (const term of Object.keys(record.termFreq)) {
      documentFrequencies[term] = (documentFrequencies[term] ?? 0) + 1;
    }
  }

  return {
    averageDocumentLength: records.length === 0 ? 0 : totalLength / records.length,
    documentCount: records.length,
    documentFrequencies
  };
}

function computeBm25Score({
  averageDocumentLength,
  b,
  documentCount,
  documentFrequencies,
  k1,
  queryTerms,
  record
}) {
  if (documentCount === 0 || averageDocumentLength === 0 || record.length === 0) {
    return 0;
  }

  let score = 0;

  for (const term of queryTerms) {
    const tf = record.termFreq[term] ?? 0;
    if (tf === 0) {
      continue;
    }

    const df = documentFrequencies[term] ?? 0;
    const idf = Math.log(1 + (documentCount - df + 0.5) / (df + 0.5));
    const lengthRatio = record.length / averageDocumentLength;
    const numerator = tf * (k1 + 1);
    const denominator = tf + k1 * (1 - b + b * lengthRatio);
    score += idf * (numerator / denominator);
  }

  return score;
}

function hydrateRecord(record) {
  if (record.termFreq && typeof record.length === "number") {
    return record;
  }

  const tokens = tokenize(record.text);

  return {
    ...record,
    length: tokens.length,
    termFreq: countTermFrequencies(tokens)
  };
}

export class VectorStore {
  constructor(options = {}) {
    this.embedder = options.embedder ?? new HashingEmbedder();
    this.chunkSize = options.chunkSize ?? 80;
    this.overlap = options.overlap ?? 20;
    this.storagePath = options.storagePath ?? null;
    this.records = (options.records ?? []).map(hydrateRecord);
    this.bm25 = options.bm25 ?? computeBm25Stats(this.records);
  }

  async addDocument({ id, text, metadata = {} }) {
    if (!id) {
      throw new Error("Document id is required");
    }

    if (!text || !text.trim()) {
      throw new Error("Document text is required");
    }

    const chunks = chunkText(text, {
      chunkSize: this.chunkSize,
      overlap: this.overlap
    });

    const inserted = [];
    const vectors = await this.embedder.embedBatch(chunks);

    chunks.forEach((chunk, index) => {
      const tokens = tokenize(chunk);
      const record = {
        id: `${id}#${index}`,
        documentId: id,
        chunkIndex: index,
        text: chunk,
        metadata: {
          ...metadata,
          sourceDocumentId: id
        },
        vector: vectors[index],
        length: tokens.length,
        termFreq: countTermFrequencies(tokens)
      };

      this.records.push(record);
      inserted.push(record);
    });

    this.bm25 = computeBm25Stats(this.records);
    return inserted;
  }

  deleteDocument(documentId) {
    const before = this.records.length;
    this.records = this.records.filter((record) => record.documentId !== documentId);
    this.bm25 = computeBm25Stats(this.records);
    return before - this.records.length;
  }

  async search(query, options = {}) {
    return this.vectorSearch(query, options);
  }

  async vectorSearchProfile(query, options = {}) {
    const start = performance.now();
    const results = await this.vectorSearch(query, options);

    return {
      results,
      timings: [
        {
          label: "vector_search",
          ms: performance.now() - start
        }
      ]
    };
  }

  async vectorSearch(query, options = {}) {
    const {
      topK = 5,
      filter = {}
    } = options;

    const queryVector = await this.embedder.embed(query);

    return this.records
      .filter((record) => matchesFilter(record.metadata, filter))
      .map((record) => buildSearchResult(record, dotProduct(queryVector, record.vector)))
      .sort((left, right) => right.score - left.score)
      .slice(0, topK);
  }

  bm25SearchProfile(query, options = {}) {
    const start = performance.now();
    const results = this.bm25Search(query, options);

    return {
      results,
      timings: [
        {
          label: "bm25_search",
          ms: performance.now() - start
        }
      ]
    };
  }

  bm25Search(query, options = {}) {
    const {
      topK = 5,
      filter = {},
      k1 = BM25_DEFAULTS.k1,
      b = BM25_DEFAULTS.b
    } = options;

    const queryTerms = [...new Set(tokenize(query))];
    if (queryTerms.length === 0) {
      return [];
    }

    return this.records
      .filter((record) => matchesFilter(record.metadata, filter))
      .map((record) => buildSearchResult(
        record,
        computeBm25Score({
          averageDocumentLength: this.bm25.averageDocumentLength,
          b,
          documentCount: this.bm25.documentCount,
          documentFrequencies: this.bm25.documentFrequencies,
          k1,
          queryTerms,
          record
        })
      ))
      .filter((result) => result.score > 0)
      .sort((left, right) => right.score - left.score)
      .slice(0, topK);
  }

  async hybridSearchProfile(query, options = {}) {
    const {
      topK = 5,
      filter = {},
      candidatePool = Math.max(topK * 6, 20),
      bm25K = 2,
      vectorK = 3
    } = options;
    const totalStart = performance.now();
    const vectorStart = performance.now();
    const vectorResults = await this.vectorSearch(query, { topK: candidatePool, filter });
    const vectorMs = performance.now() - vectorStart;
    const bm25Start = performance.now();
    const bm25Results = this.bm25Search(query, { topK: candidatePool, filter });
    const bm25Ms = performance.now() - bm25Start;
    const mergeStart = performance.now();
    const merged = new Map();

    for (const [rank, result] of vectorResults.entries()) {
      const candidate = merged.get(result.id) ?? buildMergedCandidate(result);

      candidate.vectorRank = rank + 1;
      candidate.vectorScore = result.score;
      candidate.hybridScore += 1 / (60 + rank + 1);
      merged.set(result.id, candidate);
    }

    for (const [rank, result] of bm25Results.entries()) {
      const candidate = merged.get(result.id) ?? buildMergedCandidate(result);

      candidate.bm25Rank = rank + 1;
      candidate.bm25Score = result.score;
      candidate.hybridScore += 1 / (60 + rank + 1);
      merged.set(result.id, candidate);
    }

    const combinedRanking = Array.from(merged.values())
      .sort((left, right) => right.hybridScore - left.hybridScore)
      .map((result) => ({
        ...result,
        score: result.hybridScore
      }));
    const mergeMs = performance.now() - mergeStart;

    const selectionStart = performance.now();
    const selectedIds = new Set();
    const selected = [];

    const takeUnique = (results, limit, label) => {
      for (const result of results) {
        if (selected.length >= topK || limit <= 0) {
          break;
        }

        if (selectedIds.has(result.id)) {
          continue;
        }

        result.selectedBy = [...result.selectedBy, label];
        selected.push(result);
        selectedIds.add(result.id);
        limit -= 1;
      }
    };

    takeUnique(vectorResults
      .map((result) => combinedRanking.find((candidate) => candidate.id === result.id))
      .filter(Boolean), vectorK, "vector");
    takeUnique(bm25Results
      .map((result) => combinedRanking.find((candidate) => candidate.id === result.id))
      .filter(Boolean), bm25K, "bm25");
    takeUnique(combinedRanking, topK - selected.length, "backfill");

    return {
      results: selected,
      timings: [
        {
          label: "hybrid.vector_search",
          ms: vectorMs
        },
        {
          label: "hybrid.bm25_search",
          ms: bm25Ms
        },
        {
          label: "hybrid.merge_rank",
          ms: mergeMs
        },
        {
          label: "hybrid.selection",
          ms: performance.now() - selectionStart
        },
        {
          label: "hybrid.total",
          ms: performance.now() - totalStart
        }
      ]
    };
  }

  async hybridSearch(query, options = {}) {
    return (await this.hybridSearchProfile(query, options)).results;
  }

  listDocuments() {
    const seen = new Map();

    for (const record of this.records) {
      if (!seen.has(record.documentId)) {
        seen.set(record.documentId, {
          documentId: record.documentId,
          metadata: record.metadata
        });
      }
    }

    return Array.from(seen.values());
  }

  async save() {
    if (!this.storagePath) {
      throw new Error("storagePath is required to save the store");
    }

    await mkdir(path.dirname(this.storagePath), { recursive: true });

    const payload = {
      embedder: {
        ...serializeEmbedder(this.embedder)
      },
      chunking: {
        chunkSize: this.chunkSize,
        overlap: this.overlap
      },
      bm25: this.bm25,
      records: this.records
    };

    await writeFile(this.storagePath, JSON.stringify(payload, null, 2), "utf8");
  }

  static async load(storagePath) {
    const raw = await readFile(storagePath, "utf8");
    const payload = JSON.parse(raw);

    return new VectorStore({
      storagePath,
      chunkSize: payload.chunking.chunkSize,
      overlap: payload.chunking.overlap,
      embedder: createEmbedderFromConfig(payload.embedder),
      bm25: payload.bm25,
      records: payload.records
    });
  }
}
