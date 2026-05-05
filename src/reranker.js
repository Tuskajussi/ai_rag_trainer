import { AutoModelForSequenceClassification, AutoTokenizer } from "@huggingface/transformers";
import { performance } from "node:perf_hooks";

const DEFAULT_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2";
const modelCache = new Map();

function getCacheKey(modelId, maxLength) {
  return `${modelId}::${maxLength}`;
}

async function loadReranker(modelId, maxLength) {
  const cacheKey = getCacheKey(modelId, maxLength);

  if (!modelCache.has(cacheKey)) {
    modelCache.set(cacheKey, (async () => {
      const tokenizer = await AutoTokenizer.from_pretrained(modelId);
      const model = await AutoModelForSequenceClassification.from_pretrained(modelId);

      return {
        maxLength,
        model,
        modelId,
        tokenizer
      };
    })());
  }

  return modelCache.get(cacheKey);
}

function extractScores(logits) {
  const [rowCount, columnCount] = logits.dims;
  const raw = Array.from(logits.data);

  if (columnCount === 1) {
    return raw;
  }

  if (columnCount === 2) {
    const scores = [];

    for (let row = 0; row < rowCount; row += 1) {
      const start = row * columnCount;
      scores.push(raw[start + 1] - raw[start]);
    }

    return scores;
  }

  throw new Error(`Unsupported reranker logits shape: ${JSON.stringify(logits.dims)}`);
}

export async function rerankCandidatesWithTimings(query, candidates, options = {}) {
  const {
    maxLength = 512,
    modelId = DEFAULT_MODEL_ID,
    topK = candidates.length
  } = options;

  if (candidates.length === 0) {
    return {
      results: [],
      timings: []
    };
  }

  const totalStart = performance.now();
  const loadStart = performance.now();
  const reranker = await loadReranker(modelId, maxLength);
  const loadMs = performance.now() - loadStart;
  const pairs = candidates.map((candidate) => [query, candidate.text]);
  const tokenizeStart = performance.now();
  const inputs = await reranker.tokenizer(pairs, {
    max_length: reranker.maxLength,
    padding: true,
    truncation: true
  });
  const tokenizeMs = performance.now() - tokenizeStart;
  const inferenceStart = performance.now();
  const output = await reranker.model(inputs);
  const inferenceMs = performance.now() - inferenceStart;
  const scores = extractScores(output.logits);
  const rankingStart = performance.now();

  return {
    results: candidates
      .map((candidate, index) => ({
        ...candidate,
        rerankModelId: reranker.modelId,
        rerankScore: scores[index],
        score: scores[index]
      }))
      .sort((left, right) => right.rerankScore - left.rerankScore)
      .slice(0, topK),
    timings: [
      {
        label: "rerank.model_load",
        ms: loadMs
      },
      {
        label: "rerank.tokenize",
        ms: tokenizeMs
      },
      {
        label: "rerank.inference",
        ms: inferenceMs
      },
      {
        label: "rerank.rank_sort",
        ms: performance.now() - rankingStart
      },
      {
        label: "rerank.total",
        ms: performance.now() - totalStart
      }
    ]
  };
}

export async function rerankCandidates(query, candidates, options = {}) {
  const { results } = await rerankCandidatesWithTimings(query, candidates, options);
  return results;
}

export function selectRerankedChunks(candidates, options = {}) {
  const {
    topK = 5,
    minBm25 = 1,
    minVector = 2
  } = options;

  if (candidates.length === 0 || topK <= 0) {
    return [];
  }

  const selected = [];
  const selectedIds = new Set();
  const cappedMinBm25 = Math.min(minBm25, topK);
  const cappedMinVector = Math.min(minVector, Math.max(topK - cappedMinBm25, 0));
  const bestVectorFirst = [...candidates].sort((left, right) => {
    if (left.vectorRank === null && right.vectorRank === null) {
      return right.rerankScore - left.rerankScore;
    }

    if (left.vectorRank === null) {
      return 1;
    }

    if (right.vectorRank === null) {
      return -1;
    }

    return left.vectorRank - right.vectorRank;
  });
  const bestBm25First = [...candidates].sort((left, right) => {
    if (left.bm25Rank === null && right.bm25Rank === null) {
      return right.rerankScore - left.rerankScore;
    }

    if (left.bm25Rank === null) {
      return 1;
    }

    if (right.bm25Rank === null) {
      return -1;
    }

    return left.bm25Rank - right.bm25Rank;
  });

  const takeBest = (predicate, limit, source = candidates) => {
    if (limit <= 0) {
      return;
    }

    for (const candidate of source) {
      if (selected.length >= topK || limit <= 0) {
        break;
      }

      if (selectedIds.has(candidate.id) || !predicate(candidate)) {
        continue;
      }

      selected.push(candidate);
      selectedIds.add(candidate.id);
      limit -= 1;
    }
  };

  takeBest((candidate) => candidate.vectorRank !== null, cappedMinVector, bestVectorFirst);
  takeBest((candidate) => candidate.bm25Rank !== null, cappedMinBm25, bestBm25First);
  takeBest(() => true, topK - selected.length);

  return selected.sort((left, right) => right.rerankScore - left.rerankScore);
}

export { DEFAULT_MODEL_ID };
