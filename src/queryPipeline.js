import { performance } from "node:perf_hooks";

import {
  answerQuestionWithSources,
  DEFAULT_ANSWER_MODEL,
  DEFAULT_MAX_OUTPUT_TOKENS
} from "./openaiAnswer.js";
import { rerankCandidatesWithTimings, selectRerankedChunks } from "./reranker.js";

export const DEFAULT_QUERY_OPTIONS = Object.freeze({
  answerMaxTokens: DEFAULT_MAX_OUTPUT_TOKENS,
  answerModel: DEFAULT_ANSWER_MODEL,
  bm25K: 2,
  candidatePool: null,
  candidateTopK: null,
  finalBm25Min: 1,
  finalVectorMin: 2,
  filter: {},
  mode: "vector",
  rerankModel: null,
  topK: 5,
  vectorK: 3
});

function coerceFiniteNumber(value, fallback) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function coerceOptionalFiniteNumber(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

export function normalizeQueryOptions(options = {}) {
  return {
    ...DEFAULT_QUERY_OPTIONS,
    ...options,
    answerMaxTokens: coerceFiniteNumber(options.answerMaxTokens, DEFAULT_QUERY_OPTIONS.answerMaxTokens),
    bm25K: coerceFiniteNumber(options.bm25K, DEFAULT_QUERY_OPTIONS.bm25K),
    candidatePool: coerceOptionalFiniteNumber(options.candidatePool),
    candidateTopK: coerceOptionalFiniteNumber(options.candidateTopK),
    finalBm25Min: coerceFiniteNumber(options.finalBm25Min, DEFAULT_QUERY_OPTIONS.finalBm25Min),
    finalVectorMin: coerceFiniteNumber(options.finalVectorMin, DEFAULT_QUERY_OPTIONS.finalVectorMin),
    filter: options.filter ?? {},
    topK: coerceFiniteNumber(options.topK, DEFAULT_QUERY_OPTIONS.topK),
    vectorK: coerceFiniteNumber(options.vectorK, DEFAULT_QUERY_OPTIONS.vectorK)
  };
}

export function resolveCandidateLimit(options = {}) {
  const normalized = normalizeQueryOptions(options);

  return Number.isFinite(normalized.candidateTopK)
    ? normalized.candidateTopK
    : Math.max(normalized.topK, normalized.vectorK + normalized.bm25K);
}

export async function buildRerankedSelection(store, query, options = {}) {
  const normalized = normalizeQueryOptions(options);
  const timings = [];
  const candidateLimit = resolveCandidateLimit(normalized);
  const hybridProfile = await store.hybridSearchProfile(query, {
    topK: candidateLimit,
    filter: normalized.filter,
    vectorK: normalized.vectorK,
    bm25K: normalized.bm25K,
    ...(Number.isFinite(normalized.candidatePool) ? { candidatePool: normalized.candidatePool } : {})
  });
  timings.push(...hybridProfile.timings);

  const rerankProfile = await rerankCandidatesWithTimings(query, hybridProfile.results, {
    topK: hybridProfile.results.length,
    ...(normalized.rerankModel ? { modelId: normalized.rerankModel } : {})
  });
  timings.push(...rerankProfile.timings);

  const selectionStart = performance.now();
  const results = selectRerankedChunks(rerankProfile.results, {
    minBm25: normalized.finalBm25Min,
    minVector: normalized.finalVectorMin,
    topK: normalized.topK
  });
  timings.push({
    label: "rerank.selection",
    ms: performance.now() - selectionStart
  });

  return {
    results,
    timings
  };
}

export async function runQuery(store, query, options = {}) {
  const normalized = normalizeQueryOptions(options);

  switch (normalized.mode) {
    case "vector": {
      const profile = await store.vectorSearchProfile(query, {
        topK: normalized.topK,
        filter: normalized.filter
      });

      return {
        mode: normalized.mode,
        results: profile.results,
        timings: profile.timings
      };
    }
    case "bm25": {
      const profile = store.bm25SearchProfile(query, {
        topK: normalized.topK,
        filter: normalized.filter
      });

      return {
        mode: normalized.mode,
        results: profile.results,
        timings: profile.timings
      };
    }
    case "hybrid":
      return {
        mode: normalized.mode,
        ...(await store.hybridSearchProfile(query, {
          topK: normalized.topK,
          filter: normalized.filter,
          vectorK: normalized.vectorK,
          bm25K: normalized.bm25K,
          ...(Number.isFinite(normalized.candidatePool) ? { candidatePool: normalized.candidatePool } : {})
        }))
      };
    case "rerank":
      return {
        mode: normalized.mode,
        ...(await buildRerankedSelection(store, query, normalized))
      };
    case "answer": {
      const selectedContext = await buildRerankedSelection(store, query, normalized);
      const answer = await answerQuestionWithSources(query, selectedContext.results, {
        model: normalized.answerModel,
        maxOutputTokens: normalized.answerMaxTokens
      });

      return {
        answer,
        chunks: selectedContext.results,
        mode: normalized.mode,
        timings: [...selectedContext.timings, ...answer.timings]
      };
    }
    default:
      throw new Error(`Unknown mode "${normalized.mode}". Use vector, bm25, hybrid, rerank, or answer.`);
  }
}
