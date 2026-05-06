import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

import {
  DEFAULT_ANSWER_MODEL,
  DEFAULT_MAX_OUTPUT_TOKENS,
  formatSourceDetails,
  formatSourceLine
} from "./openaiAnswer.js";
import { serializeEmbedder } from "./embedderFactory.js";
import { normalizeQueryOptions, resolveCandidateLimit, runQuery } from "./queryPipeline.js";
import { VectorStore } from "./vectorStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const storagePath = path.resolve(__dirname, "../data/vector-store.json");

function parseArgs(argv) {
  const config = normalizeQueryOptions({
    answerMaxTokens: DEFAULT_MAX_OUTPUT_TOKENS,
    answerModel: DEFAULT_ANSWER_MODEL
  });
  const positional = [];

  for (const arg of argv) {
    if (arg.startsWith("--mode=") || arg.startsWith("mode=")) {
      config.mode = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("--topK=") || arg.startsWith("topK=")) {
      config.topK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--vectorK=") || arg.startsWith("vectorK=")) {
      config.vectorK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--bm25K=") || arg.startsWith("bm25K=")) {
      config.bm25K = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--candidatePool=") || arg.startsWith("candidatePool=")) {
      config.candidatePool = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--candidateTopK=") || arg.startsWith("candidateTopK=")) {
      config.candidateTopK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--finalVectorMin=") || arg.startsWith("finalVectorMin=")) {
      config.finalVectorMin = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--finalBm25Min=") || arg.startsWith("finalBm25Min=")) {
      config.finalBm25Min = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("--rerankModel=") || arg.startsWith("rerankModel=")) {
      config.rerankModel = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("--answerModel=") || arg.startsWith("answerModel=")) {
      config.answerModel = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("--answerMaxTokens=") || arg.startsWith("answerMaxTokens=")) {
      config.answerMaxTokens = Number(arg.split("=")[1]);
      continue;
    }

    positional.push(arg);
  }

  const [query, ...rest] = positional;
  if (!query) {
    throw new Error('Usage: npm run search -- mode=vector "your query" key=value');
  }

  const filter = {};
  for (const arg of rest) {
    const [key, value] = arg.split("=");
    if (key && value) {
      filter[key] = value;
    }
  }

  return normalizeQueryOptions({
    ...config,
    filter,
    query
  });
}

function formatMs(value) {
  if (value >= 1000) {
    return `${value.toFixed(1)} ms`;
  }

  if (value >= 100) {
    return `${value.toFixed(1)} ms`;
  }

  if (value >= 10) {
    return `${value.toFixed(2)} ms`;
  }

  return `${value.toFixed(3)} ms`;
}

function printTimings(timings) {
  if (!Array.isArray(timings) || timings.length === 0) {
    return;
  }

  const width = timings.reduce((maxWidth, entry) => Math.max(maxWidth, entry.label.length), 0);

  console.log("");
  console.log("Timings");
  for (const entry of timings) {
    console.log(`${entry.label.padEnd(width)}  ${formatMs(entry.ms)}`);
  }
}

async function main() {
  const {
    answerMaxTokens,
    answerModel,
    bm25K,
    candidatePool,
    candidateTopK,
    finalBm25Min,
    finalVectorMin,
    query,
    filter,
    mode,
    rerankModel,
    topK,
    vectorK
  } = parseArgs(process.argv.slice(2));
  const pipelineStart = performance.now();
  const loadStoreStart = performance.now();
  let store;

  try {
    store = await VectorStore.load(storagePath);
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error('Vector store not found. Run "npm run demo" or "npm run index:sources" first.');
    }

    throw error;
  }
  const loadStoreMs = performance.now() - loadStoreStart;

  const execution = await runQuery(store, query, {
    answerMaxTokens,
    answerModel,
    bm25K,
    candidatePool,
    candidateTopK,
    finalBm25Min,
    finalVectorMin,
    filter,
    mode,
    rerankModel,
    topK,
    vectorK
  });
  const timings = [
    {
      label: "load_store",
      ms: loadStoreMs
    },
    ...(execution.timings ?? []),
    {
      label: "pipeline.total",
      ms: performance.now() - pipelineStart
    }
  ];
  const embedderConfig = serializeEmbedder(store.embedder);

  console.log(`Loaded ${store.records.length} chunk(s) from ${storagePath}`);
  console.log(`Mode: ${mode}`);
  console.log(`Embedder: ${embedderConfig.type}${embedderConfig.model ? ` model=${embedderConfig.model}` : ""}${embedderConfig.dimensions ? ` dimensions=${embedderConfig.dimensions}` : ""}`);
  if (mode === "hybrid") {
    console.log(`Quota: vector=${vectorK} bm25=${bm25K} topK=${topK}`);
  }
  if (mode === "rerank") {
    const candidateLimit = resolveCandidateLimit({
      bm25K,
      candidateTopK,
      topK,
      vectorK
    });
    console.log(
      `Candidate quota: vector=${vectorK} bm25=${bm25K} candidateTopK=${candidateLimit} finalTopK=${topK} finalVectorMin=${finalVectorMin} finalBm25Min=${finalBm25Min}`
    );
  }
  if (mode === "answer") {
    const candidateLimit = resolveCandidateLimit({
      bm25K,
      candidateTopK,
      topK,
      vectorK
    });
    console.log(
      `Answer context: vector=${vectorK} bm25=${bm25K} candidateTopK=${candidateLimit} selectedChunks=${topK} finalVectorMin=${finalVectorMin} finalBm25Min=${finalBm25Min} model=${answerModel} maxOutputTokens=${answerMaxTokens}`
    );
  }
  console.log(`Query: ${query}`);

  if (Object.keys(filter).length > 0) {
    console.log(`Filter: ${JSON.stringify(filter)}`);
  }

  printTimings(timings);

  if (mode === "answer") {
    if (execution.chunks.length === 0) {
      console.log("No matches");
      return;
    }

    console.log("");
    console.log("Answer");
    console.log(execution.answer.answer);

    if (execution.answer.insufficientContext) {
      console.log("");
      console.log("Note: the model reported that the provided chunks may not fully support the answer.");
    }

    console.log("");
    console.log("Sources");

    if (execution.answer.sources.length === 0) {
      console.log("- No source IDs were cited by the model.");
      return;
    }

    for (const chunk of execution.answer.sources) {
      console.log(`- ${formatSourceLine(chunk)}`);
      for (const detail of formatSourceDetails(chunk)) {
        console.log(`  ${detail}`);
      }
    }

    return;
  }

  if (execution.results.length === 0) {
    console.log("No matches");
    return;
  }

  for (const result of execution.results) {
    console.log("");
    console.log(`score=${result.score.toFixed(3)} doc=${result.documentId} chunk=${result.chunkIndex}`);

    if (mode === "hybrid") {
      console.log(JSON.stringify({
        bm25Rank: result.bm25Rank,
        bm25Score: result.bm25Score,
        selectedBy: result.selectedBy,
        vectorRank: result.vectorRank,
        vectorScore: result.vectorScore
      }));
    }

    if (mode === "rerank") {
      console.log(JSON.stringify({
        rerankModelId: result.rerankModelId,
        rerankScore: result.rerankScore,
        selectedBy: result.selectedBy,
        bm25Rank: result.bm25Rank,
        bm25Score: result.bm25Score,
        vectorRank: result.vectorRank,
        vectorScore: result.vectorScore
      }));
    }

    console.log(JSON.stringify(result.metadata));
    console.log(result.text);
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
