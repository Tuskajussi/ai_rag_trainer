import path from "node:path";
import { performance } from "node:perf_hooks";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { answerQuestionWithSources } from "./openaiAnswer.js";
import { serializeEmbedder } from "./embedderFactory.js";
import { rerankCandidatesWithTimings, selectRerankedChunks } from "./reranker.js";
import { VectorStore } from "./vectorStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const storagePath = path.resolve(__dirname, "../data/vector-store.json");
const defaultEvalPath = path.resolve(__dirname, "../evals/thesis-eval.json");
const RETRIEVAL_COMPARE_MODES = ["vector", "bm25", "hybrid", "rerank"];

function parseArgs(argv) {
  const config = {
    answerMaxTokens: 700,
    answerModel: null,
    bm25K: 10,
    candidateTopK: 10,
    evalPath: defaultEvalPath,
    finalBm25Min: 1,
    finalVectorMin: 2,
    mode: "rerank",
    rerankModel: null,
    topK: 5,
    vectorK: 10
  };

  for (const arg of argv) {
    if (arg.startsWith("mode=") || arg.startsWith("--mode=")) {
      config.mode = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("topK=") || arg.startsWith("--topK=")) {
      config.topK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("vectorK=") || arg.startsWith("--vectorK=")) {
      config.vectorK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("bm25K=") || arg.startsWith("--bm25K=")) {
      config.bm25K = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("candidateTopK=") || arg.startsWith("--candidateTopK=")) {
      config.candidateTopK = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("finalVectorMin=") || arg.startsWith("--finalVectorMin=")) {
      config.finalVectorMin = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("finalBm25Min=") || arg.startsWith("--finalBm25Min=")) {
      config.finalBm25Min = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("rerankModel=") || arg.startsWith("--rerankModel=")) {
      config.rerankModel = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("answerModel=") || arg.startsWith("--answerModel=")) {
      config.answerModel = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("answerMaxTokens=") || arg.startsWith("--answerMaxTokens=")) {
      config.answerMaxTokens = Number(arg.split("=")[1]);
      continue;
    }

    if (arg.startsWith("evalPath=") || arg.startsWith("--evalPath=")) {
      config.evalPath = path.resolve(process.cwd(), arg.split("=")[1]);
    }
  }

  return config;
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

function percentage(numerator, denominator) {
  if (denominator === 0) {
    return "0.0%";
  }

  return `${((numerator / denominator) * 100).toFixed(1)}%`;
}

function normalizeText(value) {
  return value.toLocaleLowerCase("fi-FI");
}

function accumulateTimings(target, timings) {
  for (const entry of timings ?? []) {
    const aggregate = target.get(entry.label) ?? { count: 0, totalMs: 0 };
    aggregate.count += 1;
    aggregate.totalMs += entry.ms;
    target.set(entry.label, aggregate);
  }
}

function getHitThresholds(topK) {
  return [...new Set([1, 3, 5, topK])]
    .filter((value) => value > 0 && value <= topK)
    .sort((left, right) => left - right);
}

function pad(value, width) {
  return String(value).padEnd(width);
}

function averageTimingMs(timingTotals, label) {
  const aggregate = timingTotals.get(label);

  if (!aggregate || aggregate.count === 0) {
    return null;
  }

  return aggregate.totalMs / aggregate.count;
}

async function buildRerankedSelection(store, query, options) {
  const timings = [];
  const candidateLimit = Number.isFinite(options.candidateTopK)
    ? options.candidateTopK
    : Math.max(options.topK, options.vectorK + options.bm25K);
  const hybridProfile = await store.hybridSearchProfile(query, {
    topK: candidateLimit,
    vectorK: options.vectorK,
    bm25K: options.bm25K
  });
  timings.push(...hybridProfile.timings);

  const rerankProfile = await rerankCandidatesWithTimings(query, hybridProfile.results, {
    topK: hybridProfile.results.length,
    ...(options.rerankModel ? { modelId: options.rerankModel } : {})
  });
  timings.push(...rerankProfile.timings);

  const selectionStart = performance.now();
  const results = selectRerankedChunks(rerankProfile.results, {
    minBm25: options.finalBm25Min,
    minVector: options.finalVectorMin,
    topK: options.topK
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

async function runCase(store, testCase, options, mode) {
  switch (mode) {
    case "vector":
      return store.vectorSearchProfile(testCase.question, { topK: options.topK });
    case "bm25":
      return store.bm25SearchProfile(testCase.question, { topK: options.topK });
    case "hybrid":
      return store.hybridSearchProfile(testCase.question, {
        topK: options.topK,
        vectorK: options.vectorK,
        bm25K: options.bm25K
      });
    case "rerank":
      return buildRerankedSelection(store, testCase.question, options);
    case "answer": {
      const selection = await buildRerankedSelection(store, testCase.question, options);
      const answer = await answerQuestionWithSources(testCase.question, selection.results, {
        ...(options.answerModel ? { model: options.answerModel } : {}),
        maxOutputTokens: options.answerMaxTokens
      });

      return {
        answer,
        results: selection.results,
        timings: [...selection.timings, ...answer.timings]
      };
    }
    default:
      throw new Error(`Unsupported eval mode "${mode}"`);
  }
}

function scoreRetrieval(results, relevantChunkIds) {
  const relevant = new Set(relevantChunkIds);
  const relevantRanks = [];

  results.forEach((result, index) => {
    if (relevant.has(result.id)) {
      relevantRanks.push(index + 1);
    }
  });

  const firstRelevantRank = relevantRanks[0] ?? null;

  return {
    firstRelevantRank,
    hit: relevantRanks.length > 0,
    relevantCount: relevantRanks.length,
    relevantRanks,
    reciprocalRank: firstRelevantRank ? 1 / firstRelevantRank : 0
  };
}

function scoreAnswer(answer, testCase) {
  const mustContain = testCase.mustContain ?? [];
  const normalizedAnswer = normalizeText(answer.answer);
  const coveredTerms = mustContain.filter((term) => normalizedAnswer.includes(normalizeText(term)));
  const relevant = new Set(testCase.relevantChunkIds);
  const citedRelevant = answer.sources.filter((source) => relevant.has(source.id));

  return {
    citedRelevantCount: citedRelevant.length,
    citedRelevantIds: citedRelevant.map((source) => source.id),
    mustContainCoverage: mustContain.length === 0 ? 1 : coveredTerms.length / mustContain.length,
    mustContainCovered: coveredTerms,
    pass: (mustContain.length === 0 || coveredTerms.length === mustContain.length) && citedRelevant.length > 0
  };
}

function buildSummary(scoredCases, topK, mode) {
  const thresholds = getHitThresholds(topK);
  const retrievalHits = scoredCases.filter((item) => item.retrieval.hit).length;
  const meanReciprocalRank =
    scoredCases.reduce((total, item) => total + item.retrieval.reciprocalRank, 0) / scoredCases.length;
  const meanRelevantReturned =
    scoredCases.reduce((total, item) => total + item.retrieval.relevantCount, 0) / scoredCases.length;
  const hitsByThreshold = Object.fromEntries(
    thresholds.map((threshold) => [
      threshold,
      scoredCases.filter((item) => item.retrieval.firstRelevantRank !== null && item.retrieval.firstRelevantRank <= threshold).length
    ])
  );

  const summary = {
    hitsByThreshold,
    meanReciprocalRank,
    meanRelevantReturned,
    retrievalHits,
    thresholds
  };

  if (mode === "answer") {
    const answerPasses = scoredCases.filter((item) => item.answer.pass).length;
    const sourceHits = scoredCases.filter((item) => item.answer.citedRelevantCount > 0).length;
    const meanCoverage =
      scoredCases.reduce((total, item) => total + item.answer.mustContainCoverage, 0) / scoredCases.length;

    summary.answer = {
      answerPasses,
      meanCoverage,
      sourceHits
    };
  }

  return summary;
}

async function runEvaluation(store, cases, options, mode) {
  const timingTotals = new Map();
  const scoredCases = [];

  for (const testCase of cases) {
    const caseStart = performance.now();
    const execution = await runCase(store, testCase, options, mode);
    const caseTotalMs = performance.now() - caseStart;
    accumulateTimings(timingTotals, execution.timings);
    accumulateTimings(timingTotals, [{
      label: "eval.case_total",
      ms: caseTotalMs
    }]);

    const retrieval = scoreRetrieval(execution.results, testCase.relevantChunkIds);
    const scored = {
      id: testCase.id,
      question: testCase.question,
      retrieval,
      timings: execution.timings
    };

    if (mode === "answer") {
      scored.answer = scoreAnswer(execution.answer, testCase);
    }

    scoredCases.push(scored);
  }

  return {
    mode,
    scoredCases,
    summary: buildSummary(scoredCases, options.topK, mode),
    timingTotals
  };
}

async function loadEvalCases(evalPath) {
  const raw = await readFile(evalPath, "utf8");
  const payload = JSON.parse(raw);

  if (!Array.isArray(payload) || payload.length === 0) {
    throw new Error(`No eval cases found in ${evalPath}`);
  }

  return payload;
}

function printAverageTimings(timingTotals) {
  console.log("");
  console.log("Average Timings");
  for (const [label, aggregate] of [...timingTotals.entries()].sort(([left], [right]) => left.localeCompare(right))) {
    console.log(`${label}  ${formatMs(aggregate.totalMs / aggregate.count)}`);
  }
}

function printPerQuestion(scoredCases, mode) {
  console.log("");
  console.log("Per Question");
  for (const item of scoredCases) {
    const retrievalPart = [
      `${item.id}`,
      `hit=${item.retrieval.hit ? "yes" : "no"}`,
      `rr=${item.retrieval.reciprocalRank.toFixed(3)}`,
      `ranks=${item.retrieval.relevantRanks.join(",") || "-"}`
    ].join("  ");
    console.log(retrievalPart);
    console.log(`  ${item.question}`);

    if (mode === "answer") {
      console.log(
        `  answer_pass=${item.answer.pass ? "yes" : "no"}  source_hits=${item.answer.citedRelevantCount}  must_contain=${(item.answer.mustContainCoverage * 100).toFixed(0)}%`
      );
      if (item.answer.mustContainCovered.length > 0) {
        console.log(`  covered_terms=${item.answer.mustContainCovered.join(", ")}`);
      }
      if (item.answer.citedRelevantIds.length > 0) {
        console.log(`  cited_relevant=${item.answer.citedRelevantIds.join(", ")}`);
      }
    }
  }
}

function printSingleModeEvaluation(evaluation, cases, options, embedderConfig, evalPath) {
  console.log(`Eval set: ${evalPath}`);
  console.log(`Mode: ${evaluation.mode}`);
  console.log(`Embedder: ${embedderConfig.type}${embedderConfig.model ? ` model=${embedderConfig.model}` : ""}`);
  console.log(`Questions: ${cases.length}`);
  console.log("");
  console.log("Retrieval Summary");
  for (const threshold of evaluation.summary.thresholds) {
    const hits = evaluation.summary.hitsByThreshold[threshold];
    console.log(`Hit@${threshold}: ${hits}/${cases.length} (${percentage(hits, cases.length)})`);
  }
  console.log(`MRR@${options.topK}: ${evaluation.summary.meanReciprocalRank.toFixed(3)}`);
  console.log(`Mean relevant returned: ${evaluation.summary.meanRelevantReturned.toFixed(2)}`);

  if (evaluation.mode === "answer") {
    console.log("");
    console.log("Answer Summary");
    console.log(
      `Relevant source cited: ${evaluation.summary.answer.sourceHits}/${cases.length} (${percentage(evaluation.summary.answer.sourceHits, cases.length)})`
    );
    console.log(`Must-contain coverage: ${(evaluation.summary.answer.meanCoverage * 100).toFixed(1)}%`);
    console.log(
      `Heuristic pass rate: ${evaluation.summary.answer.answerPasses}/${cases.length} (${percentage(evaluation.summary.answer.answerPasses, cases.length)})`
    );
  }

  printAverageTimings(evaluation.timingTotals);
  printPerQuestion(evaluation.scoredCases, evaluation.mode);
}

function printCompareEvaluation(evaluations, cases, options, embedderConfig, evalPath) {
  const modeWidth = Math.max(...evaluations.map((evaluation) => evaluation.mode.length), 4);
  const thresholds = getHitThresholds(options.topK);

  console.log(`Eval set: ${evalPath}`);
  console.log("Mode: compare");
  console.log(`Embedder: ${embedderConfig.type}${embedderConfig.model ? ` model=${embedderConfig.model}` : ""}`);
  console.log(`Questions: ${cases.length}`);
  console.log("");
  console.log("Mode Comparison");

  const headerParts = [
    pad("mode", modeWidth),
    ...thresholds.map((threshold) => pad(`Hit@${threshold}`, 10)),
    pad(`MRR@${options.topK}`, 10),
    pad("MeanRel", 10),
    pad("AvgMs", 10)
  ];
  console.log(headerParts.join(" "));

  for (const evaluation of evaluations) {
    const row = [
      pad(evaluation.mode, modeWidth),
      ...thresholds.map((threshold) => pad(percentage(evaluation.summary.hitsByThreshold[threshold], cases.length), 10)),
      pad(evaluation.summary.meanReciprocalRank.toFixed(3), 10),
      pad(evaluation.summary.meanRelevantReturned.toFixed(2), 10),
      pad(formatMs(averageTimingMs(evaluation.timingTotals, "eval.case_total") ?? 0), 10)
    ];
    console.log(row.join(" "));
  }

  console.log("");
  console.log("Per Question Comparison");
  for (const testCase of cases) {
    console.log(`${testCase.id}  ${testCase.question}`);
    for (const evaluation of evaluations) {
      const item = evaluation.scoredCases.find((candidate) => candidate.id === testCase.id);
      console.log(
        `  ${pad(evaluation.mode, modeWidth)}  hit=${item.retrieval.hit ? "yes" : "no"}  rr=${item.retrieval.reciprocalRank.toFixed(3)}  ranks=${item.retrieval.relevantRanks.join(",") || "-"}`
      );
    }
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const [store, cases] = await Promise.all([
    VectorStore.load(storagePath),
    loadEvalCases(options.evalPath)
  ]);
  const embedderConfig = serializeEmbedder(store.embedder);

  if (options.mode === "compare") {
    const evaluations = [];

    for (const mode of RETRIEVAL_COMPARE_MODES) {
      evaluations.push(await runEvaluation(store, cases, options, mode));
    }

    printCompareEvaluation(evaluations, cases, options, embedderConfig, options.evalPath);
    return;
  }

  if (options.mode === "all") {
    const evaluations = [];

    for (const mode of RETRIEVAL_COMPARE_MODES) {
      evaluations.push(await runEvaluation(store, cases, options, mode));
    }

    printCompareEvaluation(evaluations, cases, options, embedderConfig, options.evalPath);
    console.log("");
    console.log("Answer Evaluation");
    const answerEvaluation = await runEvaluation(store, cases, options, "answer");
    printSingleModeEvaluation(answerEvaluation, cases, options, embedderConfig, options.evalPath);
    return;
  }

  const evaluation = await runEvaluation(store, cases, options, options.mode);
  printSingleModeEvaluation(evaluation, cases, options, embedderConfig, options.evalPath);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
