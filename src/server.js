import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

import { serializeEmbedder } from "./embedderFactory.js";
import { normalizeQueryOptions, runQuery } from "./queryPipeline.js";
import { VectorStore } from "./vectorStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const publicDir = path.resolve(__dirname, "../public");
const storagePath = path.resolve(__dirname, "../data/vector-store.json");
const DEFAULT_HOST = process.env.HOST ?? "127.0.0.1";
const DEFAULT_PORT = Number(process.env.PORT ?? 4173);
const MODES = ["vector", "bm25", "hybrid", "rerank", "answer"];
const CONTENT_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8"
};

let storePromise = null;
let storeLoadedAt = null;

function createStoreSummary(store) {
  return {
    chunkCount: store.records.length,
    chunking: {
      chunkSize: store.chunkSize,
      overlap: store.overlap
    },
    documentCount: store.listDocuments().length,
    embedder: serializeEmbedder(store.embedder),
    loadedAt: storeLoadedAt,
    storagePath
  };
}

async function loadStoreFromDisk() {
  const store = await VectorStore.load(storagePath);
  storeLoadedAt = new Date().toISOString();
  return store;
}

async function getStore(forceReload = false) {
  if (!storePromise || forceReload) {
    storePromise = loadStoreFromDisk();
  }

  return storePromise;
}

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, {
    "Cache-Control": "no-store",
    "Content-Type": CONTENT_TYPES[".json"]
  });
  response.end(JSON.stringify(payload, null, 2));
}

function sendError(response, statusCode, message, extra = {}) {
  sendJson(response, statusCode, {
    error: message,
    ...extra
  });
}

function getContentType(filePath) {
  return CONTENT_TYPES[path.extname(filePath)] ?? "application/octet-stream";
}

async function sendStaticFile(response, pathname) {
  const candidatePath = pathname === "/"
    ? path.join(publicDir, "index.html")
    : path.normalize(path.join(publicDir, pathname));

  if (!candidatePath.startsWith(publicDir)) {
    sendError(response, 403, "Forbidden");
    return;
  }

  try {
    const file = await readFile(candidatePath);
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Type": getContentType(candidatePath)
    });
    response.end(file);
  } catch (error) {
    if (error.code === "ENOENT") {
      sendError(response, 404, "Not found");
      return;
    }

    throw error;
  }
}

async function readJsonBody(request) {
  const chunks = [];

  for await (const chunk of request) {
    chunks.push(chunk);
  }

  const body = Buffer.concat(chunks).toString("utf8").trim();
  if (!body) {
    return {};
  }

  try {
    return JSON.parse(body);
  } catch (error) {
    throw new Error(`Request body was not valid JSON: ${error.message}`);
  }
}

function buildSuccessfulQueryPayload(store, query, options, execution, totalMs) {
  const payload = {
    mode: options.mode,
    options,
    query,
    store: createStoreSummary(store),
    timings: [
      ...(execution.timings ?? []),
      {
        label: "api.total",
        ms: totalMs
      }
    ]
  };

  if (options.mode === "answer") {
    return {
      ...payload,
      answer: {
        insufficientContext: execution.answer.insufficientContext,
        model: execution.answer.model,
        responseId: execution.answer.responseId,
        sources: execution.answer.sources,
        text: execution.answer.answer
      },
      chunks: execution.chunks
    };
  }

  return {
    ...payload,
    results: execution.results
  };
}

async function handleQuery(request, response) {
  const body = await readJsonBody(request);
  const query = typeof body.query === "string" ? body.query.trim() : "";

  if (!query) {
    sendError(response, 400, "Query is required.");
    return;
  }

  const options = normalizeQueryOptions(body.options ?? {});
  const store = await getStore();
  const totalStart = performance.now();
  const execution = await runQuery(store, query, options);

  sendJson(
    response,
    200,
    buildSuccessfulQueryPayload(store, query, options, execution, performance.now() - totalStart)
  );
}

async function handleStatus(response) {
  const store = await getStore();

  sendJson(response, 200, {
    modes: MODES,
    ok: true,
    store: createStoreSummary(store)
  });
}

async function handleReload(response) {
  const store = await getStore(true);

  sendJson(response, 200, {
    ok: true,
    store: createStoreSummary(store)
  });
}

async function handleRequest(request, response) {
  const url = new URL(request.url, `http://${request.headers.host ?? `${DEFAULT_HOST}:${DEFAULT_PORT}`}`);

  if (request.method === "GET" && url.pathname === "/api/status") {
    await handleStatus(response);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/query") {
    await handleQuery(request, response);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/reload") {
    await handleReload(response);
    return;
  }

  if (request.method === "GET") {
    await sendStaticFile(response, url.pathname);
    return;
  }

  sendError(response, 405, "Method not allowed");
}

async function main() {
  try {
    await getStore();
  } catch (error) {
    if (error.code === "ENOENT") {
      throw new Error('Vector store not found. Run "npm run index:sources" or "npm run index:sources:openai" first.');
    }

    throw error;
  }

  const server = createServer((request, response) => {
    handleRequest(request, response).catch((error) => {
      sendError(response, 500, error.message);
    });
  });

  server.listen(DEFAULT_PORT, DEFAULT_HOST, () => {
    console.log(`UI server running at http://${DEFAULT_HOST}:${DEFAULT_PORT}`);
  });
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
