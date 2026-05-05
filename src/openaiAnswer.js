import { loadProjectEnv } from "./env.js";
import { performance } from "node:perf_hooks";

const DEFAULT_ANSWER_MODEL = "gpt-5.4-mini";
const DEFAULT_MAX_OUTPUT_TOKENS = 700;
const DEFAULT_BASE_URL = "https://api.openai.com";

function compactWhitespace(value) {
  return value.replace(/\s+/g, " ").trim();
}

function truncate(value, maxLength = 220) {
  const normalized = compactWhitespace(value);

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 3)}...`;
}

function buildChunkLabel(chunk) {
  return `${chunk.documentId}#${chunk.chunkIndex}`;
}

function buildChunkContext(chunk) {
  const metadata = {
    sourceAuthor: chunk.metadata.sourceAuthor ?? null,
    sourceFile: chunk.metadata.sourceFile ?? null,
    sourceTitle: chunk.metadata.sourceTitle ?? null,
    sourceUri: chunk.metadata.sourceUri ?? null,
    selectedBy: chunk.selectedBy ?? [],
    bm25Rank: chunk.bm25Rank ?? null,
    vectorRank: chunk.vectorRank ?? null,
    rerankScore: chunk.rerankScore ?? null
  };

  return [
    `ID: ${buildChunkLabel(chunk)}`,
    `Document: ${chunk.documentId}`,
    `Chunk index: ${chunk.chunkIndex}`,
    `Metadata: ${JSON.stringify(metadata)}`,
    "Text:",
    chunk.text
  ].join("\n");
}

function buildPrompt(query, chunks) {
  const context = chunks
    .map((chunk, index) => `### Source ${index + 1}\n${buildChunkContext(chunk)}`)
    .join("\n\n");

  return [
    `Question: ${query}`,
    "",
    "Use only the provided source chunks to answer the question.",
    "If the sources do not contain enough evidence, say so plainly.",
    "Cite only chunk IDs that directly support the answer.",
    "",
    "Sources:",
    context
  ].join("\n");
}

function extractOutputText(responseBody) {
  if (typeof responseBody.output_text === "string" && responseBody.output_text.trim()) {
    return responseBody.output_text;
  }

  const fragments = [];

  for (const item of responseBody.output ?? []) {
    if (!Array.isArray(item.content)) {
      continue;
    }

    for (const contentItem of item.content) {
      if (
        (contentItem.type === "output_text" || contentItem.type === "text") &&
        typeof contentItem.text === "string"
      ) {
        fragments.push(contentItem.text);
      }
    }
  }

  return fragments.join("\n").trim();
}

function parseModelJson(responseText) {
  try {
    return JSON.parse(responseText);
  } catch (error) {
    throw new Error(`OpenAI response was not valid JSON: ${error.message}`);
  }
}

function normalizeSourceIds(sourceIds, chunks) {
  const chunkById = new Map(chunks.map((chunk) => [buildChunkLabel(chunk), chunk]));
  const seen = new Set();
  const resolved = [];

  for (const sourceId of sourceIds) {
    const normalizedId = String(sourceId);
    if (seen.has(normalizedId) || !chunkById.has(normalizedId)) {
      continue;
    }

    seen.add(normalizedId);
    resolved.push(chunkById.get(normalizedId));
  }

  return resolved;
}

function createSchema() {
  return {
    type: "object",
    additionalProperties: false,
    properties: {
      answer: {
        type: "string",
        description: "Grounded answer written only from the provided sources."
      },
      insufficient_context: {
        type: "boolean",
        description: "True when the provided sources are not enough to answer confidently."
      },
      source_ids: {
        type: "array",
        description: "Chunk IDs that directly support the answer.",
        items: {
          type: "string"
        }
      }
    },
    required: ["answer", "insufficient_context", "source_ids"]
  };
}

function buildRequestBody(query, chunks, options) {
  return {
    model: options.model,
    store: false,
    input: buildPrompt(query, chunks),
    instructions: [
      "You are a grounded QA assistant.",
      "Answer only from the provided source chunks.",
      "Do not invent facts, pages, or sources.",
      "When the evidence is incomplete, say that the answer is not fully supported by the sources."
    ].join(" "),
    max_output_tokens: options.maxOutputTokens,
    text: {
      format: {
        type: "json_schema",
        name: "grounded_answer",
        strict: true,
        schema: createSchema()
      }
    }
  };
}

function buildErrorMessage(status, payload) {
  const apiMessage =
    payload?.error?.message ??
    payload?.message ??
    payload?.error ??
    "Unknown OpenAI API error";

  return `OpenAI API request failed (${status}): ${apiMessage}`;
}

export async function answerQuestionWithSources(query, chunks, options = {}) {
  if (!query || !query.trim()) {
    throw new Error("Query is required");
  }

  if (!Array.isArray(chunks) || chunks.length === 0) {
    throw new Error("At least one chunk is required to generate an answer");
  }

  const totalStart = performance.now();
  const envLoadStart = performance.now();
  await loadProjectEnv();
  const envLoadMs = performance.now() - envLoadStart;

  const apiKey = options.apiKey ?? process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('OPENAI_API_KEY is required for mode=answer. Put it in .env or set it in PowerShell with: $env:OPENAI_API_KEY="your-key"');
  }

  const model = options.model ?? DEFAULT_ANSWER_MODEL;
  const maxOutputTokens = options.maxOutputTokens ?? DEFAULT_MAX_OUTPUT_TOKENS;
  const baseUrl = (options.baseUrl ?? process.env.OPENAI_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  const promptBuildStart = performance.now();
  const requestBody = buildRequestBody(query, chunks, {
    model,
    maxOutputTokens
  });
  const promptBuildMs = performance.now() - promptBuildStart;

  const requestStart = performance.now();
  const response = await fetch(`${baseUrl}/v1/responses`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(requestBody)
  });
  const requestMs = performance.now() - requestStart;

  const jsonReadStart = performance.now();
  const payload = await response.json().catch(() => null);
  const jsonReadMs = performance.now() - jsonReadStart;

  if (!response.ok) {
    throw new Error(buildErrorMessage(response.status, payload));
  }

  if (payload?.error) {
    throw new Error(buildErrorMessage(response.status, payload));
  }

  const parseStart = performance.now();
  const outputText = extractOutputText(payload);
  if (!outputText) {
    throw new Error("OpenAI response did not contain any output text");
  }

  const parsed = parseModelJson(outputText);
  const answer = typeof parsed.answer === "string" ? parsed.answer.trim() : "";
  const insufficientContext = Boolean(parsed.insufficient_context);
  const citedChunks = normalizeSourceIds(parsed.source_ids ?? [], chunks);

  return {
    answer,
    insufficientContext,
    model,
    raw: parsed,
    responseId: payload.id ?? null,
    sources: citedChunks,
    timings: [
      {
        label: "openai.env_load",
        ms: envLoadMs
      },
      {
        label: "openai.prompt_build",
        ms: promptBuildMs
      },
      {
        label: "openai.http_request",
        ms: requestMs
      },
      {
        label: "openai.read_json",
        ms: jsonReadMs
      },
      {
        label: "openai.response_parse",
        ms: performance.now() - parseStart
      },
      {
        label: "openai.total",
        ms: performance.now() - totalStart
      }
    ]
  };
}

export function formatSourceLine(chunk) {
  const parts = [buildChunkLabel(chunk)];

  if (chunk.metadata.sourceTitle) {
    parts.push(chunk.metadata.sourceTitle);
  }

  if (chunk.metadata.sourceAuthor) {
    parts.push(chunk.metadata.sourceAuthor);
  }

  return parts.join(" | ");
}

export function formatSourceDetails(chunk) {
  const details = [];

  if (chunk.metadata.sourceUri) {
    details.push(`URI: ${chunk.metadata.sourceUri}`);
  }

  details.push(`Snippet: ${truncate(chunk.text)}`);
  return details;
}

export {
  DEFAULT_ANSWER_MODEL,
  DEFAULT_MAX_OUTPUT_TOKENS
};
