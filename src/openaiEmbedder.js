import { loadProjectEnv } from "./env.js";

const DEFAULT_BASE_URL = "https://api.openai.com";
const DEFAULT_BATCH_SIZE = 64;
const DEFAULT_MODEL = "text-embedding-3-small";

function buildErrorMessage(status, payload) {
  const apiMessage =
    payload?.error?.message ??
    payload?.message ??
    payload?.error ??
    "Unknown OpenAI API error";

  return `OpenAI embeddings request failed (${status}): ${apiMessage}`;
}

export class OpenAIEmbedder {
  constructor(options = {}) {
    this.type = "openai";
    this.model = options.model ?? DEFAULT_MODEL;
    this.dimensions = options.dimensions ?? null;
    this.baseUrl = options.baseUrl ?? null;
    this.batchSize = options.batchSize ?? DEFAULT_BATCH_SIZE;
  }

  async embed(text) {
    const [vector] = await this.embedBatch([text]);
    return vector;
  }

  async embedBatch(texts) {
    const inputs = texts
      .map((text) => text?.trim?.() ?? "")
      .filter((text) => text.length > 0);

    if (inputs.length !== texts.length) {
      throw new Error("OpenAI embeddings require non-empty text inputs");
    }

    await loadProjectEnv();

    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      throw new Error('OPENAI_API_KEY is required for the OpenAI embedder. Put it in .env or set it in PowerShell with: $env:OPENAI_API_KEY="your-key"');
    }

    const baseUrl = (this.baseUrl ?? process.env.OPENAI_BASE_URL ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    const vectors = [];

    for (let index = 0; index < inputs.length; index += this.batchSize) {
      const batch = inputs.slice(index, index + this.batchSize);
      const response = await fetch(`${baseUrl}/v1/embeddings`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          input: batch,
          model: this.model,
          encoding_format: "float",
          ...(typeof this.dimensions === "number" ? { dimensions: this.dimensions } : {})
        })
      });

      const payload = await response.json().catch(() => null);
      if (!response.ok || payload?.error) {
        throw new Error(buildErrorMessage(response.status, payload));
      }

      for (const item of payload.data ?? []) {
        vectors.push(item.embedding);
      }
    }

    return vectors;
  }

  serialize() {
    return {
      type: this.type,
      model: this.model,
      ...(typeof this.dimensions === "number" ? { dimensions: this.dimensions } : {})
    };
  }
}

export {
  DEFAULT_MODEL as DEFAULT_OPENAI_EMBEDDING_MODEL
};
