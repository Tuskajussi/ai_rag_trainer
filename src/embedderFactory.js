import { HashingEmbedder } from "./embedder.js";
import { OpenAIEmbedder } from "./openaiEmbedder.js";

export function createEmbedderFromConfig(config = {}) {
  const type = config.type ?? "hashing";

  switch (type) {
    case "hashing":
      return new HashingEmbedder({
        dimensions: config.dimensions
      });
    case "openai":
      return new OpenAIEmbedder({
        model: config.model,
        dimensions: config.dimensions,
        baseUrl: config.baseUrl
      });
    default:
      throw new Error(`Unsupported embedder type "${type}"`);
  }
}

export function serializeEmbedder(embedder) {
  if (typeof embedder.serialize === "function") {
    return embedder.serialize();
  }

  return {
    type: embedder.type ?? "unknown"
  };
}
