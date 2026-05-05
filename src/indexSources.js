import path from "node:path";
import { fileURLToPath } from "node:url";

import { createEmbedderFromConfig, serializeEmbedder } from "./embedderFactory.js";
import { loadSourceDocuments } from "./sourceLoader.js";
import { DEFAULT_OPENAI_EMBEDDING_MODEL } from "./openaiEmbedder.js";
import { VectorStore } from "./vectorStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const storagePath = path.resolve(__dirname, "../data/vector-store.json");
const sourcesDir = path.resolve(__dirname, "../sources");

function parseArgs(argv) {
  const config = {
    embedder: "hashing",
    embeddingModel: DEFAULT_OPENAI_EMBEDDING_MODEL,
    dimensions: null
  };

  for (const arg of argv) {
    if (arg.startsWith("embedder=") || arg.startsWith("--embedder=")) {
      config.embedder = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("embeddingModel=") || arg.startsWith("--embeddingModel=")) {
      config.embeddingModel = arg.split("=")[1];
      continue;
    }

    if (arg.startsWith("dimensions=") || arg.startsWith("--dimensions=")) {
      config.dimensions = Number(arg.split("=")[1]);
    }
  }

  return config;
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  const documents = await loadSourceDocuments(sourcesDir);

  if (documents.length === 0) {
    throw new Error(`No PDF files found in ${sourcesDir}`);
  }

  const store = new VectorStore({
    embedder: createEmbedderFromConfig(
      config.embedder === "openai"
        ? {
            type: "openai",
            model: config.embeddingModel,
            ...(Number.isFinite(config.dimensions) ? { dimensions: config.dimensions } : {})
          }
        : {
            type: "hashing"
          }
    ),
    storagePath,
    chunkSize: 120,
    overlap: 30
  });
  const embedderInfo = serializeEmbedder(store.embedder);
  console.log(
    `Indexing with embedder=${embedderInfo.type}${embedderInfo.model ? ` model=${embedderInfo.model}` : ""}${embedderInfo.dimensions ? ` dimensions=${embedderInfo.dimensions}` : ""}`
  );

  for (const document of documents) {
    const chunks = await store.addDocument(document);
    console.log(`Indexed ${document.metadata.sourceFile} into ${chunks.length} chunk(s)`);
    console.log(
      `  title=${document.metadata.sourceTitle} author=${document.metadata.sourceAuthor} language=${document.metadata.sourceLanguage}`
    );
  }

  await store.save();
  console.log(`Saved vector store to ${storagePath}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
