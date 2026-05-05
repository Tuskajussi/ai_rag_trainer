import path from "node:path";
import { fileURLToPath } from "node:url";

import { createEmbedderFromConfig } from "./embedderFactory.js";
import { VectorStore } from "./vectorStore.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const storagePath = path.resolve(__dirname, "../data/vector-store.json");

const sampleDocuments = [
  {
    id: "refund-policy",
    metadata: {
      topic: "billing",
      product: "general"
    },
    text: `
      Customers can request a refund within 30 days of purchase.
      Refunds are reviewed by support within two business days.
      Annual plans may be prorated when required by contract terms.
      After approval, refunds usually appear on the original payment method in five to ten business days.
    `
  },
  {
    id: "enterprise-auth",
    metadata: {
      topic: "authentication",
      product: "enterprise"
    },
    text: `
      Enterprise plans support single sign-on with SAML and SCIM provisioning.
      Administrators can enforce multi-factor authentication and configure domain verification.
      Audit logs are available for security investigations and compliance reviews.
    `
  },
  {
    id: "retrieval-notes",
    metadata: {
      topic: "ai",
      product: "internal"
    },
    text: `
      Retrieval systems usually separate indexing, retrieval, reranking, and answer generation.
      Chunk size and metadata filters strongly affect result quality.
      If the first-stage retrieval pulls too little context, the model may hallucinate or say the right thing for the wrong reason.
      Reranking is often used after retrieval to improve the ordering of candidate chunks.
    `
  }
];

async function main() {
  const store = new VectorStore({
    embedder: createEmbedderFromConfig({ type: "hashing" }),
    storagePath,
    chunkSize: 40,
    overlap: 10
  });

  for (const document of sampleDocuments) {
    const chunks = await store.addDocument(document);
    console.log(`Indexed ${document.id} into ${chunks.length} chunk(s)`);
  }

  await store.save();
  console.log(`Saved vector store to ${storagePath}`);

  const queries = [
    "How long do refunds take?",
    "Can enterprise customers use SSO?",
    "What does reranking do in retrieval?"
  ];

  for (const query of queries) {
    const results = await store.search(query, { topK: 2 });
    console.log(`\nQuery: ${query}`);

    results.forEach((result, index) => {
      console.log(
        `${index + 1}. score=${result.score.toFixed(3)} doc=${result.documentId} chunk=${result.chunkIndex}`
      );
      console.log(`   ${result.text}`);
    });
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
