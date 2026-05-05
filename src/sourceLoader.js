import { readFile, readdir } from "node:fs/promises";
import path from "node:path";

import { PDFParse } from "pdf-parse";

function addMetadataValue(metadata, key, value) {
  const existing = metadata[key];

  if (existing === undefined) {
    metadata[key] = value;
    return;
  }

  if (Array.isArray(existing)) {
    if (!existing.includes(value)) {
      existing.push(value);
    }
    return;
  }

  if (existing !== value) {
    metadata[key] = [existing, value];
  }
}

function getFirst(value) {
  if (Array.isArray(value)) {
    return value[0];
  }

  return value;
}

function asArray(value) {
  if (Array.isArray(value)) {
    return value;
  }

  if (value === undefined) {
    return [];
  }

  return [value];
}

function slugifyFileName(fileName) {
  return path
    .basename(fileName, path.extname(fileName))
    .normalize("NFKD")
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .toLowerCase();
}

function cleanPdfText(text) {
  return text
    .replace(/(\p{L})-\s+(\p{L})/gu, "$1$2")
    .replace(/--\s+\d+\s+of\s+\d+\s+--/g, " ")
    .replace(/\u0000/g, " ")
    .replace(/\r/g, "");
}

export async function parseMetadataFile(metaPath) {
  const raw = await readFile(metaPath, "utf8");
  const metadata = {};

  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) {
      continue;
    }

    const [key, value] = trimmed.split("\t");
    if (!key || !value || value === "-") {
      continue;
    }

    addMetadataValue(metadata, key, value.trim());
  }

  return metadata;
}

export function normalizeDocumentMetadata({ fileName, pdfInfo, rawMetadata }) {
  const subjects = asArray(rawMetadata["dc.subject.yso"]);
  const keywords = (pdfInfo.Keywords || "")
    .split(",")
    .map((keyword) => keyword.trim())
    .filter(Boolean);

  return {
    sourceType: "pdf",
    sourceFile: fileName,
    sourceTitle: getFirst(rawMetadata["dc.title"]) ?? pdfInfo.Title ?? null,
    sourceAuthor: getFirst(rawMetadata["dc.contributor.author"]) ?? pdfInfo.Author ?? null,
    sourceLanguage: getFirst(rawMetadata["dc.language.iso"]) ?? pdfInfo.Language ?? null,
    sourceUri: getFirst(rawMetadata["dc.identifier.uri"]) ?? null,
    sourceUrn: getFirst(rawMetadata["dc.identifier.urn"]) ?? null,
    sourceYear: getFirst(rawMetadata["dc.date.issued"]) ?? null,
    sourceContractor: getFirst(rawMetadata["dc.relation.contractor"]) ?? null,
    sourceDiscipline: getFirst(rawMetadata["dc.subject.discipline"]) ?? null,
    sourceSpecialization: getFirst(rawMetadata["dc.subject.specialization"]) ?? null,
    sourceDegreeProgram: getFirst(rawMetadata["dc.subject.degreeprogram"]) ?? null,
    sourceSubjects: subjects,
    sourceKeywords: keywords,
    sourcePageCount: pdfInfo.totalPages ?? null
  };
}

export async function extractPdfDocument(pdfPath, rawMetadata = {}) {
  const data = await readFile(pdfPath);
  const parser = new PDFParse({ data });

  try {
    const textResult = await parser.getText();
    const infoResult = await parser.getInfo();

    const fileName = path.basename(pdfPath);
    const documentId = slugifyFileName(fileName);
    const cleanedText = cleanPdfText(textResult.text);
    const normalizedMetadata = normalizeDocumentMetadata({
      fileName,
      pdfInfo: {
        ...infoResult.info,
        totalPages: textResult.total
      },
      rawMetadata
    });

    return {
      id: documentId,
      text: cleanedText,
      metadata: normalizedMetadata
    };
  } finally {
    await parser.destroy();
  }
}

export async function loadSourceDocuments(sourcesDir) {
  const entries = await readdir(sourcesDir, { withFileTypes: true });
  const pdfFiles = entries
    .filter((entry) => entry.isFile() && path.extname(entry.name).toLowerCase() === ".pdf")
    .map((entry) => path.join(sourcesDir, entry.name));

  const metaEntry = entries.find(
    (entry) => entry.isFile() && entry.name.toLowerCase() === "meta.txt"
  );
  const rawMetadata = metaEntry
    ? await parseMetadataFile(path.join(sourcesDir, metaEntry.name))
    : {};

  const documents = [];
  for (const pdfPath of pdfFiles) {
    documents.push(await extractPdfDocument(pdfPath, rawMetadata));
  }

  return documents;
}
