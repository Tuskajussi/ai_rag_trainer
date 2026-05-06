const form = document.querySelector("#query-form");
const statusNodes = {
  chunking: document.querySelector("#status-chunking"),
  chunks: document.querySelector("#status-chunks"),
  documents: document.querySelector("#status-documents"),
  embedder: document.querySelector("#status-embedder")
};
const reloadButton = document.querySelector("#reload-store");
const runState = document.querySelector("#run-state");
const timingsRoot = document.querySelector("#timings");
const resultsRoot = document.querySelector("#results");
const resultsTitle = document.querySelector("#results-title");
const resultsSubtitle = document.querySelector("#results-subtitle");
const answerPanel = document.querySelector("#answer-panel");
const answerText = document.querySelector("#answer-text");
const answerMeta = document.querySelector("#answer-meta");
const answerNote = document.querySelector("#answer-note");
const sourcesPanel = document.querySelector("#sources-panel");
const sourcesRoot = document.querySelector("#sources");
const timingTemplate = document.querySelector("#timing-template");
const chunkTemplate = document.querySelector("#chunk-template");

function formatMs(value) {
  const numeric = Number(value ?? 0);

  if (numeric >= 1000) {
    return `${numeric.toFixed(1)} ms`;
  }

  if (numeric >= 10) {
    return `${numeric.toFixed(2)} ms`;
  }

  return `${numeric.toFixed(3)} ms`;
}

function compactWhitespace(value) {
  return String(value ?? "").replace(/\s+/g, " ").trim();
}

function truncate(value, maxLength = 340) {
  const normalized = compactWhitespace(value);

  if (normalized.length <= maxLength) {
    return normalized;
  }

  return `${normalized.slice(0, maxLength - 3)}...`;
}

function parseFilter(value) {
  const filter = {};

  for (const rawEntry of String(value ?? "").split(/[;\n]+/)) {
    const entry = rawEntry.trim();
    if (!entry || !entry.includes("=")) {
      continue;
    }

    const [rawKey, ...rawValue] = entry.split("=");
    const key = rawKey.trim();
    const resolvedValue = rawValue.join("=").trim();

    if (key && resolvedValue) {
      filter[key] = resolvedValue;
    }
  }

  return filter;
}

function renderStatus(status) {
  statusNodes.documents.textContent = `${status.documentCount} document(s)`;
  statusNodes.chunks.textContent = `${status.chunkCount} chunk(s)`;
  statusNodes.embedder.textContent = status.embedder.model
    ? `${status.embedder.type} / ${status.embedder.model}`
    : status.embedder.type;
  statusNodes.chunking.textContent = `${status.chunking.chunkSize}w with ${status.chunking.overlap}w overlap`;
}

function setRunState(text, mode = "idle") {
  runState.textContent = text;
  runState.dataset.mode = mode;
}

function clearPanels() {
  timingsRoot.classList.add("empty-state");
  timingsRoot.textContent = "Run a query to see the pipeline timings.";

  resultsRoot.classList.add("empty-state");
  resultsRoot.textContent = "Run a query to inspect retrieved chunks, ranks, scores, and metadata.";

  answerPanel.classList.add("hidden");
  sourcesPanel.classList.add("hidden");
  answerText.textContent = "";
  answerNote.classList.add("hidden");
  answerNote.textContent = "";
  sourcesRoot.innerHTML = "";
}

function buildChips(result) {
  const chips = [];

  if (Array.isArray(result.selectedBy) && result.selectedBy.length > 0) {
    chips.push(...result.selectedBy.map((value) => ({ kind: "route", label: value })));
  }

  if (typeof result.vectorRank === "number") {
    chips.push({ kind: "vector", label: `vector #${result.vectorRank}` });
  }

  if (typeof result.bm25Rank === "number") {
    chips.push({ kind: "bm25", label: `bm25 #${result.bm25Rank}` });
  }

  if (typeof result.rerankScore === "number") {
    chips.push({ kind: "rerank", label: `rerank ${result.rerankScore.toFixed(3)}` });
  }

  return chips;
}

function renderChunks(items, title, subtitle) {
  resultsTitle.textContent = title;
  resultsSubtitle.textContent = subtitle;
  resultsRoot.classList.remove("empty-state");
  resultsRoot.innerHTML = "";

  if (!Array.isArray(items) || items.length === 0) {
    resultsRoot.classList.add("empty-state");
    resultsRoot.textContent = "No chunks returned.";
    return;
  }

  for (const item of items) {
    const fragment = chunkTemplate.content.cloneNode(true);
    const card = fragment.querySelector(".chunk-card");
    const score = fragment.querySelector(".score-pill");
    const titleNode = fragment.querySelector(".chunk-title");
    const subtitleNode = fragment.querySelector(".chunk-subtitle");
    const chipRow = fragment.querySelector(".chip-row");
    const textNode = fragment.querySelector(".chunk-text");
    const metaNode = fragment.querySelector("pre");

    titleNode.textContent = `${item.documentId}#${item.chunkIndex}`;
    subtitleNode.textContent = item.metadata?.sourceTitle
      ? `${item.metadata.sourceTitle}${item.metadata.sourceAuthor ? ` · ${item.metadata.sourceAuthor}` : ""}`
      : "Indexed chunk";
    score.textContent = typeof item.score === "number" ? item.score.toFixed(3) : "n/a";
    textNode.textContent = truncate(item.text);
    metaNode.textContent = JSON.stringify(item.metadata, null, 2);

    for (const chip of buildChips(item)) {
      const chipNode = document.createElement("span");
      chipNode.className = `chip chip-${chip.kind}`;
      chipNode.textContent = chip.label;
      chipRow.append(chipNode);
    }

    card.dataset.mode = item.selectedBy?.includes("bm25") ? "bm25" : "vector";
    resultsRoot.append(fragment);
  }
}

function renderTimings(timings) {
  timingsRoot.classList.remove("empty-state");
  timingsRoot.innerHTML = "";

  if (!Array.isArray(timings) || timings.length === 0) {
    timingsRoot.classList.add("empty-state");
    timingsRoot.textContent = "No timings returned.";
    return;
  }

  const max = Math.max(...timings.map((entry) => Number(entry.ms ?? 0)), 1);

  for (const entry of timings) {
    const fragment = timingTemplate.content.cloneNode(true);
    const labelNode = fragment.querySelector(".timing-label");
    const barNode = fragment.querySelector(".timing-bar span");
    const valueNode = fragment.querySelector(".timing-value");

    labelNode.textContent = entry.label;
    valueNode.textContent = formatMs(entry.ms);
    barNode.style.width = `${Math.max((Number(entry.ms ?? 0) / max) * 100, 3)}%`;
    timingsRoot.append(fragment);
  }
}

function renderSources(sources) {
  sourcesRoot.innerHTML = "";

  if (!Array.isArray(sources) || sources.length === 0) {
    sourcesRoot.innerHTML = '<p class="empty-inline">No cited sources returned by the model.</p>';
    return;
  }

  for (const source of sources) {
    const card = document.createElement("article");
    card.className = "source-card";

    const heading = document.createElement("h3");
    heading.textContent = `${source.documentId}#${source.chunkIndex}`;
    card.append(heading);

    const summary = document.createElement("p");
    summary.className = "source-summary";
    summary.textContent = source.metadata?.sourceTitle
      ? `${source.metadata.sourceTitle}${source.metadata.sourceAuthor ? ` · ${source.metadata.sourceAuthor}` : ""}`
      : "Indexed source chunk";
    card.append(summary);

    if (source.metadata?.sourceUri) {
      const link = document.createElement("a");
      link.href = source.metadata.sourceUri;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = source.metadata.sourceUri;
      card.append(link);
    }

    const snippet = document.createElement("p");
    snippet.className = "source-snippet";
    snippet.textContent = truncate(source.text, 280);
    card.append(snippet);

    sourcesRoot.append(card);
  }
}

async function fetchStatus() {
  const response = await fetch("/api/status");
  const payload = await response.json();

  if (!response.ok) {
    throw new Error(payload.error ?? "Failed to load store status.");
  }

  renderStatus(payload.store);
}

function buildPayload() {
  const formData = new FormData(form);

  return {
    query: String(formData.get("query") ?? "").trim(),
    options: {
      answerMaxTokens: Number(formData.get("answerMaxTokens") ?? 700),
      answerModel: String(formData.get("answerModel") ?? "gpt-5.4-mini"),
      bm25K: Number(formData.get("bm25K") ?? 10),
      candidateTopK: Number(formData.get("candidateTopK") ?? 10),
      filter: parseFilter(formData.get("filter")),
      finalBm25Min: Number(formData.get("finalBm25Min") ?? 1),
      finalVectorMin: Number(formData.get("finalVectorMin") ?? 2),
      mode: String(formData.get("mode") ?? "answer"),
      topK: Number(formData.get("topK") ?? 5),
      vectorK: Number(formData.get("vectorK") ?? 10)
    }
  };
}

async function runQuery(event) {
  event.preventDefault();
  const payload = buildPayload();

  if (!payload.query) {
    setRunState("Question required", "error");
    return;
  }

  setRunState("Running…", "busy");
  clearPanels();

  try {
    const response = await fetch("/api/query", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error ?? "Query failed.");
    }

    renderTimings(result.timings);

    if (result.mode === "answer") {
      answerPanel.classList.remove("hidden");
      sourcesPanel.classList.remove("hidden");
      answerText.textContent = result.answer.text;
      answerMeta.textContent = `Model: ${result.answer.model} · Selected chunks: ${result.chunks.length}`;
      renderSources(result.answer.sources);
      renderChunks(
        result.chunks,
        "Selected Context",
        "These are the chunks that were actually sent to the answer model."
      );

      if (result.answer.insufficientContext) {
        answerNote.classList.remove("hidden");
        answerNote.textContent = "The answer model reported that the provided chunks may not fully support the answer.";
      }
    } else {
      renderChunks(
        result.results,
        "Retrieved Chunks",
        "Inspect the ranking signals before deciding whether the answer stage should run."
      );
    }

    setRunState("Complete", "success");
  } catch (error) {
    setRunState(error.message, "error");
    resultsRoot.classList.remove("empty-state");
    resultsRoot.textContent = error.message;
  }
}

async function reloadStore() {
  setRunState("Reloading store…", "busy");

  try {
    const response = await fetch("/api/reload", { method: "POST" });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.error ?? "Failed to reload store.");
    }

    renderStatus(payload.store);
    setRunState("Store reloaded", "success");
  } catch (error) {
    setRunState(error.message, "error");
  }
}

function attachExampleButtons() {
  for (const button of document.querySelectorAll(".example-chip")) {
    button.addEventListener("click", () => {
      document.querySelector("#query").value = button.dataset.query ?? "";
    });
  }
}

form.addEventListener("submit", runQuery);
reloadButton.addEventListener("click", reloadStore);
attachExampleButtons();
clearPanels();
fetchStatus().catch((error) => {
  setRunState(error.message, "error");
});
