export function tokenize(text) {
  return text
    .toLowerCase()
    .match(/[\p{L}\p{N}]+/gu) || [];
}

export function countTermFrequencies(tokens) {
  const termFreq = Object.create(null);

  for (const token of tokens) {
    termFreq[token] = (termFreq[token] ?? 0) + 1;
  }

  return termFreq;
}
