export function chunkText(text, options = {}) {
  const {
    chunkSize = 80,
    overlap = 20
  } = options;

  if (chunkSize <= 0) {
    throw new Error("chunkSize must be greater than 0");
  }

  if (overlap < 0 || overlap >= chunkSize) {
    throw new Error("overlap must be >= 0 and less than chunkSize");
  }

  const words = text
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .filter(Boolean);

  if (words.length === 0) {
    return [];
  }

  const chunks = [];
  const step = chunkSize - overlap;

  for (let start = 0; start < words.length; start += step) {
    const slice = words.slice(start, start + chunkSize);
    if (slice.length === 0) {
      break;
    }

    chunks.push(slice.join(" "));

    if (start + chunkSize >= words.length) {
      break;
    }
  }

  return chunks;
}
