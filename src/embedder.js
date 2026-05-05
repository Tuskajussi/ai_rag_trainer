import { tokenize } from "./tokenize.js";

function fnv1a(input) {
  let hash = 2166136261;

  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}

export class HashingEmbedder {
  constructor(options = {}) {
    this.type = "hashing";
    this.dimensions = options.dimensions ?? 128;
  }

  async embed(text) {
    return this.embedSync(text);
  }

  async embedBatch(texts) {
    return texts.map((text) => this.embedSync(text));
  }

  embedSync(text) {
    const vector = new Float32Array(this.dimensions);
    const tokens = tokenize(text);

    for (const token of tokens) {
      this.#addFeature(vector, `w:${token}`, 1.0);

      if (token.length >= 3) {
        for (let i = 0; i <= token.length - 3; i += 1) {
          const trigram = token.slice(i, i + 3);
          this.#addFeature(vector, `c:${trigram}`, 0.3);
        }
      }
    }

    return normalize(vector);
  }

  serialize() {
    return {
      type: this.type,
      dimensions: this.dimensions
    };
  }

  #addFeature(vector, feature, weight) {
    const hash = fnv1a(feature);
    const index = hash % this.dimensions;
    const sign = (fnv1a(`${feature}:sign`) & 1) === 0 ? 1 : -1;
    vector[index] += sign * weight;
  }
}

function normalize(vector) {
  let magnitude = 0;

  for (const value of vector) {
    magnitude += value * value;
  }

  const norm = Math.sqrt(magnitude);
  if (norm === 0) {
    return Array.from(vector);
  }

  return Array.from(vector, (value) => value / norm);
}
