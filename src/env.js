import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DEFAULT_ENV_PATH = path.resolve(__dirname, "../.env");

let loadPromise = null;

function parseEnvValue(rawValue) {
  const value = rawValue.trim();

  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1).replace(/\\n/g, "\n");
  }

  return value;
}

function parseEnvFile(content) {
  const entries = [];

  for (const line of content.split(/\r?\n/u)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const normalized = trimmed.startsWith("export ")
      ? trimmed.slice("export ".length)
      : trimmed;
    const separatorIndex = normalized.indexOf("=");

    if (separatorIndex <= 0) {
      continue;
    }

    const key = normalized.slice(0, separatorIndex).trim();
    const rawValue = normalized.slice(separatorIndex + 1);
    if (!key) {
      continue;
    }

    entries.push([key, parseEnvValue(rawValue)]);
  }

  return entries;
}

export async function loadProjectEnv(envPath = DEFAULT_ENV_PATH) {
  if (!loadPromise) {
    loadPromise = (async () => {
      try {
        const content = await readFile(envPath, "utf8");
        const entries = parseEnvFile(content);

        for (const [key, value] of entries) {
          if (process.env[key] === undefined) {
            process.env[key] = value;
          }
        }
      } catch (error) {
        if (error.code !== "ENOENT") {
          throw error;
        }
      }
    })();
  }

  await loadPromise;
}

