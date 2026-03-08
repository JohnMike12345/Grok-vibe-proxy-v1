import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

const PORT = process.env.PORT || 10000;
const DEFAULT_MODEL = process.env.DEFAULT_MODEL || "grok-proxy";
const UPSTREAM_URL = process.env.UPSTREAM_URL || "";
const UPSTREAM_AUTH_HEADER = process.env.UPSTREAM_AUTH_HEADER || "Authorization";
const UPSTREAM_AUTH_TOKEN = process.env.UPSTREAM_AUTH_TOKEN || "";
const UPSTREAM_TIMEOUT_MS = Number(process.env.UPSTREAM_TIMEOUT_MS || 120000);

app.get("/healthz", (_req, res) => {
  res.json({ ok: true });
});

app.get("/v1/models", (_req, res) => {
  res.json({
    object: "list",
    data: [
      {
        id: DEFAULT_MODEL,
        object: "model",
        created: Math.floor(Date.now() / 1000),
        owned_by: "proxy"
      }
    ]
  });
});

app.post("/v1/chat/completions", async (req, res) => {
  try {
    if (!UPSTREAM_URL) {
      return res.status(500).json(openAIError("UPSTREAM_URL is not set", "proxy_config_error"));
    }

    const body = req.body || {};
    const stream = !!body.stream;
    const model = body.model || DEFAULT_MODEL;
    const requestId = `chatcmpl-${cryptoRandom()}`;

    const upstreamBody = buildUpstreamRequest(body);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

    const headers = {
      "Content-Type": "application/json"
    };

    // Optional upstream auth. If not needed, leave env vars empty.
    if (UPSTREAM_AUTH_TOKEN) {
      headers[UPSTREAM_AUTH_HEADER] = UPSTREAM_AUTH_TOKEN;
    }

    const upstreamResp = await fetch(UPSTREAM_URL, {
      method: "POST",
      headers,
      body: JSON.stringify(upstreamBody),
      signal: controller.signal
    });

    clearTimeout(timeout);

    if (!upstreamResp.ok) {
      const text = await upstreamResp.text().catch(() => "");
      return res.status(upstreamResp.status).json(
        openAIError(`Upstream error: ${text || upstreamResp.statusText}`, "upstream_error")
      );
    }

    if (stream) {
      return streamAsOpenAI(upstreamResp, res, requestId, model);
    } else {
      return nonStreamAsOpenAI(upstreamResp, res, requestId, model);
    }
  } catch (err) {
    const message = err?.name === "AbortError" ? "Upstream request timed out" : (err?.message || "Internal proxy error");
    return res.status(500).json(openAIError(message, "proxy_error"));
  }
});

/**
 * IMPORTANT:
 * Replace this function with the REAL upstream request body format.
 *
 * Right now it creates a generic structure from OpenAI-style messages.
 * Since your sample shows the response format, not the exact request format,
 * this part is still a placeholder you must adapt.
 */
function buildUpstreamRequest(openaiBody) {
  const messages = Array.isArray(openaiBody.messages) ? openaiBody.messages : [];
  const normalized = messages.map((m) => ({
    role: m.role,
    content: flattenContent(m.content)
  }));

  return {
    model: openaiBody.model || DEFAULT_MODEL,
    stream: !!openaiBody.stream,
    temperature: openaiBody.temperature,
    messages: normalized
  };
}

async function nonStreamAsOpenAI(upstreamResp, res, requestId, model) {
  const raw = await upstreamResp.text();
  const objects = parseConcatenatedJsonObjects(raw);

  // Prefer final assembled message if present
  let finalMessage =
    objects
      .map((o) => o?.result?.modelResponse?.message)
      .find((x) => typeof x === "string" && x.length > 0) || null;

  // Fallback: rebuild from tokens
  if (!finalMessage) {
    finalMessage = objects
      .map((o) => o?.result?.token)
      .filter((x) => typeof x === "string")
      .join("");
  }

  const response = {
    id: requestId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: finalMessage || ""
        },
        finish_reason: "stop"
      }
    ],
    usage: {
      prompt_tokens: 0,
      completion_tokens: 0,
      total_tokens: 0
    }
  };

  res.json(response);
}

async function streamAsOpenAI(upstreamResp, res, requestId, model) {
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");

  const reader = upstreamResp.body.getReader();
  const decoder = new TextDecoder();

  let buffer = "";
  let sentRole = false;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const { objects, remainder } = extractJsonObjectsFromBuffer(buffer);
      buffer = remainder;

      for (const obj of objects) {
        const token = obj?.result?.token;
        const finalMessage = obj?.result?.modelResponse?.message;

        if (!sentRole) {
          writeSSE(res, {
            id: requestId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                index: 0,
                delta: { role: "assistant" },
                finish_reason: null
              }
            ]
          });
          sentRole = true;
        }

        if (typeof token === "string" && token.length > 0) {
          writeSSE(res, {
            id: requestId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                index: 0,
                delta: { content: token },
                finish_reason: null
              }
            ]
          });
        }

        // If upstream sends a final assembled message but token stream was empty,
        // you could emit it here. Usually not necessary if tokens already arrived.
        if (
          typeof finalMessage === "string" &&
          finalMessage.length > 0 &&
          !objects.some((x) => typeof x?.result?.token === "string")
        ) {
          writeSSE(res, {
            id: requestId,
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model,
            choices: [
              {
                index: 0,
                delta: { content: finalMessage },
                finish_reason: null
              }
            ]
          });
        }
      }
    }

    writeSSE(res, {
      id: requestId,
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          delta: {},
          finish_reason: "stop"
        }
      ]
    });

    res.write("data: [DONE]\n\n");
    res.end();
  } catch (err) {
    writeSSE(res, {
      error: {
        message: err?.message || "Streaming proxy error",
        type: "server_error",
        code: "stream_proxy_error"
      }
    });
    res.write("data: [DONE]\n\n");
    res.end();
  }
}

function writeSSE(res, obj) {
  res.write(`data: ${JSON.stringify(obj)}\n\n`);
}

function flattenContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part?.type === "text") return part.text || "";
        return "";
      })
      .join("");
  }
  return "";
}

function openAIError(message, code = "proxy_error") {
  return {
    error: {
      message,
      type: "invalid_request_error",
      param: null,
      code
    }
  };
}

/**
 * Parses concatenated JSON objects like:
 * {}{}{} or separated by newlines.
 */
function parseConcatenatedJsonObjects(text) {
  const { objects } = extractJsonObjectsFromBuffer(text);
  return objects;
}

function extractJsonObjectsFromBuffer(input) {
  const objects = [];
  let start = -1;
  let depth = 0;
  let inString = false;
  let escape = false;
  let lastConsumedIndex = 0;

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];

    if (inString) {
      if (escape) {
        escape = false;
      } else if (ch === "\\") {
        escape = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }

    if (ch === '"') {
      inString = true;
      continue;
    }

    if (ch === "{") {
      if (depth === 0) start = i;
      depth++;
      continue;
    }

    if (ch === "}") {
      depth--;
      if (depth === 0 && start !== -1) {
        const candidate = input.slice(start, i + 1);
        try {
          objects.push(JSON.parse(candidate));
          lastConsumedIndex = i + 1;
          start = -1;
        } catch {
          // ignore malformed candidate
        }
      }
    }
  }

  const remainder = input.slice(lastConsumedIndex);
  return { objects, remainder };
}

function cryptoRandom() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Proxy listening on port ${PORT}`);
});
