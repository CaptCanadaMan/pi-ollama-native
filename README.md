# pi-ollama-native

> [!IMPORTANT]
> **This repository is superseded by [`CaptCanadaMan/pi-ollama`](https://github.com/CaptCanadaMan/pi-ollama).**
>
> This repo documents the original *fork-based* approach — patching pi-mono's core to add a native `ollama-native` API type. It was the prototype that catalysed the work, built while figuring out how to make pi's agent loop survive Ollama's streaming tool-call bug ([ollama#12557](https://github.com/ollama/ollama/issues/12557)).
>
> The same goal is now achieved as a pi extension — no core patches required, just `pi install npm:pi-ollama`. That's the approach pi's design philosophy actually recommends (lean core, provider-specific logic in extensions). The architectural pivot was prompted by [a design document from v2nic](https://github.com/badlogic/pi-mono/issues/3357) on the pi-mono side, recommending extension-first over core PR.
>
> **For new installs, use [`pi-ollama`](https://github.com/CaptCanadaMan/pi-ollama). This repo stays online for historical reference.**

A native Ollama provider for [pi-mono](https://github.com/mariozechner/pi-mono) that fixes tool calling under streaming.

---

## The Problem

Pi ships with an `openai-completions` provider that works against Ollama's OpenAI-compat shim at `/v1/chat/completions`. Text generation works fine. Tool calling does not.

Ollama's compat shim silently drops `tool_calls` from streamed response chunks ([ollama#12557](https://github.com/ollama/ollama/issues/12557)). The model calls a tool, the streamed delta arrives, Pi never sees it. The agent loop stalls on the first tool use.

Ollama's *native* API at `/api/chat` doesn't have this problem. Tool calls arrive intact. This provider talks to the native endpoint directly.

---

## What You Get

A single provider file that translates Pi's streaming protocol to Ollama's native `/api/chat` API. Once integrated, any tool-capable model in your Ollama library works with Pi — change the model ID in settings and you're done.

---

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- At least one tool-capable model pulled (see [Verified Models](#verified-models) below)
- Node.js 20+
- A fork of [pi-mono](https://github.com/mariozechner/pi-mono)

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_NATIVE_DEBUG` | unset | Set to `1` to log every NDJSON chunk to stderr |
| `OLLAMA_NATIVE_DUMP_DIR` | unset | Path to write paired `req-*.json` / `res-*.ndjson` files per request for replay diagnostics |
| `OLLAMA_NATIVE_GHOST_RETRIES` | `2` | Max retries on a ghost-token response before surfacing an error |

---

## Verified Models

These models are confirmed tool-capable via Ollama's native API:

**Gemma 4** (Google — has native thinking mode)
```typescript
{ id: "gemma4:26b",  name: "Gemma 4 26B (Ollama)",  contextWindow: 256000, maxTokens: 8192 },
{ id: "gemma4:12b",  name: "Gemma 4 12B (Ollama)",  contextWindow: 128000, maxTokens: 8192 },
{ id: "gemma4:4b",   name: "Gemma 4 4B (Ollama)",   contextWindow: 128000, maxTokens: 8192 },
{ id: "gemma4:e4b",  name: "Gemma 4 E4B (Ollama)",  contextWindow: 128000, maxTokens: 8192 },
```

**Qwen 2.5** (Alibaba)
```typescript
{ id: "qwen2.5:72b", name: "Qwen 2.5 72B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "qwen2.5:32b", name: "Qwen 2.5 32B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "qwen2.5:14b", name: "Qwen 2.5 14B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "qwen2.5:7b",  name: "Qwen 2.5 7B (Ollama)",  contextWindow: 128000, maxTokens: 8192 },
```

**Qwen 2.5 Coder** (Alibaba — optimised for code)
```typescript
{ id: "qwen2.5-coder:32b", name: "Qwen 2.5 Coder 32B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "qwen2.5-coder:14b", name: "Qwen 2.5 Coder 14B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "qwen2.5-coder:7b",  name: "Qwen 2.5 Coder 7B (Ollama)",  contextWindow: 128000, maxTokens: 8192 },
```

**Llama 3.1 / 3.3** (Meta)
```typescript
{ id: "llama3.1:70b", name: "Llama 3.1 70B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
{ id: "llama3.1:8b",  name: "Llama 3.1 8B (Ollama)",  contextWindow: 128000, maxTokens: 8192 },
{ id: "llama3.3:70b", name: "Llama 3.3 70B (Ollama)", contextWindow: 128000, maxTokens: 8192 },
```

### Checking any model yourself

Before adding a model, verify Ollama exposes tool calling for it:

```bash
ollama show <model> | grep -i tools
```

If `tools` appears in the capabilities output, the model is tool-capable and should work with this provider.

---

## Integration

Seven edits across six files. All changes are additive — nothing existing is modified beyond adding entries to existing maps and union types.

### 1 — Register the API type and provider

**`packages/ai/src/types.ts`**

```typescript
export type KnownApi =
  | "openai-completions"
  // ... existing entries ...
  | "ollama-native";   // ← add

export type KnownProvider =
  | "anthropic"
  // ... existing entries ...
  | "ollama";          // ← add
```

### 2 — Add the provider file

Copy `ollama-native.ts` from this repo to `packages/ai/src/providers/ollama-native.ts`.

Key design decisions documented in the file header, but briefly:
- Raw `fetch` against `/api/chat` — Ollama's NDJSON is incompatible with OpenAI's SSE
- `num_ctx` defaults to 32768 — Ollama's default of 4096 silently truncates large prompts
- Tool calls emitted as a complete burst — Ollama delivers them as parsed objects in one chunk, not as streaming JSON fragments
- `sawToolCalls` flag — Ollama always returns `done_reason: "stop"` even on tool-call turns; the flag disambiguates
- Thinking blocks handled separately — Gemma 4 emits reasoning in `message.thinking`, not `<think>` tags; dropped on replay
- Ghost-token retry — Ollama occasionally generates output tokens but streams nothing visible, typically after many tool-call rounds with growing context. The provider detects the signature (`done:true`, `eval_count > 0`, empty message) on the first NDJSON line and retries up to `OLLAMA_NATIVE_GHOST_RETRIES` times (default: 2)
- Truncation detection — if the connection closes before a `done:true` chunk arrives, the provider surfaces a clear error rather than silently accepting partial output. Cannot auto-retry since partial events have already been emitted; retry the turn manually

### 3 — Register as a builtin

**`packages/ai/src/providers/register-builtins.ts`**

Four additions following the existing lazy-loader pattern:

```typescript
// 1. With the other provider type imports at the top:
import type { OllamaNativeOptions } from "./ollama-native.js";

// 2. With the other module interfaces:
interface OllamaNativeProviderModule {
  streamOllamaNative: StreamFunction<"ollama-native", OllamaNativeOptions>;
  streamSimpleOllamaNative: StreamFunction<"ollama-native", SimpleStreamOptions>;
}

// 3. With the other lazy-loader variables and functions:
let ollamaNativeProviderModulePromise:
  | Promise<LazyProviderModule<"ollama-native", OllamaNativeOptions, SimpleStreamOptions>>
  | undefined;

function loadOllamaNativeProviderModule(): Promise<
  LazyProviderModule<"ollama-native", OllamaNativeOptions, SimpleStreamOptions>
> {
  ollamaNativeProviderModulePromise ||= import("./ollama-native.js").then((module) => {
    const provider = module as OllamaNativeProviderModule;
    return {
      stream: provider.streamOllamaNative,
      streamSimple: provider.streamSimpleOllamaNative,
    };
  });
  return ollamaNativeProviderModulePromise;
}

// 4. Inside registerBuiltinApiProviders(), with the other registerApiProvider calls:
registerApiProvider({
  api: "ollama-native",
  stream: createLazyStreamFunction(loadOllamaNativeProviderModule),
  streamSimple: createLazySimpleStreamFunction(loadOllamaNativeProviderModule),
});
```

### 4 — Add models to the generator

**`packages/ai/scripts/generate-models.ts`**

`models.generated.ts` is regenerated on every build — hand-patching it doesn't survive. Add entries to the generator instead, just before the grouping step (search for `// Group by provider`):

```typescript
// Ollama models — local inference, no API key required.
// Add any tool-capable model you've pulled. Check with: ollama show <model> | grep -i tools
const ollamaModels: Model<any>[] = [
  // Gemma 4 (Google) — has native thinking mode
  { id: "gemma4:26b",  name: "Gemma 4 26B (Ollama)",       api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: true,  input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 256000, maxTokens: 8192 },
  { id: "gemma4:12b",  name: "Gemma 4 12B (Ollama)",       api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: true,  input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "gemma4:4b",   name: "Gemma 4 4B (Ollama)",        api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: true,  input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "gemma4:e4b",  name: "Gemma 4 E4B (Ollama)",       api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: true,  input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },

  // Qwen 2.5 (Alibaba)
  { id: "qwen2.5:72b", name: "Qwen 2.5 72B (Ollama)",      api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "qwen2.5:32b", name: "Qwen 2.5 32B (Ollama)",      api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "qwen2.5:14b", name: "Qwen 2.5 14B (Ollama)",      api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "qwen2.5:7b",  name: "Qwen 2.5 7B (Ollama)",       api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },

  // Qwen 2.5 Coder (Alibaba)
  { id: "qwen2.5-coder:32b", name: "Qwen 2.5 Coder 32B (Ollama)", api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "qwen2.5-coder:14b", name: "Qwen 2.5 Coder 14B (Ollama)", api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "qwen2.5-coder:7b",  name: "Qwen 2.5 Coder 7B (Ollama)",  api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },

  // Llama 3.1 / 3.3 (Meta)
  { id: "llama3.1:70b", name: "Llama 3.1 70B (Ollama)", api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "llama3.1:8b",  name: "Llama 3.1 8B (Ollama)",  api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
  { id: "llama3.3:70b", name: "Llama 3.3 70B (Ollama)", api: "ollama-native", provider: "ollama", baseUrl: "http://localhost:11434", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 128000, maxTokens: 8192 },
];
allModels.push(...ollamaModels);
```

Only include models you've pulled locally. The model ID must exactly match what `ollama list` shows.

### 5 — Skip API key checking

**`packages/ai/src/env-api-keys.ts`**

Add before the `amazon-bedrock` block:

```typescript
if (provider === "ollama") {
  // Local inference — no API key required.
  return "<authenticated>";
}
```

### 6 — Export the options type

**`packages/ai/src/index.ts`**

```typescript
export type { OllamaNativeOptions } from "./providers/ollama-native.js";
```

### 7 — Set the default model

**`packages/coding-agent/src/core/model-resolver.ts`**

In `defaultModelPerProvider` (a `Record<KnownProvider, string>`):

```typescript
ollama: "gemma4:26b",
```

---

## Build

```bash
cd packages/ai && npm run build    # runs generate-models.ts, then compiles
cd ../agent && npm run build
cd ../coding-agent && npm run build
```

Point your `pi` command at the fork:

```bash
alias pi='node /path/to/pi-mono/packages/coding-agent/dist/cli.js'
```

---

## Configure Pi for Ollama

Create `.pi/settings.json` in your project directory:

```json
{
  "defaultProvider": "ollama",
  "defaultModel": "gemma4:26b"
}
```

Pi picks this up automatically on startup. Change `defaultModel` to any model you've added to the generator and pulled in Ollama.

---

## Run

```bash
cd ~/your-project
pi
```

Pi loads `AGENTS.md` from the directory tree and connects to Ollama via the native API. Thinking blocks, tool calls, and multi-round agent loops all work.

---

## Verifying Tool Calling Works

The event sequence for a tool-call turn should look like this in Pi's output:

```
[thinking — model reasoning through the problem]
[tool call fires and returns a result]
[model responds with the answer]
```

If the model clearly intends to call a tool but the conversation stalls, confirm you're hitting `/api/chat` and not `/v1/chat/completions`. The two endpoints behave differently under streaming.

---

## Switching Models

Change `defaultModel` in `.pi/settings.json` to any model in your generator list. No other changes needed — the provider is model-agnostic.

For models with native thinking mode (Gemma 4), thinking blocks appear automatically. For models without it, the thinking event sequence is simply absent; everything else works identically.
