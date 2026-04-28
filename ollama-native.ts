// packages/ai/src/providers/ollama-native.ts
//
// Native Ollama /api/chat provider for pi-mono.
//
// Adds an `ollama-native` API type that talks directly to Ollama's native
// /api/chat endpoint, bypassing the OpenAI-compatibility shim at
// /v1/chat/completions which silently drops `tool_calls` from streamed deltas.
// See ollama#12557 for the upstream root cause; the same drop is observable
// downstream in pi-mono via openai-completions when baseUrl points at :11434/v1.
//
// Structurally this provider mirrors `google.ts`: each tool_calls entry is
// emitted as a complete-tool-call burst (toolcall_start → toolcall_delta →
// toolcall_end) because Ollama's native endpoint delivers tool calls as
// already-parsed JSON objects in a single chunk per call. This sidesteps the
// partial-JSON streaming assumption in openai-completions that breaks on
// Ollama's whole-JSON-in-one-chunk pattern (pi-mono#4892).
//
// What's intentionally absent versus openai-completions.ts:
//   - prompt cache / cache_control machinery (Ollama has none)
//   - reasoning_effort plumbing (reasoning is automatic and surfaces via a
//     dedicated `thinking` field in the stream, not via inline <think> tags)
//   - OpenRouter / Vercel routing (not applicable)
//   - max_tokens vs. max_completion_tokens compat (Ollama uses options.num_predict)
//   - hasToolHistory + Anthropic-via-LiteLLM quirks (not applicable)
//   - OpenAI-style strict-mode tool definitions (Ollama is lenient)
//
// num_ctx default is 32768. Ollama's per-request default is 4096 regardless
// of what the model actually supports, which silently truncates large prompts
// (a system prompt plus 10+ tool definitions exceeds 4096 quickly). Setting
// num_ctx explicitly on every request avoids that failure mode. The first
// request at a given num_ctx triggers a one-time model reload (~5–10s);
// subsequent requests at the same value are cached.

import { calculateCost } from "../models.js";
import type {
	AssistantMessage,
	Context,
	ImageContent,
	Model,
	SimpleStreamOptions,
	StopReason,
	StreamFunction,
	StreamOptions,
	TextContent,
	ThinkingContent,
	Tool,
	ToolCall,
	ToolResultMessage,
} from "../types.js";
import { AssistantMessageEventStream } from "../utils/event-stream.js";
import { headersToRecord } from "../utils/headers.js";
import { sanitizeSurrogates } from "../utils/sanitize-unicode.js";
import { buildBaseOptions } from "./simple-options.js";
import { transformMessages } from "./transform-messages.js";

// ============================================================================
// Public options
// ============================================================================

export interface OllamaNativeOptions extends StreamOptions {
	/**
	 * Ollama context window in tokens. Default: 32768.
	 *
	 * Ollama's per-request default is 4096 tokens regardless of the model's
	 * max context — large prompts get silently truncated unless this is set
	 * explicitly. Setting num_ctx on the first request triggers a one-time
	 * model reload (~5–10 seconds); subsequent requests at the same value are
	 * cached.
	 */
	numCtx?: number;

	/**
	 * Ollama keep_alive — how long to keep the model loaded after this request.
	 * Accepts a duration string ("5m", "1h") or a number of seconds.
	 * Default: "5m".
	 */
	keepAlive?: string | number;

	/**
	 * Ollama format option. Either "json" for raw JSON mode, or a JSON Schema
	 * object for structured output. Most callers should pass tools via context
	 * instead; this is for the raw structured-output use case.
	 */
	format?: string | object;
}

// ============================================================================
// Wire types for /api/chat
// ============================================================================

interface OllamaTool {
	type: "function";
	function: {
		name: string;
		description: string;
		parameters: object;
	};
}

interface OllamaWireToolCall {
	id?: string;
	function: {
		index?: number;
		name: string;
		arguments: Record<string, unknown>;
	};
}

interface OllamaWireMessage {
	role: "user" | "assistant" | "tool" | "system";
	content?: string;
	thinking?: string;
	tool_calls?: OllamaWireToolCall[];
	tool_name?: string;
	images?: string[];
}

interface OllamaChunk {
	model: string;
	created_at: string;
	message?: OllamaWireMessage;
	done: boolean;
	done_reason?: string;
	total_duration?: number;
	load_duration?: number;
	prompt_eval_count?: number;
	prompt_eval_duration?: number;
	eval_count?: number;
	eval_duration?: number;
	error?: string;
}

interface OllamaRequest {
	model: string;
	messages: OllamaWireMessage[];
	tools?: OllamaTool[];
	stream: true;
	options?: {
		num_ctx?: number;
		temperature?: number;
		num_predict?: number;
	};
	keep_alive?: string | number;
	format?: string | object;
}

const DEFAULT_NUM_CTX = 32768;
const DEFAULT_KEEP_ALIVE = "5m";
const DEFAULT_BASE_URL = "http://localhost:11434";

// ============================================================================
// Type guards
// ============================================================================

function isTextContentBlock(block: { type: string }): block is TextContent {
	return block.type === "text";
}

function isToolCallBlock(block: { type: string }): block is ToolCall {
	return block.type === "toolCall";
}

function isImageContentBlock(block: { type: string }): block is ImageContent {
	return block.type === "image";
}

// ============================================================================
// Tool call ID generation (only used if Ollama omits one — rare in practice)
// ============================================================================

let toolCallCounter = 0;

function generateToolCallId(): string {
	return `ollama_${Date.now()}_${++toolCallCounter}`;
}

// ============================================================================
// Main stream function
// ============================================================================

export const streamOllamaNative: StreamFunction<"ollama-native", OllamaNativeOptions> = (
	model: Model<"ollama-native">,
	context: Context,
	options?: OllamaNativeOptions,
): AssistantMessageEventStream => {
	const stream = new AssistantMessageEventStream();

	(async () => {
		const output: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};

		try {
			const baseUrl = (model.baseUrl || DEFAULT_BASE_URL).replace(/\/+$/, "");
			const url = `${baseUrl}/api/chat`;

			let body = buildParams(model, context, options);
			const nextBody = await options?.onPayload?.(body, model);
			if (nextBody !== undefined) {
				body = nextBody as OllamaRequest;
			}

			const headers: Record<string, string> = {
				"Content-Type": "application/json",
				...(model.headers ?? {}),
				...(options?.headers ?? {}),
			};

			const response = await fetch(url, {
				method: "POST",
				headers,
				body: JSON.stringify(body),
				signal: options?.signal,
			});

			await options?.onResponse?.({ status: response.status, headers: headersToRecord(response.headers) }, model);

			if (!response.ok) {
				const text = await response.text().catch(() => "");
				throw new Error(`Ollama /api/chat returned HTTP ${response.status}: ${text.slice(0, 500)}`);
			}
			if (!response.body) {
				throw new Error("Ollama /api/chat returned no response body");
			}

			stream.push({ type: "start", partial: output });

			let currentBlock: TextContent | ThinkingContent | ToolCall | null = null;
			const blocks = output.content;
			const blockIndex = () => blocks.length - 1;

			const finishCurrentBlock = (block: typeof currentBlock) => {
				if (!block) return;
				if (block.type === "text") {
					stream.push({
						type: "text_end",
						contentIndex: blockIndex(),
						content: block.text,
						partial: output,
					});
				} else if (block.type === "thinking") {
					stream.push({
						type: "thinking_end",
						contentIndex: blockIndex(),
						content: block.thinking,
						partial: output,
					});
				} else if (block.type === "toolCall") {
					stream.push({
						type: "toolcall_end",
						contentIndex: blockIndex(),
						toolCall: block,
						partial: output,
					});
				}
			};

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = "";
			let sawToolCalls = false;

			outer: while (true) {
				const { value, done: streamDone } = await reader.read();
				if (streamDone) break;
				buffer += decoder.decode(value, { stream: true });

				while (true) {
					const nl = buffer.indexOf("\n");
					if (nl === -1) break;
					const line = buffer.slice(0, nl).trim();
					buffer = buffer.slice(nl + 1);
					if (!line) continue;

					let chunk: OllamaChunk;
					try {
						chunk = JSON.parse(line) as OllamaChunk;
					} catch {
						// Malformed NDJSON line — should never happen with /api/chat in
						// practice, but don't kill the stream over a single bad line.
						continue;
					}

					if (chunk.error) {
						throw new Error(`Ollama returned error: ${chunk.error}`);
					}

					// Synthesize a stable response identifier from the first chunk.
					// Ollama doesn't supply a response ID; created_at + model is unique
					// enough for log correlation.
					if (!output.responseId && chunk.created_at) {
						output.responseId = `ollama-${chunk.created_at}-${chunk.model}`;
					}

					const m = chunk.message;
					if (m) {
						// Thinking field — separate from content. Gemma 4 emits
						// reasoning here as token-by-token deltas, not as inline
						// <think> tags.
						if (m.thinking !== undefined && m.thinking.length > 0) {
							if (!currentBlock || currentBlock.type !== "thinking") {
								finishCurrentBlock(currentBlock);
								currentBlock = { type: "thinking", thinking: "" };
								output.content.push(currentBlock);
								stream.push({
									type: "thinking_start",
									contentIndex: blockIndex(),
									partial: output,
								});
							}
							currentBlock.thinking += m.thinking;
							stream.push({
								type: "thinking_delta",
								contentIndex: blockIndex(),
								delta: m.thinking,
								partial: output,
							});
						}

						// Content field — the visible text response.
						if (m.content !== undefined && m.content.length > 0) {
							if (!currentBlock || currentBlock.type !== "text") {
								finishCurrentBlock(currentBlock);
								currentBlock = { type: "text", text: "" };
								output.content.push(currentBlock);
								stream.push({
									type: "text_start",
									contentIndex: blockIndex(),
									partial: output,
								});
							}
							currentBlock.text += m.content;
							stream.push({
								type: "text_delta",
								contentIndex: blockIndex(),
								delta: m.content,
								partial: output,
							});
						}

						// Tool calls — Ollama delivers each tool call as a complete
						// chunk with parsed arguments. Emit start+delta+end as a single
						// burst (matches the google.ts pattern). Parallel tool calls
						// arrive as separate consecutive chunks, each with
						// tool_calls.length === 1 in practice — so the loop here
						// handles both the single-call and (theoretical) multi-call
						// case uniformly.
						if (m.tool_calls && m.tool_calls.length > 0) {
							if (currentBlock) {
								finishCurrentBlock(currentBlock);
								currentBlock = null;
							}
							sawToolCalls = true;

							for (const wireTc of m.tool_calls) {
								const args = wireTc.function.arguments ?? {};
								const argsString = JSON.stringify(args);

								const providedId = wireTc.id;
								const isDuplicate =
									providedId !== undefined &&
									output.content.some((b) => b.type === "toolCall" && b.id === providedId);
								const id = !providedId || isDuplicate ? generateToolCallId() : providedId;

								const toolCall: ToolCall = {
									type: "toolCall",
									id,
									name: wireTc.function.name,
									arguments: args,
								};

								output.content.push(toolCall);
								stream.push({
									type: "toolcall_start",
									contentIndex: blockIndex(),
									partial: output,
								});
								stream.push({
									type: "toolcall_delta",
									contentIndex: blockIndex(),
									delta: argsString,
									partial: output,
								});
								stream.push({
									type: "toolcall_end",
									contentIndex: blockIndex(),
									toolCall,
									partial: output,
								});
							}
						}
					}

					// Final chunk — populate usage and stop reason, then exit both loops.
					if (chunk.done) {
						finishCurrentBlock(currentBlock);
						currentBlock = null;

						output.usage = {
							input: chunk.prompt_eval_count ?? 0,
							output: chunk.eval_count ?? 0,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: (chunk.prompt_eval_count ?? 0) + (chunk.eval_count ?? 0),
							cost: {
								input: 0,
								output: 0,
								cacheRead: 0,
								cacheWrite: 0,
								total: 0,
							},
						};
						calculateCost(model, output.usage);

						// Ollama returns done_reason "stop" even on tool-call turns,
						// so disambiguate via the local sawToolCalls flag.
						output.stopReason = sawToolCalls ? "toolUse" : mapDoneReason(chunk.done_reason);

						break outer;
					}
				}
			}

			// Defensive cleanup if the stream ended without a done chunk.
			finishCurrentBlock(currentBlock);

			if (options?.signal?.aborted) {
				throw new Error("Request was aborted");
			}
			if (output.stopReason === "aborted" || output.stopReason === "error") {
				throw new Error(output.errorMessage || "An unknown error occurred");
			}

			stream.push({ type: "done", reason: output.stopReason, message: output });
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
			stream.push({ type: "error", reason: output.stopReason, error: output });
			stream.end();
		}
	})();

	return stream;
};

export const streamSimpleOllamaNative: StreamFunction<"ollama-native", SimpleStreamOptions> = (
	model: Model<"ollama-native">,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream => {
	// Ollama doesn't require an API key. Pass through whatever was provided
	// (some users put a reverse proxy in front and need auth headers); if
	// nothing was provided that's also fine.
	const base = buildBaseOptions(model, options, options?.apiKey);
	return streamOllamaNative(model, context, base satisfies OllamaNativeOptions);
};

// ============================================================================
// Internal helpers
// ============================================================================

function buildParams(model: Model<"ollama-native">, context: Context, options?: OllamaNativeOptions): OllamaRequest {
	const messages = convertMessages(model, context);

	const ollamaOptions: NonNullable<OllamaRequest["options"]> = {
		num_ctx: options?.numCtx ?? DEFAULT_NUM_CTX,
	};
	if (options?.temperature !== undefined) {
		ollamaOptions.temperature = options.temperature;
	}
	if (options?.maxTokens !== undefined) {
		ollamaOptions.num_predict = options.maxTokens;
	}

	const body: OllamaRequest = {
		model: model.id,
		messages,
		stream: true,
		options: ollamaOptions,
		keep_alive: options?.keepAlive ?? DEFAULT_KEEP_ALIVE,
	};

	if (context.tools && context.tools.length > 0) {
		body.tools = convertTools(context.tools);
	}
	if (options?.format !== undefined) {
		body.format = options.format;
	}

	return body;
}

function convertMessages(model: Model<"ollama-native">, context: Context): OllamaWireMessage[] {
	// transformMessages handles image-downgrade for non-vision models, orphan
	// tool-call result synthesis, and cross-model thinking-block normalization.
	// What lands here is already cleaned up.
	const transformed = transformMessages(context.messages, model);
	const out: OllamaWireMessage[] = [];

	if (context.systemPrompt) {
		out.push({ role: "system", content: sanitizeSurrogates(context.systemPrompt) });
	}

	for (const msg of transformed) {
		if (msg.role === "user") {
			if (typeof msg.content === "string") {
				out.push({
					role: "user",
					content: sanitizeSurrogates(msg.content),
				});
			} else {
				const text = msg.content
					.filter(isTextContentBlock)
					.map((b) => b.text)
					.join("\n");
				const images = msg.content.filter(isImageContentBlock).map((b) => b.data);

				const wire: OllamaWireMessage = {
					role: "user",
					content: sanitizeSurrogates(text),
				};
				if (images.length > 0) {
					wire.images = images;
				}
				if (!wire.content && !wire.images) continue;
				out.push(wire);
			}
		} else if (msg.role === "assistant") {
			// Drop thinking blocks on replay. Ollama re-derives reasoning each
			// turn; round-tripping past thinking adds prompt cost without behavior
			// gain. Text + tool calls preserve all the conversation signal Ollama
			// can act on.
			const text = msg.content
				.filter(isTextContentBlock)
				.map((b) => b.text)
				.join("");
			const toolCalls = msg.content.filter(isToolCallBlock);

			const wire: OllamaWireMessage = {
				role: "assistant",
				content: sanitizeSurrogates(text),
			};
			if (toolCalls.length > 0) {
				wire.tool_calls = toolCalls.map((tc) => ({
					id: tc.id,
					function: {
						name: tc.name,
						arguments: tc.arguments,
					},
				}));
			}

			// Skip empty assistant turns (no content and no tool calls).
			if (!wire.content && !wire.tool_calls) continue;
			out.push(wire);
		} else if (msg.role === "toolResult") {
			const tr = msg as ToolResultMessage;
			const text = tr.content
				.filter(isTextContentBlock)
				.map((b) => b.text)
				.join("\n");

			out.push({
				role: "tool",
				content: sanitizeSurrogates(text || "(no result)"),
				tool_name: tr.toolName,
			});
		}
	}

	return out;
}

function convertTools(tools: Tool[]): OllamaTool[] {
	return tools.map((tool) => ({
		type: "function",
		function: {
			name: tool.name,
			description: tool.description,
			// TypeBox already produces JSON Schema, which is what Ollama expects.
			parameters: tool.parameters as object,
		},
	}));
}

function mapDoneReason(reason: string | undefined): StopReason {
	switch (reason) {
		case "stop":
		case "end":
		case undefined:
			return "stop";
		case "length":
			return "length";
		default:
			// Ollama doesn't surface rich done_reason values in practice.
			return "stop";
	}
}
