# Describe-a-Part Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A "Describe" chat next to Capture. The user describes a part, and a Groq 70B model (OpenAI-compatible) asks for anything missing and offers clickable options. Once every size is known, our own code builds the part and opens it in Model & Export. Demo today (27 Sep 2026, submission 17:30).

**Architecture:** `POST /api/chat` sends the conversation to the chat model with a strict system prompt and JSON output. The server validates the model's structured `part` request with Pydantic. It keeps only sizes that literally appear in the user's own messages (rule 2), and lists everything else as missing. When the request is complete, `describe.spec_from_request()` deterministically builds a `MultiViewSpec` (envelope, three view outlines, holes). The frontend `Describe` screen shows the chat, the option chips, a live part summary with badges, and "Build this part →", which opens Model.

**Tech Stack:** FastAPI, openai client (OpenAI-compatible), Pydantic; React + TS in `web/`.

## Global Constraints

- The LLM never writes code or geometry. It returns JSON only: `{"reply": str, "options": [str], "part": PartRequest | null}`.
- Rule 2: a millimetre value is accepted only if the same number appears in a **user** message. Assistant messages and the system prompt don't count. Anything else is dropped and listed in `missing`. Provenance for accepted values is `user_written`; positions derived by arithmetic (flange bolt holes, a spacer's centre) are `scaled`.
- The bot answers only about Sketch-to-CAD and designing a part in its grammar. For anything else it replies in one polite sentence that it only helps design parts here.
- Grammar for chat: `plate`, `l_bracket`, `spacer`, `flange`. `profile_extrusion` and anything else: the bot says to sketch it on the Capture screen.
- Provider only from env: `CHAT_BASE_URL`, `CHAT_MODEL`, `CHAT_API_KEY`, each falling back to `VLM_BASE_URL`, `VLM_MODEL`, `VLM_API_KEY`. `.env.example` gets the Groq preset: `https://api.groq.com/openai/v1`, `openai/gpt-oss-120b` (Groq retired the 70B Llama; `qwen/qwen3.8-27b` also works). Never hard-code the model in source.
- No key or provider configured: 503 `{"error": "The chat model is not configured. Add CHAT_API_KEY to .env."}`. A provider failure: 502 `{"error": "The chat model did not answer. Try again."}`. Never leak exception text.
- At most 20 messages and 2000 characters per message (400 otherwise). Log provider, model, latency and tokens, never message text.
- Commits: plain messages, no AI attribution. Push after each commit.

## Contract

```ts
// add to web/src/api/types.ts
export type ChatRole = 'user' | 'assistant';
export interface ChatMessage { role: ChatRole; content: string }
export type PartType = 'plate' | 'l_bracket' | 'spacer' | 'flange';
export interface PartRequest { type: PartType; values: Record<string, number>; holes: { a_mm: number; b_mm: number; diameter_mm: number }[] }
export interface ChatResponse { reply: string; options: string[]; part: PartRequest | null; missing: string[]; spec: Spec | null; model: string }
```

`POST /api/chat` takes `{messages: ChatMessage[]}` and returns `ChatResponse`.

**Required values per type** (keys in `values`, all mm unless noted):

| Type | Keys |
| --- | --- |
| plate | `width_mm`, `height_mm`, `thickness_mm` |
| l_bracket | `leg_a_mm` (horizontal), `leg_b_mm` (vertical), `width_mm`, `thickness_mm` |
| spacer | `outer_diameter_mm`, `inner_diameter_mm`, `length_mm` |
| flange | `outer_diameter_mm`, `inner_diameter_mm`, `thickness_mm`, `bolt_circle_diameter_mm`, `bolt_hole_diameter_mm`, `bolt_count` (count, not mm; also must appear in user text) |

Holes (plate only) are on the front face: `a_mm` from the left edge, `b_mm` from the bottom edge.

**Geometry** (`describe.spec_from_request`; x = width, y = height, z = depth):
- **plate:** envelope (W, H, T). front rect(W, H), top rect(W, T), right rect(T, H). Holes are `FaceHole(face="front", a, b, d)`.
- **l_bracket:** envelope (A, B, W). front L `[(0,0),(A,0),(A,t),(t,t),(t,B),(0,B)]`, top rect(A, W), right rect(W, B).
- **spacer:** envelope (D, D, L). front = circle(D/2) as a polygon with 96 points, inner loop circle(d/2), both centred at (D/2, D/2). top rect(D, L), right rect(L, D).
- **flange:** as spacer with thickness T for L, plus `bolt_count` front holes of `bolt_hole_diameter_mm` on the bolt circle, starting at angle 90°. Hole positions are provenance `scaled`.

Validation, returning `missing` rather than raising:
- inner < outer
- bolt circle strictly between the inner and outer diameters
- thickness less than both legs
- holes inside the plate
- every value > 0

The spec must pass `MultiViewSpec.model_validate`.

### Task A: Backend chat and part builder

**Files:** create `s2c/web/chat.py` and `s2c/web/describe.py`, add the route in `s2c/web/api.py`, create `tests/test_web_chat.py`, modify `.env.example`.

- **System prompt:** covers the scope, the grammar table, the JSON schema, "never invent a size, ask for it", "offer 2–4 short options when a choice helps (part type, hole count, …)", and "answer in the user's language".
- **Call:** `openai.OpenAI(base_url, api_key).chat.completions.create(model, messages=[system, ...], response_format={"type": "json_object"}, temperature=0.2)`. Parse it. On invalid JSON, retry once, then reply "Sorry, could you rephrase that?" with `part` null.
- **Tests** (fake transport, no network):
  1. A plate with all three sizes written by the user returns a spec with envelope 60×40×5, all `user_written`.
  2. The model returns `thickness_mm` 5 that the user never wrote: it is dropped, and `missing` contains `thickness_mm`.
  3. A flange builds 4 holes on the bolt circle with `scaled` positions.
  4. An off-topic model reply passes through, but `part` stays null.
  5. With no key configured, the response is 503 with `error`.
  6. Too many messages returns 400.

### Task B: Describe screen

**Files:**
- create `web/src/screens/Describe.tsx`
- add `Screen 'describe'` in `store.tsx` (a `GOTO` target) and a `chat` slice: `messages: ChatMessage[]`, reset on `RESET`
- add `chat()` to `client.ts`
- add the types to `types.ts`
- route it in `App.tsx`
- add a "Describe it instead →" link in the Capture header

**UI** (same design language as Capture/Review):
- **Left:** the chat. Bubbles: user on the right with the accent tint, assistant on the left with a `var(--ai)` dashed border and a tag showing the `model` name. Option chips under the last assistant reply; clicking one sends it. A textarea, with Enter to send and Shift+Enter for a new line. Show "thinking…" while waiting.
- **Right:** a "Your part" card with the type, each value with a "✓ Written by you" badge, and missing values as "Required" badges. A mini SVG preview of the front outline once a spec exists. Then "Build this part →", enabled when `spec` exists.
- **Build this part →:** dispatch `ANALYSIS {request_id: '', spec, abstain: null, filled_by: {}}`, then `MODEL null`, then `GOTO model`. Model must pass `request_id: null` when it is `''`.
- **Errors:** 503/502/400 become an inline StopCard with the message.
- **Opening message** (local, not from the model): "Describe the part you need — for example: a 60 × 40 mm plate, 5 mm thick, with two 6 mm holes." Offer the chips "A flat plate", "An L-bracket", "A spacer", "A flange".
