# impala-scope

Provider-neutral inference profiling across agents, harnesses, and model providers.

It observes requests and responses only long enough to calculate analytics. Prompts, messages, generated text, tool arguments/results, headers, and provider-native bodies are never written to disk.

Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0).

## Use

Profile any CLI or agent process:

```bash
uvx impala-scope -- codex exec "Reply with OK"
uvx impala-scope -- claude -p "Reply with OK"
```

The wrapper starts a local HTTPS proxy, adds its CA to the child's normal trust roots, and shuts it down when the child exits. Requests and responses pass through incrementally while analytics are summarized in bounded memory and written on a fork-safe background worker. Configure it with `--db`, `--host`, `--port`, `--upstream-hosts`, `--confdir`, and `--session-id`. When set, `--upstream-hosts` also limits TLS interception, so excluded and certificate-pinned hosts pass through untouched.

The CLI prints the resolved database path at startup and completion. If the file already exists, impala-scope continues appending to it and applies compatible schema migrations in place; it never replaces the database or deletes prior analytics.

Run a standalone proxy, or print its generated CA path:

```bash
impala-scope --db trace.db --port 8080
impala-scope ca-path
```

Profile an `httpx`-based Python application in-process:

```python
import impala_scope

impala_scope.configure(database_path="trace.db", server_url="*")
```

Importing `impala_scope` installs the `httpx` hook. `configure()`, `install()`, `uninstall()`, `set_session_id()`, `current_session_id()`, and `record_session()` are available for explicit control.

Use `record_session()` when one process runs concurrent agents:

```python
with impala_scope.record_session("customer-session-123"):
    agent.run()
```

Session identifiers and fingerprints are HMACed before storage using a per-database key saved beside the database with owner-only permissions. Set `IMPALA_SCOPE_HASH_KEY` only when deliberate correlation across databases is required.

## What is stored

SQLite contains one append-only `requests` table and two derived views, `sessions` and `provider_daily`.
Records cross a strict Pydantic validation boundary before insertion.

- pseudonymous `machine_hash`, `session_hash`, `run_id`, and request fingerprint;
- provider, host, API shape, endpoint, method, and model;
- start/end timestamps and duration;
- HTTP/transport outcome, error category, and hashed error fingerprint;
- input, uncached, cache-read, cache-write, output, reasoning, and total tokens;
- a count of rejected negative, fractional, non-finite, or oversized token metrics;
- cache hit/ratio, provider-reported cost, finish reason, and tool-call count;
- request/response wire-byte counts, streaming flag, and an `analytics_truncated` visibility marker.

There are deliberately no request, response, message, prompt, content, tool-argument, tool-result, or header columns.

Set `IMPALA_SCOPE_MACHINE_ID` in containers or multi-tenant environments. Its value is SHA-256 hashed and never stored directly. Without it, the profiler uses the operating-system machine identity with a platform fingerprint fallback.

## Coverage

The proxy observes HTTP JSON, NDJSON, SSE, WebSocket, and AWS event-stream traffic from any language/runtime. Detection covers OpenAI-compatible Chat, Responses, Realtime, Completions, Embeddings, Images and Audio; Anthropic; Bedrock and SageMaker; Gemini/Vertex; Hugging Face; Copilot; Cohere; Ollama; Replicate; and inference-shaped APIs from unknown providers. Explicit `server_url` and `--upstream-hosts` scopes also capture provider-neutral endpoints even when their paths are opaque. Requests negotiate gzip or deflate so responses can be decoded safely for analytics while their original bytes are forwarded unchanged. A server-forced Brotli/zstd response, malformed compression, or exhausted decoding budget is forwarded unchanged and marked `analytics_truncated`.

Authorization and other headers are reduced to a session hash before deferral. Raw prompts, responses, headers, and tool data never enter the analytics queue. An unusable database fails startup before a wrapped child is launched. Later database failures remain fail-open, never change an inference response, and emit a visible warning on stderr.

## Query examples

```sql
SELECT machine_hash, provider, model, count(*) requests,
       sum(input_tokens) input_tokens,
       sum(cached_input_tokens) cached_tokens,
       sum(output_tokens) output_tokens,
       round(100.0 * avg(success = 0), 2) error_pct
FROM requests
GROUP BY machine_hash, provider, model;

SELECT * FROM sessions ORDER BY request_count DESC;
```
