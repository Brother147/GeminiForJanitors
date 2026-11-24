# Streaming and request-lifecycle audit

This revision fixes the request lifecycle for the whole proxy, not only the Radeon provider.

## Root cause of the Render traceback

Render/Gunicorn reported `BrokenPipeError` while writing a streaming response, followed by:

`RuntimeError: generator ignored GeneratorExit`

The critical application bug was a streaming generator that yielded a `[DONE]` frame from a `finally` block. When the downstream client disconnected, Flask/Gunicorn closed the generator with `GeneratorExit`. A generator must not yield after `GeneratorExit`; doing so produces the exact `generator ignored GeneratorExit` failure.

`BrokenPipeError` itself means the downstream socket was already closed while Gunicorn was writing. The proxy now treats that lifecycle as a normal disconnect path: it closes the upstream HTTP stream and never yields from the `GeneratorExit` path.

## Global fixes

- OpenAI-compatible streaming is now lazy: the Flask/SSE response is returned before the upstream model connection is opened. This prevents long first-token latency from blocking the downstream client.
- Gemini SSE uses the same managed streaming lifecycle.
- Upstream streaming contexts are explicitly closed even when the downstream iterator is closed before its first `next()` call.
- No streaming `finally` block yields data.
- Provider errors that happen after streaming starts are converted to a safe SSE error message instead of an unhandled generator exception.
- The stream sends an SSE heartbeat before opening the upstream model connection, so downstream clients receive bytes immediately even when first-token latency is long.
- Empty upstream events become SSE heartbeats.
- Streaming responses disable proxy buffering with `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`.
- User locks are held until the streaming response actually closes instead of being released when the route merely constructs the response.
- Redis locks now outlive the configured internal process timeout.
- Per-request full `gc.collect()` was removed from Flask teardown to avoid unnecessary CPU pauses.
- Malformed JSON/request payloads and empty message lists are rejected cleanly.
- The optional test-proxy provider validates HTTPS URLs against an explicit host allowlist, rejects private/reserved DNS results and non-standard ports, and is disabled by default.
- Provider error JSON parsing is hardened where non-JSON upstream responses were able to trigger secondary exceptions.
- `ResponseHelper` preserves the original non-200 status for a single error, while multiple queued error/proxy messages are intentionally rendered as a normal 200 chat response as required by the existing API contract.
- Gemini CLI credential parsing is hardened against malformed token data.

## Provider chain coverage

The following providers use the shared OpenAI-compatible stream path:

- Cerebras
- DeepSeek
- Groq
- Nvidia NIM
- OpenRouter
- Radeon Cloud
- Z.AI

The following providers use the shared Gemini SSE stream path:

- Google Gemini
- Gemini CLI

The optional generic `proxy` test provider uses the shared OpenAI-compatible streaming path and remains disabled unless an operator enables it and configures an exact hostname allowlist.

## Validation performed

- All Python files under `gfjproxy/` and `tests/` compile successfully with Python 3.13.
- AST audit found no `try/finally` block containing a `yield` anywhere under `gfjproxy/`.
- Dispatch audit confirms all nine standard provider entries resolve to their provider functions; the generic test proxy is separately gated.
- Isolated runtime tests passed for OpenAI and Gemini stream opening, early HTTP status failure, pre-first-next close, SSE parsing, thought filtering, upstream cleanup, and ResponseHelper disconnect/error behavior.
- Isolated model parsing checks passed for Radeon case preservation and malformed request-type rejection.

The full development test suite and Ruff were not executable in the offline analysis environment because the required third-party packages were unavailable and the package index could not be resolved. This is an environment limitation, not a reported project-test failure.
## Streaming HTTP error-body fix

Streaming HTTP status validation is performed when the lazy upstream stream is
first consumed. If the provider returns a non-2xx response, the error body is
read while the HTTP context is still open, then converted to a safe streaming
error. This avoids `ResponseNotRead` and prevents a provider 429/5xx from being
silently converted into an internal proxy failure.

## GLM-5.3 / NVIDIA NIM compatibility

NVIDIA currently documents `z-ai/glm-5.3` and `z-ai/glm-5.3-flash` as reasoning
models whose reasoning content is returned separately from the answer. The NVIDIA
provider now explicitly sends `chat_template_kwargs.clear_thinking=true` for those
two model IDs, avoiding accidental replay of hidden reasoning in chat history.
The shared OpenAI parser also accepts `delta.content`, `message.content`, and
legacy `choice.text` forms.


## JanitorAI endpoint and deployment hardening (November 24, 2026)

- Added aliases for `/v1/chat/completions` and quiet-mode equivalents while preserving the existing root and `/chat/completions` paths.
- Added bounded request sizes, message/model/token limits, finite-number validation, and a configurable cap on multi-key Authorization headers.
- Restricted browser CORS to JanitorAI origins by default; trusted deployments can extend the allowlist with `GFJPROXY_CORS_ORIGINS`.
- Required a strong persistent XUID secret in production and generated a fresh random secret in the Colab launcher instead of using a public constant.
- Added no-store cache headers for keyring responses, body-size and concurrency limits for credential validation, and more defensive OAuth credential parsing.
- Moved user-storage initialization inside the lock cleanup scope so a storage exception cannot leave a request lock stranded.
- Added Docker build-context exclusions and a bounded local health check; Docker installs locked dependencies before copying application source (`--no-install-project`) to keep the dependency layer cacheable, and Docker, CI, and Render use uv 0.13.9.
- Added targeted tests for malformed numeric settings, SSRF/allowlist checks, and lock cleanup after storage initialization failure.

Validation in this environment: Python byte-compilation, notebook JSON parsing, GitHub Actions YAML parsing, and `uv lock --check` pass. The full pytest and Ruff runs remain unverified because this sandbox has no installed project dependencies and cannot resolve the package index; install dependencies and run `uv sync --locked --all-extras --dev`, `uv run ruff check`, and `uv run pytest` in CI or locally before merging.


## Dependency security refresh (November 24, 2026)

- Updated `httpx2` and its paired `httpcore2` to `2.16.2`. The current PyPI metadata lists no outstanding advisories for `httpx2 2.16.2`; this is newer than the affected `2.9.1` versions in the original lockfile and includes fixes for the 2026 multipart-header, request-smuggling, and decompression-amplification advisories.
- Updated Gunicorn from `26.0.0` to `27.0.1`. The service uses Flask's WSGI interface with the gevent worker, not Gunicorn's ASGI worker path.
- Retained Brotli and HTTP/2 support. The optional HTTPX2 zstd extra was intentionally not enabled because its current dependency chain adds a separate backport package; this avoids introducing an unverified dependency while preserving ordinary JSON and SSE transport.
- Updated the lockfile using the exact current PyPI artifact URLs and SHA-256 metadata, then verified dependency consistency with `uv lock --check`. Artifact downloads and a full clean install could not be tested in this network-restricted environment.


## JanitorAI Thinking menu and reasoning passthrough (November 24, 2026)

- Added request intake support for JanitorAI's seven proxy thinking levels: Off, Minimal, Low, Medium, High, Extra high, and Max. JanitorAI exposes these checkboxes in its own proxy configuration; all-off hides the chat-level control.
- Added OpenAI-compatible `reasoning_effort` and OpenRouter `reasoning: {"effort": ...}` support, including `extra_high` → `xhigh` and `off` → `none` normalization. OpenRouter structured reasoning is authoritative if both forms arrive.
- Flattened SDK-style `extra_body` into raw HTTP parameters rather than incorrectly forwarding an SDK-only wrapper.
- DeepSeek receives its documented `thinking: {"type": "enabled|disabled"}` toggle and its supported `reasoning_effort` values. Janitor levels are mapped as `minimal` → `low`, `medium`/`high`/`xhigh` → `high`, and `max` → `max`.
- Gemini 2.5 receives a numeric `thinkingBudget`; Gemini 3+ receives `thinkingLevel`. Unsupported tiers are clamped to a model-supported level; Gemini 3+ and Gemini 2.5 Pro cannot truly disable thinking, so Off selects the lowest available level instead.
- Message objects retain provider-specific fields such as `tool_calls`, `tool_call_id`, `reasoning_content`, `name`, and multimodal content arrays. OpenAI-compatible providers forward these fields unchanged. Gemini's native adapter converts inline base64 images and text parts; arbitrary remote image URLs are not fetched.
- Added unit tests for both Janitor payload shapes, alias normalization, DeepSeek mappings, Gemini model-specific restrictions, `extra_body` flattening, and preservation of DeepSeek reasoning state across assistant/tool turns.

Sources checked on November 24, 2026: JanitorAI community reports for the new Thinking selector (seven proxy levels and OpenAI-compatible/OpenRouter modes); OpenRouter Chat Completions reference for supported reasoning values; DeepSeek Thinking Mode and Chat Completions references for `thinking`/`reasoning_effort`; Google Cloud Gemini thinking documentation for model-specific `thinkingLevel` and `thinkingBudget` constraints. The full pytest suite is still blocked in this sandbox because Flask and other project dependencies are not installed and the package index is unreachable; compilation and static checks are performed on this revised archive.

Final validation for the thinking-menu revision (November 24, 2026): Python `compileall`, AST parsing of all 52 Python files, notebook JSON parsing, TOML/YAML parsing, and `uv lock --check` pass. Twenty-two isolated assertions passed, covering request parsing, unknown-field retention, tool/multimodal message round-trips, OpenRouter and DeepSeek reasoning conversion, Gemini model-specific levels, `extra_body`, and image-part conversion. Full pytest collection still stops before tests run because Flask is unavailable in this environment; Ruff is also not installed.


## GitHub Actions correction (2026-11-24)

- Fixed `astral-sh/setup-uv@v10`: this action intentionally publishes immutable full-version tags from v8 onward, not a floating major tag.
- Pinned the verified v10.3.3 commit `68559e99cc807c1eb7e9238fd43a15bba460eec2`.
- The workflow still installs uv `0.13.9` explicitly; only the action reference changed.
