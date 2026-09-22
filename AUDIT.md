# Streaming and request-lifecycle audit

This revision fixes the request lifecycle for the whole proxy, not only the Radeon provider.

## Root cause of the Render traceback

Render/Gunicorn reported `BrokenPipeError` while writing a streaming response, followed by:

`RuntimeError: generator ignored GeneratorExit`

The critical application bug was a streaming generator that yielded a `[DONE]` frame from a `finally` block. When the downstream client disconnected, Flask/Gunicorn closed the generator with `GeneratorExit`. A generator must not yield after `GeneratorExit`; doing so produces the exact `generator ignored GeneratorExit` failure.

`BrokenPipeError` itself means the downstream socket was already closed while Gunicorn was writing. The proxy now treats that lifecycle as a normal disconnect path: it closes the upstream HTTP stream and never yields from the `GeneratorExit` path.

## Global fixes

- OpenAI-compatible streaming now opens and validates the upstream HTTP response before the Flask response is returned.
- Gemini SSE uses the same managed streaming lifecycle.
- Upstream streaming contexts are explicitly closed even when the downstream iterator is closed before its first `next()` call.
- No streaming `finally` block yields data.
- Provider errors that happen after streaming starts are converted to a safe SSE error message instead of an unhandled generator exception.
- A real OpenAI-compatible empty assistant delta is emitted immediately so downstream clients see the stream start without waiting for the first model token.
- Empty upstream events become SSE heartbeats.
- Streaming responses disable proxy buffering with `Cache-Control: no-cache, no-transform` and `X-Accel-Buffering: no`.
- User locks are held until the streaming response actually closes instead of being released when the route merely constructs the response.
- Redis locks now outlive the configured internal process timeout.
- Per-request full `gc.collect()` was removed from Flask teardown to avoid unnecessary CPU pauses.
- Malformed JSON/request payloads and empty message lists are rejected cleanly.
- The proxy test provider now validates its `<api_key>@<url>` format.
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

The `proxy` provider is non-streaming and was hardened separately.

## Validation performed

- All Python files under `gfjproxy/` and `tests/` compile successfully with Python 3.13.
- AST audit found no `try/finally` block containing a `yield` anywhere under `gfjproxy/`.
- Dispatch audit confirms all ten provider entries resolve to their provider functions.
- Isolated runtime tests passed for OpenAI and Gemini stream opening, early HTTP status failure, pre-first-next close, SSE parsing, thought filtering, upstream cleanup, and ResponseHelper disconnect/error behavior.
- Isolated model parsing checks passed for Radeon case preservation and malformed request-type rejection.

The full development test suite and Ruff were not executable in the offline analysis environment because the required third-party packages were unavailable and the package index could not be resolved. This is an environment limitation, not a reported project-test failure.
## Streaming HTTP error-body fix

When an upstream OpenAI-compatible streaming request returns a non-2xx status,
`httpx2.Client.stream()` exposes a response whose body has not yet been read.
Provider error handlers may call `response.json()` or `response.text()` after
`raise_for_status()`, which raises `ResponseNotRead` unless the body is consumed
while the stream context is still open. `_open_stream()` now reads the error body
before re-raising `HTTPStatusError`, preserving the original upstream status and
allowing all OpenAI-compatible providers to report their real 4xx/5xx errors.
A regression test covers the read-before-reraise contract, and Radeon has a
specific streaming 429 regression test.

