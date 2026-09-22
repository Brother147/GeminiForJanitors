from dataclasses import dataclass

import gfjproxy.routes.proxy as proxy_module


@dataclass
class _FakeStorage:
    announcement: str = ""
    unlock_calls: int = 0

    def lock(self, xuid):
        return True

    def unlock(self, xuid):
        self.unlock_calls += 1

    def get(self, xuid):
        return {}, False

    def put(self, xuid, data):
        return False


class _FakeUser:
    def __init__(self, storage, xuid):
        self.valid = True
        self._rcounter = 0

    def last_seen(self):
        return 0

    def get_rcounter(self):
        return self._rcounter

    def inc_rcounter(self):
        self._rcounter += 1

    def last_seen_msg(self):
        return "test user"

    def save(self):
        pass


class _TrackingStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __iter__(self):
        yield "answer"

    def close(self):
        if not self.closed:
            self.closed = True
            self.events.append("stream_closed")


def _install_stream_handler(monkeypatch, stream):
    monkeypatch.setattr(
        proxy_module,
        "handle_chat_message",
        lambda user, jai_req, response: response.add_stream(stream),
    )
    monkeypatch.setattr(proxy_module, "get_cooldown", lambda: 0)
    monkeypatch.setattr(proxy_module, "xlog", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy_module, "xlogtime", lambda *args, **kwargs: 0)
    monkeypatch.setattr(proxy_module, "print_exception", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy_module, "xuid_secret", "test-secret")


def _stream_request_context(app):
    return app.test_request_context(
        "/chat/completions",
        method="POST",
        json={
            "messages": [
                {"content": "System", "role": "system"},
                {"content": "Hello", "role": "user"},
            ],
            "model": "gemini-2.5-pro",
            "stream": True,
        },
        headers={"Authorization": "Bearer test-key"},
    )


def test_proxy_stream_unlocks_only_after_response_closes(monkeypatch):
    from flask import Flask

    app = Flask(__name__)
    storage = _FakeStorage()
    events = []
    stream = _TrackingStream(events)

    monkeypatch.setattr(proxy_module, "storage", storage)
    monkeypatch.setattr(proxy_module, "UserSettings", _FakeUser)
    _install_stream_handler(monkeypatch, stream)

    with _stream_request_context(app):
        response = proxy_module.handle()

    assert storage.unlock_calls == 0

    first = next(response.response)
    assert isinstance(first, bytes)
    assert storage.unlock_calls == 0
    assert events == []

    response.close()

    assert events == ["stream_closed"]
    assert storage.unlock_calls == 1


def test_proxy_stream_unlocks_after_normal_stream_completion(monkeypatch):
    from flask import Flask

    app = Flask(__name__)
    storage = _FakeStorage()
    events = []
    stream = _TrackingStream(events)

    monkeypatch.setattr(proxy_module, "storage", storage)
    monkeypatch.setattr(proxy_module, "UserSettings", _FakeUser)
    _install_stream_handler(monkeypatch, stream)

    with _stream_request_context(app):
        response = proxy_module.handle()

    assert storage.unlock_calls == 0

    chunks = list(response.response)

    assert all(isinstance(chunk, bytes) for chunk in chunks)
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert events == ["stream_closed"]
    assert storage.unlock_calls == 0

    response.close()

    assert storage.unlock_calls == 1
