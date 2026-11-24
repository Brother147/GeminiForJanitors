from flask import Flask

import gfjproxy.routes.keyring as keyring_module


class _BusySlots:
    def acquire(self, blocking=False):
        assert blocking is False
        return False

    def release(self):
        raise AssertionError("a slot that was not acquired must not be released")


class _TrackingSlots:
    def __init__(self):
        self.acquired = 0
        self.released = 0

    def acquire(self, blocking=False):
        assert blocking is False
        self.acquired += 1
        return True

    def release(self):
        self.released += 1


def test_keyring_busy_rejects_before_reading_request_body(monkeypatch):
    app = Flask(__name__)
    monkeypatch.setattr(keyring_module, "_KEYRING_SLOTS", _BusySlots())

    with app.test_request_context(
        "/keyring/api", method="POST", data=b"{}", content_type="application/json"
    ):
        current_request = keyring_module.request._get_current_object()

        def fail_if_read():
            raise AssertionError("body should not be read when no slot is available")

        monkeypatch.setattr(current_request, "get_data", fail_if_read)
        result, status = keyring_module.api()

    assert status == 429
    assert result["error"] == "Keyring is busy. Please retry shortly."


def test_keyring_releases_slot_after_oversized_request(monkeypatch):
    app = Flask(__name__)
    slots = _TrackingSlots()
    monkeypatch.setattr(keyring_module, "_KEYRING_SLOTS", slots)
    body = b"x" * (keyring_module.KEYRING_MAX_REQUEST_BYTES + 1)

    with app.test_request_context(
        "/keyring/api", method="POST", data=body, content_type="application/json"
    ):
        result, status = keyring_module.api()

    assert status == 413
    assert result["error"] == "Keyring request body is too large."
    assert slots.acquired == 1
    assert slots.released == 1
