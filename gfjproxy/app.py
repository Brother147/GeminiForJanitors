################################################################################

# Early initialization

from ._globals import (
    BANDWIDTH_WARNING,
    CLOUDFLARED,
    COOLDOWN,
    MAX_REQUEST_BYTES,
    PRODUCTION,
    PROXY_BRANCH,
    PROXY_NAME,
    PROXY_VERSION,
    XUID_SECRET,
)

if PRODUCTION:
    # Production runs on gunicorn with gevent.
    # Make sure to prevent issues with ssl monkey-patching.
    from gevent import monkey

    monkey.patch_all()

################################################################################


import os

from colorama import just_fix_windows_console
from flask import Flask, request
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

from .logging import hijack_loggers, xlog
from .storage import storage
from .utils import run_cloudflared
from .xuiduser import RedisUserStorage

try:
    from psutil import Process

    _psutil_process = Process()
except ModuleNotFoundError:
    _psutil_process = None


def _teardown(exception):
    if _psutil_process is not None:
        xlog(None, f"Memory {_psutil_process.memory_info().rss / 1048576:.1f} MiB")


def create_app():
    just_fix_windows_console()
    hijack_loggers()

    if PRODUCTION:
        if not isinstance(storage, RedisUserStorage):
            raise RuntimeError("Production user storage must be Redis")
        if not XUID_SECRET or len(XUID_SECRET.encode("utf-8")) < 32:
            raise RuntimeError("Production XUID secret must be at least 32 UTF-8 bytes")

    if CLOUDFLARED is not None:
        run_cloudflared(CLOUDFLARED)

    xlog(
        None,
        " ".join(
            [
                f"{PROXY_NAME} ({PROXY_VERSION} {PROXY_BRANCH})",
                "production" if PRODUCTION else "development",
                f"bwarning={BANDWIDTH_WARNING / 1024:.1f}GiB",
                f"cpolicy={COOLDOWN!r}",
                f"ustorage={'redis' if isinstance(storage, RedisUserStorage) else 'local'}",
                f"xuid={'set' if XUID_SECRET else 'unset'}",
            ]
        ),
    )

    # Application initialization

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

    # JanitorAI's web frontend is the expected browser client. Operators can
    # extend this list for a trusted app/webview origin or use '*' deliberately.
    cors_origins = [
        origin.strip()
        for origin in os.environ.get(
            "GFJPROXY_CORS_ORIGINS",
            "https://janitorai.com,https://www.janitorai.com",
        ).split(",")
        if origin.strip()
    ]
    CORS(
        app,
        origins=cors_origins,
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        supports_credentials=False,
        max_age=600,
    )

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-Frame-Options", "DENY")
        if request.path.startswith("/keyring"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def handle_request_too_large(_error):
        return {"error": "Request body is too large."}, 413

    from .routes.keyring import keyring
    from .routes.proxy import proxy
    from .routes.system import system

    app.register_blueprint(keyring)
    app.register_blueprint(proxy)
    app.register_blueprint(system)

    app.teardown_request(_teardown)

    return app
