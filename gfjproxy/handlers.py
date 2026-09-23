import re
from typing import Any, cast

from ._globals import BANNER, BANNER_VERSION
from .commands import CommandError, CommandExit
from .logging import xlog
from .models import JaiMessage, JaiRequest, JaiResult, JaiResultMetadata
from .providers.cerebras import cerebras_generate_content
from .providers.deepseek import deepseek_generate_content
from .providers.gemini import gemini_generate_content
from .providers.gemini_cli import gemini_cli_generate_content
from .providers.groq import groq_generate_content
from .providers.nvidia import nvidia_generate_content
from .providers.openrouter import openrouter_generate_content
from .providers.proxy import proxy_generate_content
from .providers.radeon import radeon_generate_content
from .providers.z_ai import z_ai_generate_content
from .statistics import track_stats
from .utils import ResponseHelper
from .xuiduser import XUID, UserSettings

################################################################################

API_KEY_PREFIXES = {
    "AIza": "google",  # Standard API keys
    "AQ.": "google",  # Authorization keys
    "csk-": "cerebras",
    "nvapi-": "nvidia",
    "sk-ant-": "anthropic",
    "sk-or-v1-": "openrouter",
    "sk-proj-": "openai",
    "gfjproxy.gemini_cli.": "gemini_cli",
    "gsk_": "groq",
    "rc-": "radeon",
}

PROVIDER_FUNCS = {
    "cerebras": cerebras_generate_content,
    "deepseek": deepseek_generate_content,
    "gemini_cli": gemini_cli_generate_content,
    "google": gemini_generate_content,
    "groq": groq_generate_content,
    "nvidia": nvidia_generate_content,
    "openrouter": openrouter_generate_content,
    "proxy": proxy_generate_content,
    "radeon": radeon_generate_content,
    "z_ai": z_ai_generate_content,
}


def _resolve_provider(api_key: str) -> tuple[str | None, str]:
    """Resolves which provider an API key belongs to.

    Returns:
        provider (str | None): The provider's name if any.
        api_key (str): The cleaned up API key.
    """
    api_key_split = api_key.split("/", maxsplit=1)
    if len(api_key_split) == 2:  # "provider/api_key" syntax
        return api_key_split[0].lower(), api_key_split[1]

    # The API key is plain and needs to be pattern matched
    for prefix, provider in API_KEY_PREFIXES.items():
        if api_key.startswith(prefix):
            return provider, api_key

    return None, api_key


def _handle_request(
    user: XUID,
    api_key: str,
    models: dict[str, str],
    messages: list[JaiMessage],
    settings: dict[str, Any] | None = None,
) -> JaiResult:
    """Dispatch a JaiRequest request to the appropriate providen given the API key."""
    provider_name, api_key = _resolve_provider(api_key)
    if not provider_name:
        return JaiResult(
            400,
            "The proxy couldn't recognize an API key.",
            extras=(
                f"Your API key `{api_key}` didn't match any of the proxy's prefixes.\n"
                "You should specify the provider at the start of your API key. For example:\n"
                "- If the key is for Cerebras, add `cerebras/` at the start of it.\n"
                "- If the key is for DeepSeek, add `deepseek/` at the start of it.\n"
                "- If the key is for Google AI or Vertex AI, add `google/` at the start of it.\n"
                "- If the key is for Groq, add `groq/` at the start of it.\n"
                "- If the key is for AMD Radeon Cloud, add `radeon/` at the start of it.\n"
                "- If the key is for Nvidia NIM, add `nvidia/` at the start of it.\n"
                "- If the key is for Z.AI, add `z_ai/` at the start of it.\n"
                "- If the key is for OpenRouter, add `openrouter/` at the start of it.\n"
                # No mention of Gemini CLI since support is WIP and its API key always resolve
            ),
            metadata=JaiResultMetadata(api_key_valid=False),
        )

    provider_func = PROVIDER_FUNCS.get(provider_name)
    if not provider_func:
        return JaiResult(
            500,
            f"You have a `{provider_name}` API key but this proxy does not support it.",
        )

    model = models.get(provider_name)
    if not model:
        extras = (
            f"You have a `{provider_name}` API key but you didn't specify a model for it.\n"
            "Make sure to use OpenRouter model syntax `provider/model`.\n"
            "Examples: `google/gemini-2.5-flash`, `cerebras/llama3.1-8b`, `deepseek/deepseek-chat`, etc."
        )

        if provider_name in ("openrouter", "nvidia"):
            extras += (
                "\n**Note For OpenRouter and Nvidia NIM API keys**:"
                "  use an extended model name:"
                " `openrouter/anthropic/claude-3.5-sonnet`,"
                " `nvidia/deepseek-ai/deepseek-v4-pro`, etc."
            )

        return JaiResult(
            400,
            f"Missing model for {provider_name}",
            extras=extras,
        )

    xlog(user, f"Using {provider_name}/{model}")

    return provider_func(user, api_key, model, messages, settings)


################################################################################


PERSONA_REGEX = re.compile(r"</([^<>]+?)'s Persona>")


def parse_user_persona_names(
    user: UserSettings, jai_req: JaiRequest
) -> tuple[str, str]:
    # JanitorAI sends at least four messages with roles: system, user, assistant, user
    if len(jai_req.messages) < 4:
        return "User", "Narrator"

    user_name: str | None = None
    first_user_message = next(
        (m for m in jai_req.messages if m.role == "user" and len(m.content) > 1),
        None,
    )
    if (
        first_user_message is not None
        and (user_name_index := first_user_message.content.find(": ")) > 0
    ):
        user_name = first_user_message.content[:user_name_index].strip()
        xlog(user, f"Parsed user name: {user_name!r}")
    if not user_name:
        xlog(user, "User name not parsed")
        user_name = "User"

    persona_name: str | None = None
    system_message = jai_req.messages[0]
    if system_message.role == "system" and (
        persona_match := PERSONA_REGEX.search(system_message.content)
    ):
        persona_name = str(persona_match.group(1)).strip()
        xlog(user, f"Parsed persona name: {persona_name!r}")
    if not persona_name:
        xlog(user, "Persona name not parsed")
        persona_name = "Narrator"

    return user_name, persona_name


################################################################################


def handle_proxy_test(
    user: UserSettings, jai_req: JaiRequest, response: ResponseHelper
) -> ResponseHelper:
    """Proxy test handler.

    The sole purpose of this is to test out the user's API key and model."""

    # Pass no settings. Defaults should allow for a successfuly proxy test.
    result = _handle_request(
        user.xuid,
        jai_req.api_key,
        jai_req.models,
        jai_req.messages,
    )

    user.valid = result.metadata.api_key_valid

    if not result:
        track_stats("r.test.failed")
        extra = ""
        if result.extras:
            extra = "\n(Send a chat message to get the full error)"
        return response.add_error(
            result.error + extra,
            result.status,
        )

    track_stats("r.test.succeeded")
    return response.add_message(
        "TEST"  # Don't send result.text in case it isn't perfect a "TEST" string
    )


def handle_chat_message(
    user: UserSettings, jai_req: JaiRequest, response: ResponseHelper
) -> ResponseHelper:
    """Chat message handler.

    This handles when the user sends a simple chat message to the bot."""

    # For in-prod print debugging and data mining lmao
    xlog(
        user,
        f"Request has {len(jai_req.messages)} message(s) with role(s): "
        + "".join(m.role[0] if m.role else "?" for m in jai_req.messages),
    )

    if not jai_req.messages:
        return response.add_error("Invalid request: messages cannot be empty.", 400)

    user_name, persona_name = parse_user_persona_names(user, jai_req)

    last_user_message = jai_req.messages[-1]
    if last_user_message.role == "assistant":
        if len(jai_req.messages) < 2:
            return response.add_error(
                "Invalid request: assistant prefill requires a previous user message.",
                400,
            )
        xlog(user, "User set prefill detected")
        last_user_message = jai_req.messages[-2]

    fwp_prefill = "SYSTEM NOTE: Do not include the following words/phrases in your output under any circumstances: "
    fwp_index = last_user_message.content.find(fwp_prefill)
    if fwp_index != -1:
        xlog(user, "User set forbidden words/phrases detected")

    if last_user_message.content.startswith("Rewrite/Enhance this message: "):
        xlog(user, "Handling enhance message ...")
        rtype = "enhance"
    elif last_user_message.content.startswith("Create a brief, focused summary"):
        xlog(user, "Handling auto summary ...")
        rtype = "summary"
    else:
        xlog(user, "Handling chat message ...")
        rtype = "message"

    command_exit = False
    command_exit_list: list[str] = []

    for command in last_user_message.commands:
        xlog(user, f"//{command.name} {command.args}")

        try:
            response = cast(ResponseHelper, command(user, jai_req, response))
        except CommandError as e:
            message = f"Error: {e} (Command has been ignored.)"
            response.add_proxy_message(message)
            xlog(user, message)
        except CommandExit:
            xlog(user, "Command exit set")
            command_exit = True
            command_exit_list.append(command.name)

    if command_exit:
        command_exit_list_str = ", ".join(f"//{c}" for c in command_exit_list)
        response.add_proxy_message(
            f"\n***\n\nRemove the command(s) {command_exit_list_str} to continue.",
        )
        return response

    # `//fixturns` is the only persistent prompt-shaping command.
    if jai_req.use_fixturns or user.use_fixturns:
        xlog(
            user,
            "Fixing request turns"
            + (" (for this message only)." if not user.use_fixturns else "."),
        )

        if jai_req.messages[-1].role != "user":
            jai_req.messages.append(JaiMessage(content=".", role="user"))

    settings = {"stream": jai_req.stream}

    # Forward generation settings supplied by JanitorAI. Zero means provider default.
    for setting in (
        "temperature",
        "frequency_penalty",
        "repetition_penalty",
        "top_k",
        "top_p",
    ):
        value = getattr(jai_req, setting)
        if value:
            xlog(user, f"Adding generation setting {setting}={value} to model.")
            settings[setting] = value

    if jai_req.max_tokens:
        xlog(user, f"Adding max_tokens={jai_req.max_tokens} to model.")
        settings["max_tokens"] = jai_req.max_tokens

    result = _handle_request(
        user.xuid,
        jai_req.api_key,
        jai_req.models,
        jai_req.messages,
        settings,
    )

    user.valid = result.metadata.api_key_valid

    if not result:
        track_stats(f"r.{rtype}.failed")

        if feedback := result.metadata.rejection_feedback:
            if feedback == "MAX_TOKENS":
                result.error += '\nTry increasing "Max tokens" in your Generation Settings or set it to zero to disable it.'
            elif result.error:
                result.error += "\nCheck the selected model/provider and your generation settings."

        response.add_error(result.error, result.status)

        if result.extras:
            response.add_proxy_message(result.extras)

        return response

    if result.stream is not None:
        stream = result.stream
        if result.extras:
            response.add_proxy_message(result.extras)

        response.add_stream(stream)
        track_stats(f"r.{rtype}.succeeded")
        return response

    result.text = result.text.strip()

    xlog(user, f"Result text is {len(result.text.split())} words")

    response.add_message(result.text)

    if result.extras:
        response.add_proxy_message(result.extras)

    if usage := result.metadata.token_usage:
        xlog(user, f" - Prompt   tokens {usage.prompt_tokens}")
        xlog(user, f" - Response tokens {usage.completion_tokens}")
        xlog(user, f" - Thinking tokens {usage.reasoning_tokens}")
        xlog(user, f" - Total    tokens {usage.total_tokens}")
    else:
        xlog(user, " - No usage metadata")

    if not jai_req.quiet and user.do_show_banner(BANNER_VERSION):
        xlog(
            user, f"Showing{' new ' if not user.exists else ' '}user the latest banner"
        )
        response.add_message(BANNER)

    track_stats(f"r.{rtype}.succeeded")
    return response
