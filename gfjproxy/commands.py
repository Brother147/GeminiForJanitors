"""Commands and processing of messages."""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from random import randint

from ._globals import BANNER, BANNER_VERSION
from .utils import ResponseHelper

################################################################################


def _stripmultispace(string, *, regex=re.compile(r" +")):
    """Coalesce multiple consecutive spaces into one."""

    return regex.sub(" ", string)


PROXY_TAG_REGEX = re.compile(
    re.escape(ResponseHelper.PROXY_TAG_OPEN)
    + r".*?"
    + re.escape(ResponseHelper.PROXY_TAG_CLOSE),
    re.DOTALL,
)


def _stripproxytext(
    string,
    *,
    regex=PROXY_TAG_REGEX,
):
    """Remove <proxy></proxy> tags and their content."""

    return regex.sub("", string)


def _tokenize(string, *, regex=re.compile(r"/+|\w+|\s+|.")):
    """Split a token into '//' or longer, words, white space and punctuation."""

    return (match.group(0) for match in regex.finditer(string))


################################################################################


@dataclass
class Command:
    """User proxy command."""

    # Command name (lowercase, without any leading //)
    name: str

    # Optional command arguments (a single string, must match command argspec)
    args: str = ""

    # Pointer to command function
    func: Callable | None = field(default=None, repr=False, compare=False, kw_only=True)

    # Prefer to this method instead of directly calling func
    def __call__(self, user, jai_req, response):
        if self.func is None:
            raise RuntimeError(f"Calling {self} without a function pointer")
        return self.func(self.args, user, jai_req, response)


class CommandError(Exception):
    """User error raised by commands."""


class CommandExit(Exception):
    """Exception to exit processing early raised by commands."""


COMMANDS = {}


def command(*, argspec: str = "", **kwargs):
    if argspec:
        regex = re.compile(argspec)

    def outer_wrapper(func):
        cmd_name = func.__name__

        @wraps(func)
        def inner_wrapper(args, user, jai_req, response):
            if argspec and not regex.fullmatch(args):  # pyright: ignore[reportPossiblyUnboundVariable]
                if not args:
                    raise CommandError(
                        f'`//{cmd_name}` requires an argument "`{argspec}`".'
                    )
                raise CommandError(
                    f'`//{cmd_name}` only accepts "`{argspec}`", not "`{args}`".'
                )

            if argspec == r"off|on|this" and (setting := kwargs.get("setting")):
                attr = f"use_{setting}"

                if args == "this":
                    setattr(jai_req, attr, True)
                elif args == "on":
                    setattr(jai_req, attr, True)
                    setattr(user, attr, True)
                else:  # "off"
                    setattr(jai_req, attr, False)
                    setattr(user, attr, False)

            return func(args, user, jai_req, response)

        COMMANDS[cmd_name] = {
            "argcount": 1 if argspec else 0,
            "func": inner_wrapper,
        }

        return inner_wrapper

    return outer_wrapper


################################################################################



@command()
def aboutme(args, user, jai_req, response):
    """Show persistent proxy settings and usage information."""
    response.add_proxy_message(
        f"Your user ID on this proxy is `{user.xuid!r}`.",
        f"You have used this proxy {user.get_rcounter()} time(s).",
        f"You were {user.last_seen_msg()}.",
        "Persistent commands:",
        f"\u200b- //fixturns {'on' if user.use_fixturns else 'off'}",
        f"This message is using API key {jai_req.api_key_index + 1} out of {jai_req.api_key_count}.",
    )
    raise CommandExit()


@command()
def banner(args, user, jai_req, response):
    user.do_show_banner(BANNER_VERSION)
    return response.add_proxy_message(BANNER, "***")


@command(argspec=r"off|on|this", setting="fixturns")
def fixturns(args, user, jai_req, response):
    if jai_req.quiet_commands:
        return response
    return response.add_proxy_message(
        f"Fix request turns {'enabled' if jai_req.use_fixturns else 'disabled'}"
        + (" (for this message only)." if args == "this" else ".")
    )


@command(argspec=r".+")
def dice_roll(args, user, jai_req, response):
    dice = args.replace("p", "+").replace("m", "-")

    match = re.fullmatch(
        r"(\d+)?d(\d+)([+-]\d+)?([a-z])?", dice, re.ASCII | re.IGNORECASE
    )
    if not match:
        return response.add_proxy_message(
            f"Invalid dice syntax `{args}`\nUse the `//help dice` command for more info."
        )

    count = min(max(int(match.group(1) or "1", base=10), 1), 100)
    faces = max(int(match.group(2), base=10), 2)
    extra = int((match.group(3) or "0"), base=10)

    rolls = [randint(1, faces) for _ in range(count)]
    result = sum(rolls) + extra

    extra_str = f" {'-' if extra < 0 else '+'} {abs(extra)}" if extra != 0 else ""

    result_str = f"{dice} roll: {' + '.join(map(str, rolls))}{extra_str}"
    if extra != 0 or len(rolls) > 1:
        result_str += f" = {result}"
    result_str += "."

    jai_req.append_message(
        "user",
        "\n".join(
            [
                "<system>",
                f"  User {result_str}.",
                "</system>",
            ]
        ),
    )

    return response.add_proxy_message(result_str)


@command(argspec=r".+")
def roll(args, user, jai_req, response):
    """Short alias for //dice_roll."""
    return dice_roll(args, user, jai_req, response)


################################################################################


HELP_COMMANDS = """***
You can include one or more commands in your messages, separated by spaces.
Persistent settings are intentionally limited to commands that have a useful effect on normal RP.

- `//aboutme`
  Shows your proxy user ID, usage counter, and persistent command settings.

- `//banner`
  Shows the current proxy banner again.

- `//help commands|dice|multikey|providers`
  Shows information about specific proxy features.

- `//fixturns on|off|this`
  Adds an empty user message when the request would otherwise end on an assistant turn.

- `//dice_roll [count]d(faces)[(p|m)(extra)]`
  Rolls real random dice and inserts the result into the request.

- `//roll [count]d(faces)[(p|m)(extra)]`
  Short alias for `//dice_roll`.
"""


HELP_ADVSETTINGS = """***
# **Generation Settings**

The proxy accepts JanitorAI generation settings and forwards supported values directly to providers.

- **Temperature** controls response randomness.
- **Top P** and **Top K** control token sampling where supported.
- **Frequency Penalty** and **Repetition Penalty** reduce repeated wording where supported.
- **Max Tokens** is controlled by JanitorAI generation settings and is forwarded directly when supported.

Provider support varies, so unsupported settings may be ignored or rejected by the upstream API.
"""


HELP_DICE = """***
# **Dice Commands**

The proxy can generate random numbers itself, so the model does not have to invent them.

Examples:
- `//dice_roll d6`
- `//roll 3d20`
- `//dice_roll 2d6p3`
- `//dice_roll d20m2`

Syntax: `[count]d(faces)[p|m(extra)]`.

The result is inserted into the request as a system-style note for the model and is also shown to you.
"""


HELP_MULTIKEY = """***
# **Multiple API Keys**

You can put multiple API keys, separated by commas, in your proxy settings. \
You can have multiples keys from the same company as well as from different companies.

The proxy switches between them on every request you make \
(chat message, enhance draft, continue reply, auto summarize).

You can use this to get more messages out of keys with limited quotas \
(such as Requests per Day) by distributing your usage across multiple of them.

## **Notes**

There is no limit as to how many keys you can use.

The proxy goes one by one through every key in the same order you put them.

After going through all your keys, the proxy loops to your first key and start over.

Your commands are stored in your first key. \
If you change the first key on your proxy settings, \
then you will have to send your commands again.
"""

HELP_PROVIDERS = """***
# **Model Providers**

This proxy supports different AI companies, called providers, for models and API keys.

The proxy will try to automatically dispatch your requests to \
the appropriate provider, given your models and API keys. \
If that fails, you can always override the proxy's dispatch logic \
by adding the provider at the start of the model name or the API key. \
For example: `google/AQ.Ab8RN...` (Vertex AI API key), `openrouter/z-ai/glm-4.5-air:free` (Z.AI model through OpenRouter).

## **Google AI Studio and Vertex AI** (`google`)

`https://aistudio.google.com/`

All models that start with `gemini-` or `gemma-` will be routed to Google.

All API keys that start with `AIza` will be used with Google models. \
Vertex AI API keys have a different format and you must add `google/` at the start.

## **Cerebras Cloud Inference** (`cerebras`)

`https://cloud.cerebras.ai/`

To use any Cerebras model, you must add `cerebras/` at the start.

All API keys that start with `csk-` will be used with Cerebras models.

## **DeepSeek** (`deepseek`)

`https://platform.deepseek.com/`

To use any DeepSeek model, you must add `deepseek/` at the start.

You must add `deepseek/` at the start of any DeepSeek API key.

## **AMD Radeon Cloud** (`radeon`)

`https://developer.amd.com.cn/radeon/`

To use any AMD Radeon Cloud model, add `radeon/` at the start of the model name.
The model ID is passed through unchanged, for example `radeon/DeepSeek-V4-Flash`.

API keys can be supplied as `radeon/<key>`. AMD Radeon Cloud keys currently use the `rc-` prefix and are also auto-detected.

## **Nvidia NIM** (`nvidia`)

`https://build.nvidia.com/`

To use any model through Nvidia NIM, you must add `nvidia/` first and then the full model name.

All API keys that start with `nvapi-` will be used with Nvidia NIM.

## **OpenRouter** (`openrouter`)

`https://openrouter.ai/`

To use any model through OpenRouter, you must add `openrouter/` first and then the full model name.

All API keys that start with `sk-or-v1-` will be used with OpenRouter.

## **Z.AI** (`z_ai`)

`https://chat.z.ai/`

To use any Z.AI model, you must add `z_ai/` at the start.

You must add `z_ai/` at the start of any Z.AI API key.
"""

HELP = {
    "commands": HELP_COMMANDS,
    "dice": HELP_DICE,
    "multikey": HELP_MULTIKEY,
    "providers": HELP_PROVIDERS,
}


@command(argspec=r".+")
def help(args, user, jai_req, response):
    response.add_proxy_message(
        HELP.get(
            args,
            (
                f"There is no help topic `{args}`.\n"
                + "Available topics are: `commands`, `dice`, `multikey`, `providers`."
            ),
        )
    )
    raise CommandExit()


################################################################################

def parse_message(message: str) -> tuple[list[Command], str]:
    """Parse an message into a list of commands and the message's content."""

    message = message.strip()

    if "//" not in message:
        # No commands to parse
        return [], _stripmultispace(message)

    commands = []
    content = []

    cmd_argcount = 0
    prev_token = ""
    for token in _tokenize(message):
        if not cmd_argcount:
            if prev_token.startswith("//") and (cmd := COMMANDS.get(token.lower())):
                cmd_argcount = cmd["argcount"]
                commands.append(Command(token, func=cmd["func"]))
                content.pop()  # Remove "//" token
            else:
                content.append(token)
        elif token.isspace():
            continue  # Skip white space between a command and its arguments
        elif token.isalnum():
            cmd_argcount -= 1
            commands[-1].args += token  # Valid argument
        else:
            # Invalid token means there was no valid or not enough arguments
            cmd_argcount = 0  # Stop parsing
            content.append(token)  # Add the token as if there wasn't a command
            # The command function will show the appropriate error to the user

        prev_token = token

    return commands, _stripmultispace("".join(content).strip())


################################################################################


def strip_message(raw_message: str) -> str:
    """Clean up the text of a message, meant for model's output."""

    message = _stripproxytext(raw_message.strip("\n")).split("\n")

    if not message:
        return ""

    result = []

    for line in message:
        index = max(line.find("-"), line.find("*"))
        if index != -1 and line[:index].isspace():
            line = line.rstrip()
            is_list = True
        else:
            line = line.strip()
            is_list = False

        if is_list:
            line = line[:index] + _stripmultispace(line[index:])
        else:
            line = _stripmultispace(line)

        result.append(line)

    return "\n".join(result)


################################################################################
