import json
import re

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def extract_json(text: str) -> dict:
    """Return the first JSON object found in model text.

    Handles <think> blocks, ``` fences and chatter around the JSON.
    Raises ValueError if no valid JSON object can be found.
    """
    if "<think>" in text and "</think>" not in text:
        raise ValueError("reasoning block was never closed (output probably truncated)")

    text = _THINK_BLOCK.sub("", text)
    text = _FENCE.sub("", text)

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in model output")

    data = json.loads(text[start : end + 1])  # JSONDecodeError is a ValueError
    if not isinstance(data, dict):
        raise ValueError("JSON output is not an object")
    return data