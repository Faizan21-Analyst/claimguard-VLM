"""The fine-tuned Qwen3-VL-2B running on a remote GPU service.

The service is a Gradio app with one endpoint: a photo goes in, the model's raw text comes out (service/app.py,
normally a Hugging Face ZeroGPU Space). This class only moves the photo there and the text back. Parsing and
validation stay in VLMBackend.analyze(), so the remote model is judged by exactly the same rules as every other.

Honesty rule: if the service cannot be reached or has no GPU time left, this raises ModelUnavailable. It never
returns an "unclear" finding for that, because a down server is not evidence about a photo.
"""
import logging
import tempfile
import threading
from pathlib import Path

from configs.settings import load_settings
from claimguard.vlm.base import ModelUnavailable, VLMBackend
from claimguard.vlm.images import image_to_png_bytes

logger = logging.getLogger(__name__)

_CLIENTS: dict = {}               # (space, has_token) -> gradio Client; connecting costs a network round trip
_CLIENT_LOCK = threading.Lock()


def _handle_file(path: str):
    """Wrap a local file for upload. Separate function so tests can replace it without gradio_client."""
    from gradio_client import handle_file

    return handle_file(path)


def _connect(space: str, token: str | None):
    key = (space, bool(token))
    with _CLIENT_LOCK:
        if key not in _CLIENTS:
            try:
                from gradio_client import Client
            except ImportError as exc:
                raise ModelUnavailable("gradio_client is not installed (pip install gradio_client).") from exc
            try:
                _CLIENTS[key] = Client(space, token=token, verbose=False)
            except Exception as exc:  # sleeping, renamed, private or offline Space: all mean "not available"
                raise ModelUnavailable(_explain(exc)) from exc
        return _CLIENTS[key]


def _explain(exc: Exception) -> str:
    """A short reason that is safe to log and show. Never includes the raw server text or any token."""
    text = str(exc).lower()
    if "quota" in text:
        return "The remote GPU has no free time left today."
    if isinstance(exc, TimeoutError) or "timed out" in text or "timeout" in text:
        return "The remote GPU did not answer in time."
    return f"The remote GPU service could not be reached ({type(exc).__name__})."


class RemoteFinetunedVLM(VLMBackend):
    """Pass `client` to inject a fake in tests."""

    def __init__(self, client=None):
        s = load_settings().vlm
        if not s.remote_space and client is None:
            raise ValueError("VLM_REMOTE_SPACE is not set: give the name of the GPU Space, e.g. username/claimguard-vlm.")

        self.space = s.remote_space
        self.token = s.remote_token.get_secret_value() if s.remote_token else None
        self.api_name = s.remote_api_name
        self.timeout = s.remote_timeout_s
        self.max_side = s.local_max_side   # the size the model was trained on; the service resizes no further
        self._client = client
        self.name = f"remote_ft:{self.space or 'injected'}"

    def _raw(self, image_path: Path) -> str:
        client = self._client or _connect(self.space, self.token)
        png = image_to_png_bytes(image_path, self.max_side)

        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder) / "photo.png"
            photo.write_bytes(png)

            job = None
            try:
                job = client.submit(_handle_file(str(photo)), api_name=self.api_name)
                result = job.result(timeout=self.timeout)
            except Exception as exc:  # every transport problem, including a daily-quota refusal
                if job is not None:
                    try:
                        job.cancel()
                    except Exception:  # noqa: BLE001 - best effort only
                        pass
                logger.warning("%s: call failed: %s: %s", self.name, type(exc).__name__, exc)
                raise ModelUnavailable(_explain(exc)) from exc

        text = result[0] if isinstance(result, (tuple, list)) and result else result
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model returned an empty response")   # bad output: analyze() retries, then "unclear"
        return text
