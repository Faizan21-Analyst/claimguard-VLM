from claimguard.vlm.base import VLMBackend

AVAILABLE = "groq, local_ft, local_base, remote_ft"


def _build(backend: str) -> VLMBackend:
    """One plain backend by name. Imports are lazy so unused backends cost nothing."""
    if backend == "groq":
        from claimguard.vlm.groq_backend import GroqVLM

        return GroqVLM()

    if backend in ("local_ft", "local_base"):
        from claimguard.vlm.local_backend import LocalQwenVLM

        return LocalQwenVLM(use_adapter=(backend == "local_ft"))

    if backend == "remote_ft":
        from claimguard.vlm.remote_backend import RemoteFinetunedVLM

        return RemoteFinetunedVLM()

    raise ValueError(f"Unknown VLM backend {backend!r}. Available: {AVAILABLE}")


def get_vlm(backend: str | None = None) -> VLMBackend:
    """Build a VLM backend.

    With NO name (what the API does) the settings decide, and "remote_ft" is wrapped so that, when the GPU
    service is unavailable, the configured fallback answers instead (VLM_REMOTE_FALLBACK, "none" to disable).

    With an explicit name (evaluation scripts, tests) you get exactly that backend and never a substitute, so a
    score can never quietly belong to a different model than the one you asked for.
    """
    if backend is not None:
        return _build(backend)

    from configs.settings import load_settings

    s = load_settings().vlm
    primary = _build(s.backend)

    fallback = s.remote_fallback.strip().lower()
    if s.backend == "remote_ft" and fallback not in ("", "none"):
        from claimguard.vlm.fallback import FallbackVLM

        return FallbackVLM(primary, _build(fallback), cooldown_s=s.remote_cooldown_s)
    return primary
