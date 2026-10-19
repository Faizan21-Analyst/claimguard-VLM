from claimguard.vlm.base import VLMBackend


def get_vlm(backend: str | None = None) -> VLMBackend:
    """Build a VLM backend by name. Imports are lazy so unused backends cost nothing."""
    if backend is None:
        from configs.settings import load_settings

        backend = load_settings().vlm.backend

    if backend == "groq":
        from claimguard.vlm.groq_backend import GroqVLM

        return GroqVLM()

    if backend in ("local_ft", "local_base"):
        from claimguard.vlm.local_backend import LocalQwenVLM

        return LocalQwenVLM(use_adapter=(backend == "local_ft"))

    raise ValueError(f"Unknown VLM backend {backend!r}. Available: groq, local_ft, local_base")