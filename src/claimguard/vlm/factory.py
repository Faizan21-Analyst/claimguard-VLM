from claimguard.vlm.base import VLMBackend


def get_vlm(backend: str | None = None) -> VLMBackend:
    """Build a VLM backend by name. Imports are lazy so unused backends cost nothing."""
    if backend is None:
        from configs.settings import load_settings

        backend = load_settings().vlm.backend

    if backend == "groq":
        from claimguard.vlm.groq_backend import GroqVLM

        return GroqVLM()

    raise ValueError(f"Unknown VLM backend {backend!r}. Available: groq")