"""Cliente LLM para respuestas en lenguaje natural (Claude / Anthropic SDK).

Usa el SDK oficial `anthropic` con import perezoso. Si el SDK no está instalado
o no hay clave de API, `available` es False y los llamadores usan un fallback de
plantilla, de modo que el sistema funciona offline.

El modelo por defecto es configurable; se usa Claude Opus 4.8 (`claude-opus-4-8`)
salvo que se indique otro en la configuración.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-4-8"


class LLMClient:
    """Envoltorio fino sobre el SDK de Anthropic para generar texto."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None,
                 max_tokens: int = 1024) -> None:
        self.model = model or DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

    @property
    def available(self) -> bool:
        """True si hay clave y el SDK se puede importar."""
        if not self._api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False
        return True

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def generate(self, prompt: str, system: str | None = None,
                 max_tokens: int | None = None) -> str:
        """Genera texto. Lanza RuntimeError si el cliente no está disponible."""
        if not self.available:
            raise RuntimeError("LLM no disponible (falta SDK anthropic o ANTHROPIC_API_KEY)")
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        message = client.messages.create(**kwargs)
        return "".join(block.text for block in message.content if block.type == "text")

    async def agenerate(self, prompt: str, system: str | None = None,
                        max_tokens: int | None = None) -> str:
        """Versión async (usa AsyncAnthropic)."""
        if not self.available:
            raise RuntimeError("LLM no disponible")
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self._api_key)
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        message = await client.messages.create(**kwargs)
        return "".join(block.text for block in message.content if block.type == "text")
