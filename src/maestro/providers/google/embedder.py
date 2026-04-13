"""Google Gemini embeddings provider.

Uses google-genai SDK for embedding generation.
"""

from __future__ import annotations

from google import genai
from google.genai import types

from maestro.core.models import Embedding, EmbedOptions
from maestro.providers.registry import provider


@provider("google", "embedder")
class Embedder:
    """Google Gemini embeddings provider."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._token = token
        self._model = model
        self._client = genai.Client(api_key=self._token)

    async def embed(
        self,
        texts: list[str],
        options: EmbedOptions | None = None,
    ) -> Embedding:
        options = options or EmbedOptions()

        contents = [types.Content(role="user", parts=[types.Part(text=t)]) for t in texts]

        config: dict = {}
        if options.dimensions is not None:
            config["output_dimensionality"] = options.dimensions

        embed_config = types.EmbedContentConfig(**config) if config else None

        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=contents,
            config=embed_config,
        )

        result = Embedding(model=self._model)

        vectors: list[list[float]] = []
        if resp.embeddings:
            for emb in resp.embeddings:
                vectors.append(list(emb.values))

        result = result.model_copy(update={"embeddings": vectors})

        return result
