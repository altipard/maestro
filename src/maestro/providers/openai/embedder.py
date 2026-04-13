from __future__ import annotations

from openai import AsyncOpenAI

from maestro.core.models import Embedding, EmbedOptions, Usage
from maestro.providers.registry import provider


@provider("openai", "embedder")
class Embedder:
    """OpenAI embeddings provider."""

    def __init__(self, url: str = "", token: str = "", model: str = "") -> None:
        self._url = url or "https://api.openai.com/v1/"
        self._token = token
        self._model = model
        self._client = AsyncOpenAI(base_url=self._url.rstrip("/") + "/", api_key=self._token)

    async def embed(
        self,
        texts: list[str],
        options: EmbedOptions | None = None,
    ) -> Embedding:
        options = options or EmbedOptions()

        params: dict = {
            "model": self._model,
            "input": texts,
            "encoding_format": "float",
        }

        if options.dimensions is not None:
            params["dimensions"] = options.dimensions

        response = await self._client.embeddings.create(**params)

        result = Embedding(model=self._model)

        if response.usage and response.usage.prompt_tokens:
            result = result.model_copy(update={
                "usage": Usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=0,
                ),
            })

        embeddings = [item.embedding for item in response.data]
        result = result.model_copy(update={"embeddings": embeddings})

        return result
