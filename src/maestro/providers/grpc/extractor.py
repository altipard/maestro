"""gRPC extractor provider.

Sends files over gRPC and receives extracted text (markdown).
"""

from __future__ import annotations

import asyncio

import grpc

from maestro.core.models import FileData

from ._proto import extractor_pb2, extractor_pb2_grpc

_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB


class Extractor:
    """gRPC client for an extractor service."""

    def __init__(self, url: str = "", **_: object) -> None:
        if not url:
            raise ValueError("extractor requires a gRPC url (e.g. grpc://localhost:50051)")

        address = url.removeprefix("grpc://")

        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
            ],
        )
        self._stub = extractor_pb2_grpc.ExtractorStub(self._channel)

    async def extract(self, file: FileData) -> str:
        req = extractor_pb2.ExtractRequest(
            file=extractor_pb2.File(
                name=file.name,
                content=file.content,
                content_type=file.content_type,
            ),
        )

        resp = await asyncio.to_thread(self._stub.Extract, req)
        return resp.text
