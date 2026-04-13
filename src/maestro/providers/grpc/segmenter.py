"""gRPC segmenter provider.

Sends text over gRPC and receives segmented chunks.
"""

from __future__ import annotations

import asyncio

import grpc

from maestro.core.models import Segment

from ._proto import segmenter_pb2, segmenter_pb2_grpc

_MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100 MB


class Segmenter:
    """gRPC client for a segmenter service."""

    def __init__(
        self,
        url: str = "",
        segment_length: int = 1000,
        segment_overlap: int = 0,
        **_: object,
    ) -> None:
        if not url:
            raise ValueError("segmenter requires a gRPC url (e.g. grpc://localhost:50051)")

        address = url.removeprefix("grpc://")

        self._channel = grpc.insecure_channel(
            address,
            options=[
                ("grpc.max_receive_message_length", _MAX_MESSAGE_SIZE),
                ("grpc.max_send_message_length", _MAX_MESSAGE_SIZE),
            ],
        )
        self._stub = segmenter_pb2_grpc.SegmenterStub(self._channel)
        self._segment_length = segment_length
        self._segment_overlap = segment_overlap

    async def segment(self, text: str) -> list[Segment]:
        req = segmenter_pb2.SegmentRequest(
            file=segmenter_pb2.File(
                name="input.txt",
                content=text.encode("utf-8"),
                content_type="text/plain",
            ),
            segment_length=self._segment_length,
            segment_overlap=self._segment_overlap,
        )

        resp = await asyncio.to_thread(self._stub.Segment, req)

        return [Segment(text=s.text) for s in resp.segments]
