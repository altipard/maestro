from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SegmentRequest(_message.Message):
    __slots__ = ("file", "segment_length", "segment_overlap")
    FILE_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    file: File
    segment_length: int
    segment_overlap: int
    def __init__(self, file: _Optional[_Union[File, _Mapping]] = ..., segment_length: _Optional[int] = ..., segment_overlap: _Optional[int] = ...) -> None: ...

class SegmentResponse(_message.Message):
    __slots__ = ("segments",)
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    segments: _containers.RepeatedCompositeFieldContainer[Segment]
    def __init__(self, segments: _Optional[_Iterable[_Union[Segment, _Mapping]]] = ...) -> None: ...

class File(_message.Message):
    __slots__ = ("name", "content", "content_type")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    content: bytes
    content_type: str
    def __init__(self, name: _Optional[str] = ..., content: _Optional[bytes] = ..., content_type: _Optional[str] = ...) -> None: ...

class Segment(_message.Message):
    __slots__ = ("text",)
    TEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    def __init__(self, text: _Optional[str] = ...) -> None: ...
