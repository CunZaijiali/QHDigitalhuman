"""Session layer: timing, output sinks and action rotation."""

from .session import StreamResult, stream_action
from .sinks import CompositeSink, FrameSink, VirtualCameraSink, create_sink, create_sinks
from .streaming import RTMPOutput
from .transition import ActionTransition

__all__ = [
    "ActionTransition",
    "CompositeSink",
    "FrameSink",
    "RTMPOutput",
    "StreamResult",
    "VirtualCameraSink",
    "create_sink",
    "create_sinks",
    "stream_action",
]
