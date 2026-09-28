from __future__ import annotations

from typing import TYPE_CHECKING

from gamspy._execution.base import ExecutionEngine
from gamspy._execution.socket_engine import SocketEngine

if TYPE_CHECKING:
    from gamspy import Container


def create_execution_engine(container: Container) -> ExecutionEngine:
    return SocketEngine(container)


__all__ = ["ExecutionEngine", "SocketEngine", "create_execution_engine"]
