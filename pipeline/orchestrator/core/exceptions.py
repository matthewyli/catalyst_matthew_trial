from __future__ import annotations


class OrchestratorError(Exception):
    """Base orchestrator exception."""


class ToolLoadError(OrchestratorError):
    """Raised when a tool cannot be imported or validated."""


class PipelineExecutionError(OrchestratorError):
    """Raised when pipeline execution fails."""
