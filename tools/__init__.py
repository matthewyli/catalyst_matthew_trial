from __future__ import annotations

from typing import Iterable, List

from .asknews_tool import AskNewsImpactTool
from .base import BaseTool
from .execution_tool import ExecutionAdapterTool
from .textql_context_tool import TextQLPrimerTool, TextQLRuntimeTool
from .tvl_tool import TVLGrowthTool
from .volatility_tool import VolatilityPercentileTool

__all__ = [
    'TextQLPrimerTool',
    'TextQLRuntimeTool',
    'AskNewsImpactTool',
    'TVLGrowthTool',
    'VolatilityPercentileTool',
    'ExecutionAdapterTool',
    'default_tools',
]


def default_tools() -> List[BaseTool]:
    """Instantiate the default set of tools used by the strategy pipeline."""

    tools: Iterable[BaseTool] = (
        TextQLPrimerTool(),
        TextQLRuntimeTool(),
        AskNewsImpactTool(),
        TVLGrowthTool(),
        VolatilityPercentileTool(),
        ExecutionAdapterTool(),
    )
    return list(tools)
