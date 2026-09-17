"""Agent implementations for controlled comparison."""

from .bad_agents import BadAgentAlwaysCancel, BadAgentChecksButIgnores
from .scripted_agent import ScriptedGoodAgent

__all__ = ["ScriptedGoodAgent", "BadAgentAlwaysCancel", "BadAgentChecksButIgnores"]

