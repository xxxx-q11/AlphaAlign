"""LLM Provider Adapter Module

Uses lazy/safe imports to prevent the entire module from being unavailable due to missing dependencies
"""

_IMPORT_ERRORS = {}

try:
    from .qwen_agent import QwenAgent
except ImportError as e:
    QwenAgent = None
    _IMPORT_ERRORS['QwenAgent'] = str(e)

try:
    from .claude_agent import ClaudeAgent
except ImportError as e:
    ClaudeAgent = None
    _IMPORT_ERRORS['ClaudeAgent'] = str(e)

try:
    from .google_agent import GoogleAgent
except ImportError as e:
    GoogleAgent = None
    _IMPORT_ERRORS['GoogleAgent'] = str(e)

try:
    from .openai_agent import OpenAIAgent
except ImportError as e:
    OpenAIAgent = None
    _IMPORT_ERRORS['OpenAIAgent'] = str(e)

try:
    from .third_party_agent import ThirdPartyAgent
except ImportError as e:
    ThirdPartyAgent = None
    _IMPORT_ERRORS['ThirdPartyAgent'] = str(e)

__all__ = [
    "QwenAgent",
    "ClaudeAgent",
    "GoogleAgent",
    "OpenAIAgent",
    "ThirdPartyAgent",
    "get_import_errors",
]


def get_import_errors():
    """Get import error information for debugging"""
    return _IMPORT_ERRORS.copy()
