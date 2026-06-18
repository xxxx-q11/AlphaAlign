"""Evaluators module"""
from .correlation_evaluator import CorrelationEvaluator

try:
    from .llm_evaluator import LLMFactorEvaluator, convert_to_bool
except ImportError:
    LLMFactorEvaluator = None
    convert_to_bool = None

__all__ = ['CorrelationEvaluator', 'LLMFactorEvaluator', 'convert_to_bool']
