"""Woof Behavior Vision worker.

The worker is deliberately evidence-first: perception adapters emit objective observations,
and the fusion layer produces a conservative canonical contract. No adapter has authority
to infer internal emotion or recommend direct dog-to-dog greeting.
"""

from .pipeline import BehaviorVisionPipeline

__all__ = ["BehaviorVisionPipeline"]
