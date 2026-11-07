"""Discrete hidden Markov models with explicit inference and training audits."""
from .core import fit,infer,viterbi,validate_model
__all__=['fit','infer','viterbi','validate_model']
