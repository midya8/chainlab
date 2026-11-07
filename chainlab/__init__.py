"""Discrete hidden Markov models with explicit inference and training audits."""
from .core import fit,infer,viterbi,validate_model
from .online import filter_sequence,forecast
__all__=['fit','infer','viterbi','validate_model','filter_sequence','forecast']
