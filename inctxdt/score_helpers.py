from dataclasses import dataclass, field
from typing import Dict


@dataclass
class EvalScore:
    eval_score: float
    eval_score_std: float
    normalized_score: float
    normalized_score_std: float


@dataclass
class EvalScores:
    target: Dict[str, EvalScore] = field(default_factory=dict)

    @property
    def mean_eval_score(self):
        return sum([score.eval_score for score in self.target.values()]) / len(self.target)

    @property
    def mean_normalized_score(self):
        return sum([score.normalized_score for score in self.target.values()]) / len(self.target)


@dataclass
class BestScore:
    eval: float = -float("inf")
    norm: float = -float("inf")
    step: int = -1

    def update(self, scores: EvalScores, step: int):
        if scores.mean_eval_score > self.eval:
            self.eval = scores.mean_eval_score
            self.norm = scores.mean_normalized_score
            self.step = step
