"""Round-snapshot image teacher used during refinement."""

from copy import deepcopy
from typing import Any, Dict

import torch
from torch import nn


class EMATeacher:
    """Track an accepted student, copying it exactly by default.

    Refinement updates this object once after each accepted round, after the
    trainer has restored that round's best epoch.  Consequently, ``decay=0``
    is the faithful default round-teacher operation.  A nonzero decay remains
    available for explicitly named sensitivity analyses.
    """

    def __init__(self, student: nn.Module, decay: float = 0.0) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        self.decay = decay
        self.model = deepcopy(student).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, student: nn.Module) -> None:
        teacher_state = self.model.state_dict()
        student_state = student.state_dict()
        if teacher_state.keys() != student_state.keys():
            raise ValueError("teacher and student state dictionaries differ")
        for name, teacher_value in teacher_state.items():
            student_value = student_state[name].detach().to(teacher_value.device)
            if torch.is_floating_point(teacher_value):
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "model": self.model.state_dict()}

    def load_state_dict(self, values: Dict[str, Any]) -> None:
        self.decay = float(values["decay"])
        self.model.load_state_dict(values["model"])
