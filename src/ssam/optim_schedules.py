# src/lin_sgd/optim_schedules.py
from typing import Callable
import math

def constant_lr(lr: float) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return lr
    return schedule

def step_decay_lr(lr0: float, delta: float, eta_sched: Callable[[int], float]) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return (1-delta)/lr0 * eta_sched(k)**2 
    return schedule


def bound_lr(lr: float, delta: float, eta_sched: Callable[[int], float]) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return (1-delta)/lr * eta_sched(k)**2 
    return schedule

def inverse_time_eta(eta0: float, alpha:float) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return eta0 / (1.0 + k)**(alpha)
    return schedule
