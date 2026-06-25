# src/lin_sgd/optim_schedules.py
from typing import Callable, List
import math
import numpy as np

def constant_lr(lr: float) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return lr
    return schedule

def step_decay_lr(lr0: float, delta: float, eta_sched: Callable[[int], float]) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return (1-delta)/lr0 * eta_sched(k)**2 
    return schedule


def lr_convergence(const: float, delta: float, eta_sched: Callable[[int], float]) -> Callable[[int, float], float]:
    def schedule(k: int, loss: float) -> float:
        return 2*(1-delta)/(const*loss) * eta_sched(k)**2 
    return schedule


def lr_balancing(L: int, lamb: float) -> Callable[[int,float,float], List[float]]:
    def schedule(k: int, loss: float, eta_k:float) -> dict:
        lr_1 = (eta_k**2)/(4*loss)
        lr_2 = 3*lamb*(eta_k**(2*L-2)) * lr_1**2
        lr_3 = 1/(lamb *eta_k**(2*L-2))
        lr = np.min([lr_1,lr_2,lr_3])
        return  {"lr": lr, "lr_1": lr_1, "lr_2": lr_2, "lr_3": lr_3}
    return schedule


def bound_lr(lr: float, delta: float, eta_sched: Callable[[int], float]) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return (1-delta)/lr * eta_sched(k)**2 
    return schedule

def inverse_time_eta(eta0: float, alpha:float) -> Callable[[int], float]:
    def schedule(k: int) -> float:
        return eta0 / (1.0 + k)**(alpha)
    return schedule
