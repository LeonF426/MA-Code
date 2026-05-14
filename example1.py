import sys
import os
import math
# Add the project's src/ directory to sys.path
ROOT = os.path.dirname(__file__)
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
# scripts/train_sgd_noisy.py
# train_sgd_noisy.py
import torch

from ssam.models import MixedLinearNet
from ssam.data import sample_gaussian_linear
from ssam.trainers import sgd_with_weight_noise, gradient_descent_train
from ssam.optim_schedules import constant_lr, inverse_time_eta, step_decay_lr
from ssam.plotting import plot_training_history
from ssam.losses import compute_initial_LR, exact_L_R_for_diagonal_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 4

    layer_specs = [
        {"type": "diag",  "in_dim": d, "out_dim": d},
        {"type": "diag", "in_dim": d, "out_dim": d},
        {"type": "diag", "in_dim": d, "out_dim": d}
    ]
    L=3
    model = MixedLinearNet(layer_specs)

    w_star = torch.randn(d, device=device)

    def data_sampler(batch_size: int):
        return sample_gaussian_linear(batch_size, d, device=device, w_star=w_star, noise_std=0.12)

    eta_sched = inverse_time_eta(eta0=0.5, alpha=1/2)
    
    C = math.sqrt(L)*(2+2*math.sqrt(L)*d + 2*math.sqrt(L-1))
    LR0=compute_initial_LR(model=model,
                          data_sampler=data_sampler,
                          eta0=0.5,
                          batch_size=10000,
                          device=device) 

    print("first loss :",LR0)
    lr_sched =  step_decay_lr(lr0= (C*LR0)/2,delta=1/2, eta_sched=eta_sched)
    history = gradient_descent_train(
        model=model,
        data_sampler=data_sampler,
        n_steps=400,
        lr_schedule=lr_sched,
        eta_schedule=eta_sched,
        batch_size=10000,
        device=device,
    )

    plot_training_history(history, title="./plots/gd/GD_first.png")

if __name__ == "__main__":
    main()
