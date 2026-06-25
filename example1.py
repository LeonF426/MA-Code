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
from ssam.data import sample_gaussian_linear, load_fixed_dataset,create_fixed_dataset, make_batch_sampler_from_fixed_dataset
from ssam.trainers import sgd_with_weight_noise, gradient_descent_train
from ssam.optim_schedules import constant_lr, inverse_time_eta, lr_convergence
from ssam.plotting import plot_training_history
from ssam.losses import compute_initial_LR, exact_L_R_for_diagonal_model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = 10
    eta0= 2

    layer_specs = [
        {"type": "diag",  "in_dim": d, "out_dim": d},
        {"type": "diag", "in_dim": d, "out_dim": d},
        {"type": "diag", "in_dim": d, "out_dim": d}
    ]
    L= len(layer_specs)
    dataset_path = "./src/ssam/data/"


    model = MixedLinearNet(layer_specs)

    w_star = torch.randn(d, device=device)
    # Here we decide if we do 
    # def data_sampler(batch_size: int):
    #     return sample_gaussian_linear(batch_size, d, device=device, w_star=w_star, noise_std=0.12)
    X,Y,_ = load_fixed_dataset(dataset_path + f"train_dataset_d{d}.pt",device=device)

    data_sampler = make_batch_sampler_from_fixed_dataset(X,Y)

    eta_sched = inverse_time_eta(eta0=eta0, alpha=1/(L))
    
    delta= 0.5
    C = math.sqrt(L)*(2+2*math.sqrt(L)*d + 5*math.sqrt(L-1))

    # LR0=compute_initial_LR(model=model,
    #                       data_sampler=data_sampler,
    #                       eta0=eta0,
    #                       batch_size=10000,
    #                       device=device) 
    #
    # print("first loss :",LR0)

    lr_sched =  lr_convergence(const=C,delta= delta, eta_sched=eta_sched)
    history = gradient_descent_train(
        model=model,
        data_sampler=data_sampler,
        n_steps=4000,
        lr_schedule=lr_sched,
        eta_schedule=eta_sched,
        batch_size=10000,
        device=device,
    )
    print("true param:",w_star)
        # After training, read parameters:
    final_params = []
    for p in model.parameters():
        final_params.append(p.detach().cpu().clone())  # store a copy

    # Example: print norms or save
    for i, p in enumerate(final_params):
        print(f"Layer {i} param norm:", p.norm().item())
    print(final_params)
    # torch.save({"state_dict": model.state_dict()}, "final_model.pt")
    plot_training_history(history, title="./plots/gd/GD_first.png")
    print(history["balancedness"][0])

    print(history["balancedness"][-1])
if __name__ == "__main__":
    main()
