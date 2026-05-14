
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearDeepNet(nn.Module):
    """
    Purely linear L-layer network with optional Gaussian noise on weights
    and support for diagonal layers.

    Args:
        layer_shapes: list of (in_dim, out_dim) tuples or "diag" strings.
                      - Tuple (din, dout): full dense Linear(din, dout)
                      - "diag": diagonal layer, param as vector of len(min(prev_out, din))
        eta: std of Gaussian noise added to each weight at every forward call.
        L: depth of network, must equal len(layer_shapes).
    """
    def __init__(self, layer_shapes, eta=0.0, L=None, bias=True, device="cpu",prev_dout=None):
        super().__init__()
        if L is None:
            L = len(layer_shapes)
        assert L == len(layer_shapes), "L must equal len(layer_shapes)"
        self.L = L
        self.eta = eta
        self.device = torch.device(device)

        layers = []
        #prev_dout = None
        for i, shape in enumerate(layer_shapes):
            if isinstance(shape, str) and shape == "diag":
                # Diagonal layer: infer dim from prev layer's dout (or next if first)
                if prev_dout is None:
                    # First layer: use next layer's din (will be set later)
                    raise ValueError(
                        '"diag" as first layer requires later layers to define din.'
                    )
                diag_dim = prev_dout
                layer = _DiagLayer(diag_dim, bias=bias)
                dout = diag_dim  # output dim unchanged
            else:
                # Full dense layer
                assert isinstance(shape, tuple) and len(shape) == 2
                din, dout = shape
                layer = nn.Linear(din, dout, bias=bias)
                # Optional: nice initialization for linear regression
                nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if bias:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.bias, -bound, bound)

            layers.append(layer)
            prev_dout = dout  # for next diag inference

        self.layers = nn.ModuleList(layers)
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass with additive Gaussian noise on weights (if eta > 0).
        """
        x = x.to(self.device)
        for layer in self.layers:
            if isinstance(layer, _DiagLayer):
                x = layer(x)
            else:
                # Dense layer
                W = layer.weight
                b = layer.bias

                if self.eta > 0.0 and self.training:
                    noise = torch.randn_like(W) * self.eta
                    W_eff = W + noise
                else:
                    W_eff = W

                x = x @ W_eff.t()
                if b is not None:
                    x = x + b

        return x

    def apply_weight_noise_inplace(self):
        """
        Perturb stored parameters in-place (for diffusion semantics).
        """
        if self.eta <= 0.0:
            return
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, _DiagLayer):
                    layer.diag_param.add_(torch.randn_like(layer.diag_param) * self.eta)
                else:
                    for p in layer.parameters():
                        p.add_(torch.randn_like(p) * self.eta)

# class DiagonalDeepNet(LinearDeepNet):  # inherits everything
#     """
#     Diagonal-only version: layer_shapes must be ["diag"] * L.
#     """
#     def __init__(self, layer_shapes, eta=0.0, L=None, bias=True, device="cpu"):
#         assert all(s == "diag" for s in layer_shapes), "DiagonalDeepNet requires all 'diag'"
#         if L is None:
#             L = len(layer_shapes)
#         assert L == len(layer_shapes)
#
#         # Use single d from first layer (all must be square)
#         d = self.d = self._infer_diag_dim(layer_shapes[0])  # or pass d explicitly
#
#         super().__init__(layer_shapes, eta, L, bias, device)  # but override layers
#
#         # Override: only diagonal params
#         self.layers = nn.ModuleList([_DiagLayer(d, bias=bias) for _ in range(L)])
#         self.to(self.device)

class _DiagLayer(nn.Module):
    """
    Internal: efficient diagonal scaling layer.

    Params only on diagonal (vector of length D).
    Handles rectangular: projects to D then scales and pads/expands to dout.
    But for simplicity here, we assume square (din == dout == D).
    """
    def __init__(self, D, bias=True):
        super().__init__()
        self.D = D
        self.diag_param = nn.Parameter(torch.ones(D))
        self.bias = nn.Parameter(torch.zeros(D)) if bias else None

        # Nice init for diag
        nn.init.kaiming_uniform_(self.diag_param.view(D, 1), a=math.sqrt(5))

    def forward(self, x):
        # Assume x is (..., D)
        assert x.shape[-1] == self.D
        x = x * self.diag_param
        if self.bias is not None:
            x = x + self.bias
        return x


    def apply_weight_noise_inplace(self):
        """
        Alternative interface: perturb stored parameters in-place.

        This is *not* used in forward(), but you may call it explicitly if
        you want true parameter diffusion between steps instead of noisy
        forward passes only. We guard it with no_grad() as recommended.[web:6][web:9]
        """
        if self.eta <= 0.0:
            return
        with torch.no_grad():
            for p in self.parameters():
                p.add_(torch.randn_like(p) * self.eta)


def generate_linear_regression_data(
    n_samples=1000,
    d_in=5,
    d_out=1,
    layer_shapes=None,  # NEW: for multi-layer diagonal true weights
    w_true=None,
    b_true=None,
    noise_std=0.1,
    device="cpu",
):
    """
    NEW: if layer_shapes=["diag"]*L, generate diagonal W_true product.
    """
    device = torch.device(device)
    x = torch.randn(n_samples, d_in, device=device)

    if layer_shapes is None or w_true is not None:
        # Original single-layer case
        if w_true is None:
            w_true = torch.randn(d_out, d_in, device=device)
        if b_true is None:
            b_true = torch.randn(d_out, device=device)
        y = x @ w_true.t() + b_true
    else:
        # Multi-layer diagonal true network
        assert all(s == "diag" for s in layer_shapes)
        L = len(layer_shapes)
        d = d_in  # assume square diagonals
        diag_vecs = [torch.randn(d, device=device) for _ in range(L)]
        if b_true is None:
            b_true = torch.zeros(d_out, device=device)

        # Apply full diagonal network
        y = x
        for diag_vec in diag_vecs:
            y = y * diag_vec  # diag mul
        y = y[:, :d_out] + b_true  # final projection + bias

        # Effective W_true = product of diagonals (for inspection)
        w_true = torch.diag(diag_vecs[0])
        for dv in diag_vecs[1:]:
            w_true = torch.diag(dv) @ w_true

    if noise_std > 0.0:
        y = y + noise_std * torch.randn_like(y)

    return x, y, w_true, b_true

def train_model(
    model,
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    lr=1e-2,
    weight_decay=0.0,
    num_epochs=200,
    batch_size=64,
    use_inplace_noise=False,
):
    """
    Simple training loop for MSE regression with Adam and optional validation.
    Collects loss curves for plotting later.[web:5][web:10][web:13]
    """
    device = model.device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    if x_val is not None:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = x_train.shape[0]
    idx = torch.arange(n_train, device=device)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        # Shuffle indices
        perm = idx[torch.randperm(n_train)]
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = perm[start:end]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            optimizer.zero_grad()

            if use_inplace_noise and model.eta > 0.0:
                # Diffuse parameters before each step (alternative semantics)
                model.apply_weight_noise_inplace()

            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        epoch_loss /= max(num_batches, 1)
        train_losses.append(epoch_loss)

        # Validation loss
        if x_val is not None:
            model.eval()
            with torch.no_grad():
                preds_val = model(x_val)
                val_loss = criterion(preds_val, y_val).item()
            val_losses.append(val_loss)
        else:
            val_losses.append(None)

        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            if x_val is not None:
                print(
                    f"Epoch {epoch+1:4d}/{num_epochs}: "
                    f"train_loss={epoch_loss:.4e}, val_loss={val_loss:.4e}"
                )
            else:
                print(
                    f"Epoch {epoch+1:4d}/{num_epochs}: "
                    f"train_loss={epoch_loss:.4e}"
                )

    return train_losses, val_losses


def plot_losses(train_losses, val_losses=None):
    """
    Plot training (and validation) loss against epochs using matplotlib.[web:10][web:13]
    """
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    if val_losses is not None and any(v is not None for v in val_losses):
        v = [vv for vv in val_losses if vv is not None]
        plt.plot(range(len(v)), v, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main_example():
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Problem dimensions
    d_in = 8
    d_out = 1
    n_train = 1000
    n_val = 200

    # Generate synthetic data
    x_train, y_train, W_true, b_true = generate_linear_regression_data(
        n_samples=n_train,
        d_in=d_in,
        d_out=d_out,
        noise_std=0.1,
        device=device,
    )
    x_val, y_val, _, _ = generate_linear_regression_data(
        n_samples=n_val,
        d_in=d_in,
        d_out=d_out,
        w_true=W_true,
        b_true=b_true,
        noise_std=0.1,
        device=device,
    )

    L = 4
    # Example: dense -> diag -> dense -> diag
    # layer_shapes = [
    # (d_in, 10),      # dense
    # "diag",          # diagonal on dim=10
    # (10, 5),         # dense
    # "diag",          # diagonal on dim=5
    # ]
    # L = len(layer_shapes)
    layer_shapes = ["diag"]*L
    eta = 0.05  # std of Gaussian noise on weights each forward

    model = LinearDeepNet(layer_shapes=layer_shapes, eta=eta, L=L, device=device,prev_dout=d_in)
    print(model)
    print(model.layers[0].diag_param)

    # Train
    # train_losses, val_losses = train_model(
    #     model,
    #     x_train,
    #     y_train,
    #     x_val=x_val,
    #     y_val=y_val,
    #     lr=1e-2,
    #     num_epochs=200,
    #     batch_size=64,
    #     use_inplace_noise=False,  # set True to use parameter diffusion
    # )
    #
    # # Plot loss curves
    # plot_losses(train_losses, val_losses)
    #
    # # Inspect a few predictions vs true targets
    # model.eval()
    # with torch.no_grad():
    #     x_sample = x_val[:5]
    #     y_true_sample = y_val[:5]
    #     y_pred_sample = model(x_sample)
    #
    # print("\nSome example predictions:")
    # for i in range(x_sample.shape[0]):
    #     print(
    #         f"y_true = {y_true_sample[i].cpu().numpy()}, "
    #         f"y_pred = {y_pred_sample[i].cpu().numpy()}"
    #     )
    #
    # # Optional: check learned effective linear map for the full network
    # # by propagating an identity basis.
    # with torch.no_grad():
    #     eye = torch.eye(d_in, device=device)
    #     # Turn off noise for inspecting the deterministic map
    #     model.eval()
    #     old_eta = model.eta
    #     model.eta = 0.0
    #     W_eff = model(eye).t()  # shape (d_out, d_in)
    #     model.eta = old_eta
    # print("\nTrue W and learned effective W (noiseless forward):")
    # print("W_true:\n", W_true.cpu().numpy())
    # print("W_eff:\n", W_eff.cpu().numpy())


if __name__ == "__main__":
    main_example()

