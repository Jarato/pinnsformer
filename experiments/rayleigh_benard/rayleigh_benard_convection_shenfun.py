import torch.optim
from collections import OrderedDict
import torch
import os
import scipy.io as io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.autograd import grad
import numpy as np
import time
import matplotlib
import argparse
from types import SimpleNamespace
from functools import partial
import wandb
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from pinntorch import *


matplotlib.use("Agg")


hyperparameters = SimpleNamespace(
    number_hidden_layers=15,
    number_hidden_dimension=50,
    learning_rate=0.003,
    epochs=300,
    torch_seed=71,
    numpy_seed=71,
    pde="rayleigh benard convection",
    pde_parameter_ra=1.0,
    pde_parameter_pr=1.0,
    pde_parameter_dt=0.2,
    pde_parameter_domain=[[-1, 1], [0, 2 * np.pi]],
    pde_parameter_discretization_domain=(64, 96),
    pde_parameter_bc_temperature_top=1.0,
    pde_parameter_bc_temperature_bottom=2.0,
    data_noise_variance=0.1,
    moo_method="ls",
    moo_mgda_max_iter=250,
    moo_mgda_stop_crit=1e-5,
    moo_ls_alpha_weight=0.5,
    moo_normalization="norm",  # "norm", "loss", "loss+", "none"
    log_period_validation=1,
    log_period_checkpoint=50,
    log_period_visualization=50,
    base_optimizer="Adam",
    activation="ReLU",
    validation_split=0.2,
    data_path="data_300_z.h5",
    train_data_seconds=60,
    train_data_every=2,
    collocation_pde_every=4,
    collocation_boundary_every=4,
    batch_size=10,  # number of time steps in one batch
    wiggle_weights_variance=0.1,
    wiggle_weights_seed=0,
    init_weights_seed=70,
    torch_device="cuda",
    disable_visualization=False,
    use_third_objective=False,
    load_model=None,  # "pinngroup/rayleigh_benard_convection_new/runs/h3t25bws",
)

# dict hyperparameters: default values
# args argpasre: overwrite default values
# wandb.config: only for wandb


def setup_device(config=hyperparameters):
    device = torch.device(config.torch_device)  # Use 'cuda' for GPU or 'cpu' for CPU
    print(f"Using device: {device}")

    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.manual_seed(config.torch_seed)
    np.random.seed(config.numpy_seed)

    return device


# define nn
class Net(nn.Module):
    def __init__(self, seq_net, name="MLP"):
        super().__init__()
        self.features = OrderedDict()
        for i in range(len(seq_net) - 1):
            self.features["{}_{}".format(name, i)] = nn.Linear(
                seq_net[i], seq_net[i + 1], bias=True
            )
            self.features = nn.ModuleDict(self.features)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        length = len(self.features)
        i = 0
        for name, layer in self.features.items():
            x = layer(x)
            if i == length - 1:
                break
            i += 1
            act = nn.ELU()
            x = act(x)
        return x


def d(f, x):
    return grad(
        f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True
    )[0]


def compute_pde_net(u_v_p_K, x_f, y_f, t_f, Ra, Pr):
    u = u_v_p_K[:, 0:1]
    v = u_v_p_K[:, 1:2]
    p = u_v_p_K[:, 2:3]
    K = u_v_p_K[:, 3:]

    out_1 = d(u, x_f) + d(v, y_f)
    out_2 = (
        d(u, t_f)
        + u * d(u, x_f)
        + v * d(u, y_f)
        + d(p, x_f)
        - pow(Pr / Ra, 0.5) * (d(d(u, x_f), x_f) + d(d(u, y_f), y_f))
        - K
    )
    out_3 = (
        d(v, t_f)
        + u * d(v, x_f)
        + v * d(v, y_f)
        + d(p, y_f)
        - pow(Pr / Ra, 0.5) * (d(d(v, x_f), x_f) + d(d(v, y_f), y_f))
    )
    out_4 = (
        d(K, t_f)
        + u * d(K, x_f)
        + v * d(K, y_f)
        - pow(Ra * Pr, -0.5) * (d(d(K, x_f), x_f) + d(d(K, y_f), y_f))
    )

    return out_1, out_2, out_3, out_4, u, v, p, K


def compute_pde(ux_uy_p_temp, x, y, time, Ra, Pr):
    # ux = ux_uy_p_temp[:, 0:1]
    # uy = ux_uy_p_temp[:, 1:2]
    # p = ux_uy_p_temp[:, 2:3]
    # temperature = ux_uy_p_temp[:, 3:]

    ux = ux_uy_p_temp[0]
    uy = ux_uy_p_temp[1]
    p = ux_uy_p_temp[2]
    temperature = ux_uy_p_temp[3]

    # Momentum equation in x-direction (Navier-Stokes for ux)
    pde_1_ux = (
        df(ux, wrt=time)
        + ux * df(ux, wrt=x)
        + uy * df(ux, wrt=y)
        + df(p, wrt=x)
        - pow(Pr / Ra, 0.5)
        * (df(ux, wrt=x, order=2) + df(ux, wrt=y, order=2))  # Viscous dissipation
        - temperature  # Buoyancy term
    )
    # Momentum equation in y-direction (Navier-Stokes for uy)
    pde_1_uy = (
        df(uy, wrt=time)
        + ux * df(uy, wrt=x)
        + uy * df(uy, wrt=y)
        + df(p, wrt=y)
        - pow(Pr / Ra, 0.5)
        * (df(uy, wrt=x, order=2) + df(uy, wrt=y, order=2))  # Viscous dissipation
    )

    # Energy equation for temperature field
    pde_2 = (
        df(temperature, wrt=time)
        + ux * df(temperature, wrt=x)
        + uy * df(temperature, wrt=y)
        - pow(Ra * Pr, -0.5)
        * (
            df(temperature, wrt=x, order=2) + df(temperature, wrt=y, order=2)
        )  # Thermal diffusion
    )

    # Continuity equation (incompressibility)
    pde_3 = df(ux, wrt=x) + df(uy, wrt=y)

    return pde_1_ux, pde_1_uy, pde_2, pde_3, ux, uy, p, temperature


def compute_boundary_loss(
    model,
    x_boundary,
    y_boundary,
    time_boundary,
    config,
    device,
):
    criterion = torch.nn.MSELoss()

    time_steps = len(time_boundary)

    # [bottom, top, left, right]
    x_f = torch.cat([x.repeat(time_steps) for x in x_boundary]).unsqueeze(1)
    y_f = torch.cat([y.repeat(time_steps) for y in y_boundary]).unsqueeze(1)

    time_f = (
        torch.tensor(
            [time for a in x_boundary for time in time_boundary for _ in range(len(a))],
        )
        .unsqueeze(1)
        .to(device)
    )

    # print(torch.unique(time_f))
    # print(torch.cat((x_f, y_f, time_f), dim=1).detach().cpu().numpy())

    output = model(torch.cat((x_f, y_f, time_f), dim=1))

    def visualize_boundary(x_f, y_f, time_f, temp_f):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            x_f.detach().cpu().numpy(),
            y_f.detach().cpu().numpy(),
            time_f.detach().cpu().numpy(),
            c=temp_f.detach().cpu().numpy(),  # color of points
            cmap="coolwarm",  # color map
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Time")
        fig.colorbar(scatter, ax=ax, label="Temperature")

        plt.savefig("boundary_points.png")

    # ux, uy, p, temperature
    ux_model = output[0]
    uy_model = output[1]
    p_model = output[2]
    temperature_model = output[3]

    # visualize_boundary(x_f, y_f, time_f, temperature_model)

    # time_boundary = np.repeat(
    #     np.arange(0, steps * dt, dt),
    #     (len(x_bottom) + len(x_top) + len(x_left) + len(x_right)),
    # )  # calculate time for boundary points
    # time_boundary = torch.tensor(
    #     time_boundary.astype(np.float32), requires_grad=True
    # ).to(device)

    # Calculate the number of points in each boundary
    num_bottom = len(x_boundary[0]) * time_steps
    num_top = len(x_boundary[1]) * time_steps
    num_left = len(x_boundary[2]) * time_steps
    num_right = len(x_boundary[3]) * time_steps

    # Calculate the cumulative sum of the number of points in each boundary
    cumulative_points = torch.cumsum(
        torch.tensor([num_bottom, num_top, num_left, num_right]), dim=0
    )

    # Calculate the splits
    split_1, split_2, split_3, split_4 = cumulative_points
    temperture_bottom = temperature_model[:split_1]
    temperture_top = temperature_model[split_1:split_2]
    temperture_left = temperature_model[split_2:split_3]
    temperture_right = temperature_model[split_3:split_4]

    # points = torch.cat((x_f, y_f, time_f), dim=1).detach().cpu().numpy()
    # print("bottom", points[:split_1])
    # print("top", points[split_1:split_2])
    # print("left", points[split_2:split_3])
    # print("right", points[split_3:split_4])

    ux_bottom = ux_model[:split_1]
    ux_top = ux_model[split_1:split_2]
    ux_left = ux_model[split_2:split_3]
    ux_right = ux_model[split_3:split_4]

    uy_bottom = uy_model[:split_1]
    uy_top = uy_model[split_1:split_2]
    uy_left = uy_model[split_2:split_3]
    uy_right = uy_model[split_3:split_4]

    # Compute the loss for each boundary
    mse_boundary_temperature_top = criterion(
        temperture_top,
        torch.ones_like(temperture_top) * config.pde_parameter_bc_temperature_top,
    )
    mse_boundary_temperature_bottom = criterion(
        temperture_bottom,
        torch.ones_like(temperture_bottom) * config.pde_parameter_bc_temperature_bottom,
    )

    # Periodic boundary condition
    mse_boundary_temperature_side = criterion(
        temperture_left, temperture_right
    )  # periodic boundary condition, only possible if the collocation points are sampled for both sides
    mse_boundary_u_side = (
        criterion(ux_left, ux_right) + criterion(uy_left, uy_right)
    ) / 2

    # No-slip boundary condition
    mse_boundary_u_top = (
        criterion(ux_top, torch.zeros_like(ux_top))
        + criterion(uy_top, torch.zeros_like(uy_top)) / 2
    )
    mse_boundary_u_bottom = (
        criterion(ux_bottom, torch.zeros_like(ux_bottom))
        + criterion(uy_bottom, torch.zeros_like(uy_bottom))
    ) / 2

    mse_boundary = (
        mse_boundary_temperature_top
        + mse_boundary_temperature_bottom
        + mse_boundary_temperature_side
        + mse_boundary_u_side
        + mse_boundary_u_top
        + mse_boundary_u_bottom
    ) / 6

    return mse_boundary


def compute_pde_loss(
    output,
    x_train,
    y_train,
    time_train,
    ux_train,
    uy_train,
    temperature_train,
    config,
    device,
    model=None,
    fixed_x=None,
    fixed_y=None,
    boundary_x=None,
    boundary_y=None,
):
    criterion = torch.nn.MSELoss()

    distinct_time = np.unique(
        time_train.cpu().detach().numpy()
    )  # Use the same time points as the training data
    time_steps = len(distinct_time)

    if model is not None and fixed_x is not None and fixed_y is not None:
        # time_f = torch.repeat_interleave(
        #     distinct_time, len(x_f) // len(distinct_time)
        # ).unsqueeze(1)
        # repeats = len(x_f) // len(distinct_time)
        # time_f_np = np.repeat(distinct_time, repeats)
        # time_f = (
        #     torch.from_numpy(
        #         time_f_np,
        #     )
        #     .unsqueeze(1)
        #     .to(device)
        # )

        x_f = torch.cat([x.repeat(time_steps) for x in fixed_x]).unsqueeze(1)
        y_f = torch.cat([y.repeat(time_steps) for y in fixed_y]).unsqueeze(1)

        time_f = (
            torch.tensor(
                [
                    time
                    for a in fixed_x
                    for time in distinct_time
                    for _ in range(len(a))
                ],
            )
            .unsqueeze(1)
            .to(device)
        )

        tensors = [x_f, y_f, time_f]

        for tensor in tensors:
            if tensor is not None:
                tensor.requires_grad_(True)
                if tensor.grad is not None:
                    tensor.grad.zero_()

        output = model(torch.cat((x_f, y_f, time_f), dim=1))
    else:
        x_f = x_train
        y_f = y_train
        time_f = time_train  # TODO: Add output model with gradients

    # points = torch.cat((x_f, y_f, time_f), dim=1).detach().cpu().numpy()
    # print(points[:100])

    PDE_1, PDE_2, PDE_3, PDE_4, u_u, u_v, u_p, u_k = compute_pde(
        output,
        x_f,
        y_f,
        time_f,
        config.pde_parameter_ra,
        config.pde_parameter_pr,
    )

    for tensor in tensors:
        if tensor is not None:
            tensor.requires_grad_(False)

    mse_PDE_1 = criterion(PDE_1, torch.zeros_like(PDE_1))
    mse_PDE_2 = criterion(PDE_1, torch.zeros_like(PDE_2))
    mse_PDE_3 = criterion(PDE_3, torch.zeros_like(PDE_3))
    mse_PDE_4 = criterion(PDE_4, torch.zeros_like(PDE_4))
    mse_PDE = (mse_PDE_1 + mse_PDE_2 + mse_PDE_3 + mse_PDE_4) / 4

    if boundary_x is not None and boundary_y is not None:
        mse_boundary = compute_boundary_loss(
            model,
            boundary_x,
            boundary_y,
            distinct_time,
            config,
            device,
        )

        mse_PDE = (mse_PDE + mse_boundary) / 2

    return mse_PDE


def compute_data_loss(
    output,
    x_train,
    y_train,
    time_train,
    ux_train,
    uy_train,
    temperature_train,
    config,
    device,
):
    criterion = torch.nn.MSELoss()

    # ux, uy, p, temperature
    # ux = output[:, 0:1]
    # uy = output[:, 1:2]
    # p = output[:, 2:3]
    # temperature = output[:, 3:4]

    # points = torch.cat((x_train, y_train, time_train), dim=1).detach().cpu().numpy()
    # print(points[:100])

    ux = output[0]
    uy = output[1]
    p = output[2]
    temperature = output[3]

    mse_Data_1 = criterion(ux, ux_train)
    mse_Data_2 = criterion(uy, uy_train)
    mse_Data_3 = criterion(temperature, temperature_train)
    mse_Data = (mse_Data_1 + mse_Data_2 + mse_Data_3) / 3

    return mse_Data


def compute_weight_loss(
    output,
    x_train,
    y_train,
    time_train,
    ux_train,
    uy_train,
    temperature_train,
    config,
    device,
    model=None,
):
    total_squared_sum = 0.0
    total_weights = 0

    for name, param in model.named_parameters():
        if "weight" in name:
            total_squared_sum += torch.sum(param**2)
            total_weights += param.numel()

    mse_weights = total_squared_sum / total_weights

    return mse_weights


def load_model(run_path):
    api = wandb.Api()

    # Access the config from the wandb run
    run = api.run(run_path)
    run_config = run.config
    run_config = SimpleNamespace(**run_config)

    # Download the latest model of the run
    artifacts = run.logged_artifacts()
    model = [artifact for artifact in artifacts if artifact.type == "model"][-1]
    print(f"Downloading model {model.name}...")
    artifact_dir = model.download()

    # Load the model checkpoint
    checkpoint_path = f"{artifact_dir}/model.pth"  # Update this path if necessary

    return run_config, checkpoint_path


def visualize_data(
    x,
    y,
    temperature,
    ux,
    uy,
    domain,
    dt,
    steps,
    vmin=None,
    vmax=None,
    time_stamps=None,
    epoch=0,
    log_name="visualization/rbc",
    config=hyperparameters,
):
    if config.disable_visualization:
        return

    print(f"Visualizing data over {steps} time steps...")

    domainsize = domain[0] * domain[1]

    # use the same x, y for all time frames
    x = x[:domainsize]
    y = y[:domainsize]

    x2d = x.reshape(domain[0], domain[1])
    y2d = y.reshape(domain[0], domain[1])

    # Create a figure
    fig, ax = plt.subplots()

    # Create temperature legend
    if vmin is None:
        vmin = temperature.min()  # Tmin=1
    if vmax is None:
        vmax = temperature.max()  # Tmax=2
    # temperature2d = temperature[0:domainsize].reshape(domain[0], domain[1])
    # cont = ax.contourf(
    #     y2d, x2d, temperature2d, 100, cmap="viridis", vmin=vmin, vmax=vmax
    # )
    # # Create the colorbar
    # cbar = fig.colorbar(cont)

    # Create a Normalize instance
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Create a ScalarMappable instance with the colormap and normalization
    mappable = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

    # Create the colorbar
    cbar = plt.colorbar(mappable, ax=ax, orientation="vertical")

    if time_stamps is not None:
        steps = len(time_stamps)

    # Function to update the plot
    def update(i):
        ax.clear()
        temperature2d = temperature[i * domainsize : (i + 1) * domainsize].reshape(
            domain[0], domain[1]
        )
        ux2d = ux[i * domainsize : (i + 1) * domainsize].reshape(domain[0], domain[1])
        uy2d = uy[i * domainsize : (i + 1) * domainsize].reshape(domain[0], domain[1])

        # y, x for matplotlib [column, row]
        cont = ax.contourf(
            y2d, x2d, temperature2d, 100, cmap="viridis", norm=norm
        )  # 100 contour levels

        quiv = None
        if np.all(ux2d == 0):
            print(f"Zero ux at time {i*dt}")
        else:
            quiv = ax.quiver(y2d, x2d, uy2d, ux2d, color="white", alpha=0.5)

        if time_stamps is not None:
            time = time_stamps[i]
        else:
            time = i * dt
        ax.set_title(f"Time: {time:.1f}")

        return (
            cont,
            quiv,
        )

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=range(steps), interval=20, blit=False
    )  # 20ms interval

    ani.save("rbc.gif", writer="ffmpeg", fps=30)
    # ani.save(log_name + ".gif", writer="pillow", fps=30)

    plt.close(fig)

    wandb.log({log_name: wandb.Video("rbc.gif")}, step=epoch)


def plot_error_over_time(
    predicted_temperatures,
    actual_temperatures,
    dt,
    num_frames,
    domainsize,
    epoch,
    log_name="visualization/error_over_time",
    config=hyperparameters,
):
    if config.disable_visualization:
        return

    print(f"Plotting error over time...")
    # print(f"Predicted temperatures shape: {predicted_temperatures.shape}")
    # print(f"Actual temperatures shape: {actual_temperatures.shape}")
    # print(f"Actual temperatures: {actual_temperatures[:100]}")
    # print(f"Predicted temperatures: {predicted_temperatures[:10]}")

    error_over_time = []

    # Calculate the error at each time step
    for i in range(num_frames):
        start = i * domainsize
        end = (i + 1) * domainsize
        error_over_time.append(
            np.mean(
                np.square(
                    predicted_temperatures[start:end] - actual_temperatures[start:end]
                )
            )
        )

    # Calculate the mean error
    mean_error = np.mean(error_over_time)
    wandb.log({"test/mean_error": mean_error}, step=epoch)

    fig, ax = plt.subplots()

    # Plot the error over time
    ax.plot(np.arange(num_frames) * dt, error_over_time)

    # Plot the mean error
    ax.axhline(y=mean_error, color="r", linestyle="--", label="Mean error")

    ax.set_title("Error over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("MSE")

    wandb.log({log_name: wandb.Image(plt)}, step=epoch)
    plt.close(fig)


def load_data(config, device):
    data_file = config.data_path
    with h5py.File(data_file, "r") as f:
        temperature = torch.tensor(
            f["temperature"][:].astype(np.float32), requires_grad=False
        ).to(device)
        ux = torch.tensor(f["ux"][:].astype(np.float32), requires_grad=False).to(device)
        uy = torch.tensor(f["uy"][:].astype(np.float32), requires_grad=False).to(device)
        x = torch.tensor(f["x"][:].astype(np.float32), requires_grad=False).to(device)
        y = torch.tensor(f["y"][:].astype(np.float32), requires_grad=False).to(device)

        discretization_domain = f.attrs["N"]
        Ra = f.attrs["Ra"]
        Pr = f.attrs["Pr"]
        dt = f.attrs["dt"]
        steps = f.attrs["steps"]
        domain = f.attrs["domain"]

    # domain = (
    #     x.min().item(),
    #     x.max().item(),
    # ), (
    #     y.min().item(),
    #     y.max().item(),
    # )

    discretization_domainsize = discretization_domain[0] * discretization_domain[1]
    time = np.repeat(
        np.arange(0, steps * dt, dt), discretization_domainsize
    )  # calculate time for each point
    time = torch.tensor(time.astype(np.float32), requires_grad=False).to(device)

    # x, y (64, 96)
    # xc = np.linspace(-1, 1, N[0])
    # yc = np.linspace(0, 2 * np.pi, N[1])

    # Shrink the dataset in time and space
    seconds = config.train_data_seconds  # trim data to x seconds
    steps = int(seconds / dt)
    number_points = discretization_domainsize * steps

    x = x[:number_points]
    y = y[:number_points]
    time = time[:number_points]
    ux = ux[:number_points]
    uy = uy[:number_points]
    temperature = temperature[:number_points]

    # PDE points
    step_pde = config.collocation_pde_every

    x_pde = x[:discretization_domainsize]
    y_pde = y[:discretization_domainsize]

    x_pde = x_pde.reshape(discretization_domain[0], discretization_domain[1])
    y_pde = y_pde.reshape(discretization_domain[0], discretization_domain[1])

    x_pde = x_pde[::step_pde, ::step_pde]
    y_pde = y_pde[::step_pde, ::step_pde]

    x_pde = x_pde.flatten().unsqueeze(1).requires_grad_(False).to(device)
    y_pde = y_pde.flatten().unsqueeze(1).requires_grad_(False).to(device)

    # Boundary points
    step_boundary = config.collocation_boundary_every

    x_boundary = x[:discretization_domainsize]
    y_boundary = y[:discretization_domainsize]

    x_boundary = x_boundary.reshape(discretization_domain[0], discretization_domain[1])
    y_boundary = y_boundary.reshape(discretization_domain[0], discretization_domain[1])

    x_boundary = x_boundary[::step_boundary, ::step_boundary]
    y_boundary = y_boundary[::step_boundary, ::step_boundary]

    y_column = x_boundary[:, 0]  # Get row of x colloction points
    x_row = y_boundary[0, :]  # Get column of y colloction points

    num_points_x = len(x_row)
    num_points_y = len(y_column)

    # Create tensors for the x and y coordinates of the boundary points
    x_bottom = x_row
    y_bottom = torch.full(
        (num_points_x,), domain[0][0], device=device
    )  # bottom boundary y value

    x_top = x_row
    y_top = torch.full(
        (num_points_x,), domain[0][1], device=device
    )  # top boundary y value

    x_left = torch.full(
        (num_points_y,), domain[1][0], device=device
    )  # left boundary x value
    y_left = y_column

    x_right = torch.full(
        (num_points_y,), domain[1][1], device=device
    )  # right boundary x value
    y_right = y_column

    # Concatenate the boundary points, x and y are flipped in shenfun data
    y_boundary = [
        x_bottom.requires_grad_(False).to(device),
        x_top.requires_grad_(False).to(device),
        x_left.requires_grad_(False).to(device),
        x_right.requires_grad_(False).to(device),
    ]
    x_boundary = [
        y_bottom.requires_grad_(False).to(device),
        y_top.requires_grad_(False).to(device),
        y_left.requires_grad_(False).to(device),
        y_right.requires_grad_(False).to(device),
    ]

    # bounday_split = [
    #     len(x_bottom),
    #     len(x_top),
    #     len(x_left),
    #     len(x_right),
    # ]

    # Data points
    # reshape the arrays to 3D (time, x, y)
    x = x.reshape(-1, discretization_domain[0], discretization_domain[1])
    y = y.reshape(-1, discretization_domain[0], discretization_domain[1])
    time = time.reshape(-1, discretization_domain[0], discretization_domain[1])
    ux = ux.reshape(-1, discretization_domain[0], discretization_domain[1])
    uy = uy.reshape(-1, discretization_domain[0], discretization_domain[1])
    temperature = temperature.reshape(
        -1, discretization_domain[0], discretization_domain[1]
    )

    step = config.train_data_every

    x = x[:, ::step, ::step]
    y = y[:, ::step, ::step]
    time = time[:, ::step, ::step]
    ux = ux[:, ::step, ::step]
    uy = uy[:, ::step, ::step]
    temperature = temperature[:, ::step, ::step]

    x = x.flatten()
    y = y.flatten()
    time = time.flatten()
    ux = ux.flatten()
    uy = uy.flatten()
    temperature = temperature.flatten()

    discretization_domain = [
        int(discretization_domain[0] / step),
        int(discretization_domain[1] / step),
    ]
    discretization_domainsize = discretization_domain[0] * discretization_domain[1]

    # Boundary
    # x_boundary = x[:discretization_domainsize]
    # y_boundary = y[:discretization_domainsize]

    # x_min, x_max = (x.min().item(), x.max().item())
    # y_min, y_max = (y.min().item(), y.max().item())

    # # Create masks for points on each boundary
    # left_boundary_mask = x_boundary == x_min
    # right_boundary_mask = x_boundary == x_max
    # bottom_boundary_mask = y_boundary == y_min
    # top_boundary_mask = y_boundary == y_max

    # print out the temperature data every 8th point in the x, y coordinates like a grid
    # for i in range(0, len(x), 8):
    #     print(f"Temperature at x={x[i]}, y={y[i]}: {temperature[i]}")

    print(
        f"Loaded RBC data with Ra={Ra} and Pr={Pr} with discretization domain size {discretization_domain} over {steps} time steps ({dt}). Total points: {len(x)}. PDE points: {len(x_pde)}. Boundary points: {len(x_bottom) + len(x_top) + len(x_left) + len(x_right)}."
    )

    # Visualize training data
    visualize_data(
        x.cpu().detach().numpy(),
        y.cpu().detach().numpy(),
        temperature.cpu().detach().numpy(),
        ux.cpu().detach().numpy(),
        uy.cpu().detach().numpy(),
        discretization_domain,
        dt,
        steps,
        config=config,
        log_name="visualization/shenfun_data",
    )

    config.pde_parameter_pr = Pr
    config.pde_parameter_ra = Ra
    config.pde_parameter_dt = dt
    config.pde_parameter_domain = domain
    config.pde_parameter_discretization_domain = discretization_domain

    # this order defines the order of inputs and order of outputs is used in the loss functions
    # dataset = TensorDataset(
    #     x.unsqueeze(1),
    #     y.unsqueeze(1),
    #     time.unsqueeze(1),
    #     ux.unsqueeze(1),
    #     uy.unsqueeze(1),
    #     temperature.unsqueeze(1),
    # )

    # val_size = int(len(dataset) * config.validation_split)  # x% for validation
    # train_size = len(dataset) - val_size

    # # Split the dataset into training and validation sets
    # train_dataset, val_dataset = random_split(
    #     dataset, [train_size, val_size]
    # )  # TODO: Only use full frames for training and validation

    # Identify the timeframes for validation
    val_indices = (
        torch.arange(time.shape[0]) % (10 * discretization_domainsize)
        < discretization_domainsize
    )

    # Create training and validation datasets
    train_mask = ~val_indices
    val_mask = val_indices

    x_masked = (x[train_mask]).unsqueeze(1)
    y_masked = (y[train_mask]).unsqueeze(1)
    time_masked = (time[train_mask]).unsqueeze(1)
    ux_masked = (ux[train_mask]).unsqueeze(1)
    uy_masked = (uy[train_mask]).unsqueeze(1)
    temperature_masked = (temperature[train_mask]).unsqueeze(1)

    noise_variance = config.data_noise_variance
    ux_masked = ux_masked + noise_variance * torch.randn_like(ux_masked)
    uy_masked = uy_masked + noise_variance * torch.randn_like(uy_masked)
    temperature_masked = temperature_masked + noise_variance * torch.randn_like(
        temperature_masked
    )

    visualize_data(
        x_masked.cpu().detach().numpy(),
        y_masked.cpu().detach().numpy(),
        temperature_masked.cpu().detach().numpy(),
        ux_masked.cpu().detach().numpy(),
        uy_masked.cpu().detach().numpy(),
        discretization_domain,
        dt,
        steps,
        time_stamps=time_masked.flatten()[::discretization_domainsize]
        .cpu()
        .detach()
        .numpy(),
        config=config,
        log_name="visualization/training_data",
    )

    train_dataset = TensorDataset(
        x_masked.requires_grad_(False).to(device),
        y_masked.requires_grad_(False).to(device),
        time_masked.requires_grad_(False).to(device),
        ux_masked.requires_grad_(False).to(device),
        uy_masked.requires_grad_(False).to(device),
        temperature_masked.requires_grad_(False).to(device),
    )

    val_dataset = TensorDataset(
        (x[val_mask]).unsqueeze(1).detach().to(device),
        (y[val_mask]).unsqueeze(1).detach().to(device),
        (time[val_mask]).unsqueeze(1).detach().to(device),
        (ux[val_mask]).unsqueeze(1).detach().to(device),
        (uy[val_mask]).unsqueeze(1).detach().to(device),
        (temperature[val_mask]).unsqueeze(1).detach().to(device),
    )

    print(
        f"Training data size: {len(train_dataset)}, Validation data size: {len(val_dataset)}, Batch size: {config.batch_size * discretization_domainsize}"
    )

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size * discretization_domainsize,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size * discretization_domainsize,
        shuffle=False,
    )

    return (
        train_dataloader,
        val_dataloader,
        x,  # for visualization all true data is used
        y,
        time,
        ux,
        uy,
        temperature,
        x_pde,
        y_pde,
        x_boundary,
        y_boundary,
    )


def train(config=hyperparameters, device=torch.device(hyperparameters.torch_device)):
    with wandb.init(project=config.wandb_project, config=config, save_code=True):
        # data loading
        (
            train_dataloader,
            val_dataloader,
            x_dataset,
            y_dataset,
            time_dataset,
            ux_dataset,
            uy_dataset,
            temperature_dataset,
            x_pde,
            y_pde,
            x_boundary,
            y_boundary,
        ) = load_data(config=config, device=device)

        # model
        model = PINN(
            3,
            config.number_hidden_layers,
            config.number_hidden_dimension,
            4,
            activation=config.activation,
        ).to(device)

        model.initialize_weights(seed=config.init_weights_seed)

        # model = Net(seq_net=[3, 90, 90, 90, 90, 90, 4]).to(device)

        # # Proceed with training
        if config.load_model:
            loaded_config, model_path = load_model(
                config.load_model,
            )
            loaded_model = torch.load(model_path, map_location=device)
            model.load_state_dict(loaded_model["model_state_dict"])

        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {parameter_count}")

        # callbacks
        callbacks = [
            TrainLossMonitor(),
            CheckpointMonitor(log_period=config.log_period_checkpoint),
            WandBMonitor(),
            ValidationMonitor(val_dataloader, log_period=config.log_period_validation),
            VisualizationMonitor(
                visualize_data,
                plot_error_over_time,
                x_dataset,
                y_dataset,
                time_dataset,
                ux_dataset,
                uy_dataset,
                temperature_dataset,
                config,
                log_period=config.log_period_visualization,
            ),
        ]

        begin = time.time()

        # loss functions
        loss_fn_pde = partial(
            compute_pde_loss,
            config=config,
            device=device,
            model=model,
            fixed_x=x_pde,
            fixed_y=y_pde,
            boundary_x=x_boundary,
            boundary_y=y_boundary,
        )
        loss_fn_data = partial(compute_data_loss, config=config, device=device)

        loss_fn_weights = partial(
            compute_weight_loss, config=config, device=device, model=model
        )

        loss_fn = [
            loss_fn_pde,
            loss_fn_data,
        ]  # always pde first, data second

        if config.use_third_objective:
            loss_fn.append(loss_fn_weights)  # weights third

        method_params = {}

        if config.moo_method == Moo_method.mgda:
            method_params["normalization"] = config.moo_normalization
            method_params["stop_crit"] = config.moo_mgda_stop_crit
            method_params["max_iter"] = config.moo_mgda_max_iter
        elif (
            config.moo_method == Moo_method.ls
            or config.moo_method == Moo_method.scaleinvls
        ):
            task_weights = torch.tensor(
                [
                    config.moo_ls_alpha_weight,
                    1 - config.moo_ls_alpha_weight,
                ]  # always pde first, data second
            ).to(device)
            method_params["task_weights"] = task_weights

        # training methods
        method = WeightMethods(
            method=config.moo_method,
            n_tasks=2,
            **method_params,
            device=device,
        )

        # if config.moo_method == Moo_method.ls:
        #     method.method.task_weights = torch.tensor(
        #         [
        #             config.moo_ls_alpha_weight,
        #             1 - config.moo_ls_alpha_weight,
        #         ]  # always pde first, data second
        #     ).to(device)

        # training
        trained_model = train_model_dataloader(
            model=model,
            loss_fn=loss_fn,
            data_loader=train_dataloader,
            moo=method,
            learning_rate=config.learning_rate,
            max_epochs=config.epochs,
            optimizer_fn=config.base_optimizer,  # wandb config converts to string, therefore use the dict
            epoch_callbacks=callbacks,
            log_period=config.log_period_validation,
        )

        torch.save(
            trained_model.state_dict(), "RB_train_lambda({}).pth".format(config.epochs)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--number_hidden_layers",
        type=int,
        default=hyperparameters.number_hidden_layers,
        help="number of hidden layers",
    )
    parser.add_argument(
        "--number_hidden_dimension",
        type=int,
        default=hyperparameters.number_hidden_dimension,
        help="number of hidden dimension",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=hyperparameters.learning_rate,
        help="learning rate",
    )
    parser.add_argument(
        "--epochs", type=int, default=hyperparameters.epochs, help="number of epochs"
    )
    parser.add_argument(
        "--moo_method",
        type=str,
        default=hyperparameters.moo_method,
        help="multi-objective optimization method",
    )
    parser.add_argument(
        "--moo_normalization",
        type=str,
        default=hyperparameters.moo_normalization,
        help="multi-objective optimization normalization",
    )
    parser.add_argument(
        "--base_optimizer",
        type=str,
        default=hyperparameters.base_optimizer,
        help="base optimizer",
    )
    parser.add_argument(
        "--data_noise_variance",
        type=float,
        default=hyperparameters.data_noise_variance,
        help="data noise variance",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=hyperparameters.torch_seed,
        help="torch seed",
    )
    parser.add_argument(
        "--torch_device",
        type=str,
        default=hyperparameters.torch_device,
        help="torch device",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="rayleigh_benard_convection_new",
        help="wandb project name",
    )
    parser.add_argument(
        "--wiggle_weights_variance",
        type=float,
        default=hyperparameters.wiggle_weights_variance,
        help="wiggle weights variance",
    )
    parser.add_argument(
        "--wiggle_weights_seed",
        type=int,
        default=hyperparameters.wiggle_weights_seed,
        help="wiggle weights seed",
    )
    parser.add_argument(
        "--init_weights_seed",
        type=int,
        default=hyperparameters.init_weights_seed,
        help="init weights seed",
    )
    parser.add_argument(
        "--numpy_seed",
        type=int,
        default=hyperparameters.numpy_seed,
        help="numpy seed",
    )
    parser.add_argument(
        "--pde",
        type=str,
        default=hyperparameters.pde,
        help="pde",
    )
    parser.add_argument(
        "--moo_ls_alpha_weight",
        type=float,
        default=hyperparameters.moo_ls_alpha_weight,
        help="moo ls alpha weight",
    )
    parser.add_argument(
        "--moo_mgda_max_iter",
        type=int,
        default=hyperparameters.moo_mgda_max_iter,
        help="moo mgda max iter",
    )
    parser.add_argument(
        "--moo_mgda_stop_crit",
        type=float,
        default=hyperparameters.moo_mgda_stop_crit,
        help="moo mgda stop crit",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=hyperparameters.batch_size,
        help="batch size",
    )
    parser.add_argument(
        "--pde_parameter_ra",
        type=float,
        default=hyperparameters.pde_parameter_ra,
        help="pde parameter ra",
    )
    parser.add_argument(
        "--pde_parameter_pr",
        type=float,
        default=hyperparameters.pde_parameter_pr,
        help="pde parameter pr",
    )
    parser.add_argument(
        "--pde_parameter_dt",
        type=float,
        default=hyperparameters.pde_parameter_dt,
        help="pde parameter dt",
    )
    parser.add_argument(
        "--pde_parameter_domain",
        type=tuple,
        default=hyperparameters.pde_parameter_domain,
        help="pde parameter domain",
    )
    parser.add_argument(
        "--pde_parameter_discretization_domain",
        type=list,
        default=hyperparameters.pde_parameter_discretization_domain,
        help="pde parameter discretization domain",
    )
    parser.add_argument(
        "--pde_parameter_bc_temperature_top",
        type=float,
        default=hyperparameters.pde_parameter_bc_temperature_top,
        help="pde parameter bc temperature top",
    )
    parser.add_argument(
        "--pde_parameter_bc_temperature_bottom",
        type=float,
        default=hyperparameters.pde_parameter_bc_temperature_bottom,
        help="pde parameter bc temperature bottom",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=hyperparameters.data_path,
        help="data path",
    )
    parser.add_argument(
        "--train_data_seconds",
        type=int,
        default=hyperparameters.train_data_seconds,
        help="train data seconds",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=hyperparameters.validation_split,
        help="validation split",
    )
    parser.add_argument(
        "--log_period_validation",
        type=int,
        default=hyperparameters.log_period_validation,
        help="log period validation",
    )
    parser.add_argument(
        "--log_period_checkpoint",
        type=int,
        default=hyperparameters.log_period_checkpoint,
        help="log period checkpoint",
    )
    parser.add_argument(
        "--log_period_visualization",
        type=int,
        default=hyperparameters.log_period_visualization,
        help="log period visualization",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default=hyperparameters.activation,
        help="activation",
    )
    parser.add_argument(
        "--train_data_every",
        type=int,
        default=hyperparameters.train_data_every,
        help="train data every",
    )
    parser.add_argument(
        "--collocation_pde_every",
        type=int,
        default=hyperparameters.collocation_pde_every,
        help="collocation pde every",
    )
    parser.add_argument(
        "--collocation_boundary_every",
        type=int,
        default=hyperparameters.collocation_boundary_every,
        help="collocation boundary every",
    )
    parser.add_argument(
        "--disable_visualization",
        type=bool,
        default=hyperparameters.disable_visualization,
        help="disable visualization",
    )
    parser.add_argument(
        "--use_third_objective",
        type=bool,
        default=hyperparameters.use_third_objective,
        help="use third objective",
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default=hyperparameters.load_model,
        help="load model",
    )

    args = parser.parse_args()

    if hasattr(torch.optim, args.base_optimizer):
        OptimClass = getattr(torch.optim, args.base_optimizer)
        args.base_optimizer = OptimClass
        print(f"Using optimizer: {args.base_optimizer}")

    if hasattr(Moo_method, args.moo_method):
        args.moo_method = getattr(Moo_method, args.moo_method)
        print(f"Using moo method: {args.moo_method}")

    return args


if __name__ == "__main__":
    args = parse_args()
    device = setup_device(config=args)
    train(config=args, device=device)


# %%
