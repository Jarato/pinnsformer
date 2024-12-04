from pinnsform import *
from pinnsform.model import PINNsformer, PINNsformerTanh, PINNsformerNoAttention
import matplotlib.pyplot as plt

device = 'cuda'

problem_domain = ([0, 2*np.pi], [0, 1])
train_points = (51, 51)
sequence_mesh, _ = generate_mesh_object(train_points, domain=problem_domain, device=device, full_requires_grad=False, border_requires_grad=False, num_seq_steps=5, seq_step_size=1e-3)

# 51 x 51 mesh
np_mesh = generate_mesh(train_points, problem_domain)
# create spatial sequences
np_mesh_sequence = listify_sequence(make_temporal_sequence(np_mesh, 5, 1e-3))
# flatten the sequence
np_sequence_mesh = np.expand_dims(np_mesh_sequence, 1)
# make torch Mesh object
single_mesh = torchify(np_sequence_mesh, device, False)

temporal_locations = list(np_mesh_sequence[np_mesh_sequence[:,0] == 0][:,1])

base_dir = os.path.dirname(os.path.abspath(__file__))
experiment_name = "PINNFORMER_VAR_TS_LBFGS_200"
path = os.path.join(base_dir, 'results', experiment_name, 'run')
analysis_path = os.path.join(base_dir, 'analysis', experiment_name)

def add_noise(model, noise_seed = 0, noise=0.001):
    set_random_seed(noise_seed)
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn(param.size(), device=device) * noise)

def estimate_temporal_correlation(model, model_weights, mesh, repeats = 1000, noise = 0.001):
    model.load_state_dict(model_weights)
    u_before = f(model, mesh).detach().cpu().numpy()

    u_neighborhood = []
    for i in range(repeats):
        model.load_state_dict(model_weights)
        add_noise(model, noise_seed = i, noise=noise)
        u_neighborhood.append(f(model, mesh).detach().cpu().numpy())

    diffs = []
    for u_after in u_neighborhood:
        diffs.append(u_after - u_before)

    diffs = np.reshape(np.array(diffs), (repeats, 51, 51*5))
    corr_matrices = [np.corrcoef(diffs[i].T) for i in range(repeats)]
    return np.mean(corr_matrices, axis=0)[0]

def correlation_train_history(weight_history_path, estimateTC_fn, pbar):
    history = []
    for epoch in range(201):
        model_weight_path = os.path.join(weight_history_path, f"model_at_{epoch}.pth")
        if os.path.exists(model_weight_path):
            history.append(estimateTC_fn(model_weights = torch.load(model_weight_path)))
        pbar.update(1)
    return history


models = [PINNsformer, PINNsformerNoAttention, PINNsformer]
model_names = ["PINNsformer", "PINNsformerNoAttention", "NoSequence_PINNsformer"]
meshes = [sequence_mesh, sequence_mesh, single_mesh]
NUM_SEEDS = 100
INIT_SEEDS = np.array(range(NUM_SEEDS))

TOTAL_STEPS = len(models) * NUM_SEEDS * 201

if __name__ == '__main__':
    pbar = tqdm(total=TOTAL_STEPS, ncols=100)
    
    for k, model_class in enumerate(models):
        model_name = model_names[k]
        model_result_path = os.path.join(path, model_name)
        model_analysis_path = os.path.join(analysis_path, model_name, "temporal_correlation")
        os.makedirs(model_analysis_path, exist_ok=True)

        rmae_list = []
        for seed in INIT_SEEDS:
            seed_folder_path = os.path.join(model_result_path, f"seed_{seed}")

            model = model_class(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)
            estimateTC_function = partial(estimate_temporal_correlation, model=model, mesh=meshes[k], repeats = 1000, noise = 0.0001)
            correlation_history = correlation_train_history(seed_folder_path, estimateTC_function, pbar)

            pd.DataFrame(np.array(correlation_history), columns=temporal_locations).to_csv(os.path.join(model_analysis_path, f"correlation_history_seed_{seed}.csv"), index = False)