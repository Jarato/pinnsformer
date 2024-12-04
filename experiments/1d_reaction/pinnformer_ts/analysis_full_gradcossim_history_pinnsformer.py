from pinnsform import *
from pinnsform.model import PINNsformer, PINNsformerTanh, PINNsformerNoAttention
import matplotlib.pyplot as plt
from numpy import dot
from numpy.linalg import norm

base_dir = os.path.dirname(os.path.abspath(__file__))
experiment_name = "PINNFORMER_VAR_TS_LBFGS_200"
path = os.path.join(base_dir, 'results', experiment_name, 'run')
analysis_path = os.path.join(base_dir, 'analysis', experiment_name)

device = 'cuda'

problem_domain = ([0, 2*np.pi], [0, 1])
RHO = 5.0

def loss_residue_fn(model, mesh):
    u = f(model, mesh)
    pde_residue = df(model, mesh, wrt=1) - RHO*u*(1.0-u)
    pde_loss = pde_residue.pow(2).mean()
    return pde_loss

def loss_boundary_fn(model, b_left, b_right):
    boundary_residue = f(model, b_left) - f(model, b_right)
    boundary_loss = boundary_residue.pow(2).mean()
    return boundary_loss

def loss_initial_fn(model, initial, initial_values):
    initial_residue = f(model, initial)[:,0] - initial_values
    initial_loss = initial_residue.pow(2).mean()
    return initial_loss

def loss_final_fn(model, mesh, b_left, b_right, initial, initial_values):
    # pde
    u = f(model, mesh)
    pde_residue = df(model, mesh, wrt=1) - RHO*u*(1.0-u)
    pde_loss = pde_residue.pow(2).mean()

    # boundary
    boundary_residue = f(model, b_left) - f(model, b_right)
    boundary_loss = boundary_residue.pow(2).mean()

    # initial
    initial_residue = f(model, initial)[:,0] - initial_values
    initial_loss = initial_residue.pow(2).mean()

    final_loss = pde_loss + boundary_loss + initial_loss

    return final_loss


def intial_value_function(x):
    return torch.exp(- (x - torch.pi)**2 / (2*(torch.pi/4.0)**2))

def h(x):
    return np.exp( - (x-np.pi)**2 / (2 * (np.pi/4)**2))

def u_ana(x,t):
    return h(x) * np.exp(RHO*t) / ( h(x) * np.exp(RHO*t) + 1 - h(x))

train_points = (51, 51)
sequence_mesh, sequence_boundaries = generate_mesh_object(train_points, domain=problem_domain, device=device, full_requires_grad=True, border_requires_grad=False, num_seq_steps=5, seq_step_size=1e-3)

seq_b_left = sequence_boundaries[0][0]
seq_b_right = sequence_boundaries[0][1]
seq_initial = sequence_boundaries[1][0]

with torch.no_grad():
    initial_values = intial_value_function(seq_initial.part[0][:,0])

loss_final_seq = partial(loss_final_fn, mesh=sequence_mesh, b_left=seq_b_left, b_right=seq_b_right, initial=seq_initial, initial_values=initial_values)
loss_residue_seq = partial(loss_residue_fn, mesh=sequence_mesh)
loss_boundary_seq = partial(loss_boundary_fn, b_left=seq_b_left, b_right=seq_b_right)
loss_initial_seq = partial(loss_initial_fn, initial=seq_initial, initial_values=initial_values)


# 51 x 51 mesh
np_mesh = generate_mesh(train_points, problem_domain)
# create spatial sequences
np_mesh_sequence = listify_sequence(make_temporal_sequence(np_mesh, 5, 1e-3))
# flatten the sequence
np_sequence_mesh = np.expand_dims(np_mesh_sequence, 1)
# make torch Mesh object
single_mesh = torchify(np_sequence_mesh, device, True)
single_boundaries = torchified_borders(np_mesh_sequence, problem_domain, device, False)

sing_b_left = single_boundaries[0][0]
sing_b_right = single_boundaries[0][1]
sing_initial = single_boundaries[1][0]

loss_final_sing = partial(loss_final_fn, mesh=single_mesh, b_left=sing_b_left, b_right=sing_b_right, initial=sing_initial, initial_values=initial_values)
loss_residue_sing = partial(loss_residue_fn, mesh=single_mesh)
loss_boundary_sing = partial(loss_boundary_fn, b_left=sing_b_left, b_right=sing_b_right)
loss_initial_sing = partial(loss_initial_fn, initial=sing_initial, initial_values=initial_values)

sequence_loss_functions = [loss_final_seq, loss_residue_seq, loss_boundary_seq, loss_initial_seq]
single_loss_functions = [loss_final_sing, loss_residue_sing, loss_boundary_sing, loss_initial_sing] 


def calculate_loss_grads(model, l_final, l_residue, l_boundary, l_initial):
    l_final_grad = grad_vector_of(model, l_final(model))
    l_residue_grad = grad_vector_of(model, l_residue(model))
    l_boundary_grad = grad_vector_of(model, l_boundary(model))
    l_initial_grad = grad_vector_of(model, l_initial(model))

    l_final_norm = norm(l_final_grad)
    l_residue_norm = norm(l_residue_grad)
    l_boundary_norm = norm(l_boundary_grad)
    l_initial_norm = norm(l_initial_grad)

    final_residue_cossim = cosine_similarity(l_final_grad, l_residue_grad)
    final_boundary_cossim = cosine_similarity(l_final_grad, l_boundary_grad)
    final_initial_cossim = cosine_similarity(l_final_grad, l_initial_grad)
    residue_boundary_cossim = cosine_similarity(l_residue_grad, l_boundary_grad)
    residue_initial_cossim = cosine_similarity(l_residue_grad, l_initial_grad)
    boundary_initial_cossim = cosine_similarity(l_boundary_grad, l_initial_grad)
    
    return l_final_norm, l_residue_norm, l_boundary_norm, l_initial_norm, final_residue_cossim,final_boundary_cossim, final_initial_cossim, residue_boundary_cossim, residue_initial_cossim, boundary_initial_cossim

def history_loss_cosine_similarity(model, weight_history_path, lfinal, lresidue, lboundary, linitial, pbar):
    history = []
    for epoch in range(201):
        model_weight_path = os.path.join(weight_history_path, f"model_at_{epoch}.pth")
        if os.path.exists(model_weight_path):
            model.load_state_dict(torch.load(model_weight_path))
            history.append(calculate_loss_grads(model, lfinal, lresidue, lboundary, linitial))
        pbar.update(1)
    return history

temporal_locations = list(np_mesh_sequence[np_mesh_sequence[:,0] == 0][:,1])

def grad_vector_of(model, of):
    model.zero_grad()
    of.backward()
    grad_vec = torch.cat([param.grad.view(-1) for param in model.parameters()]).detach().cpu().numpy()
    return grad_vec

def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def temporal_cosine_similarity(model, mesh, pbar):
    mesh_temporal_cossim = []
    for spatial_slice in mesh.full.reshape((51, 255, 2)):
        grad_vectors = [grad_vector_of(model, model(point)) for point in spatial_slice]
        mesh_temporal_cossim.append([cosine_similarity(grad_vectors[0], b) for b in grad_vectors])
    return np.mean(np.array(mesh_temporal_cossim), axis=0)#, np.std(np.array(mesh_temporal_cossim, axis=0))

def history_temp_cosine_similarity(model, mesh, weight_history_path, pbar):
    history = []
    for epoch in range(201):
        model_weight_path = os.path.join(weight_history_path, f"model_at_{epoch}.pth")
        if os.path.exists(model_weight_path):
            model.load_state_dict(torch.load(model_weight_path))
            history.append(temporal_cosine_similarity(model, mesh, pbar))
        pbar.update(1)
    return history

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


models = [PINNsformer]#PINNsformer, PINNsformerNoAttention, PINNsformer]
model_names = ["NoSequence_PINNsformer"]#"PINNsformer", "PINNsformerNoAttention", "NoSequence_PINNsformer"]
loss_function_lists = [single_loss_functions]#sequence_loss_functions, sequence_loss_functions, single_loss_functions]
NUM_SEEDS = 100
INIT_SEEDS = np.array(range(NUM_SEEDS))

TOTAL_STEPS = len(models) * NUM_SEEDS * 201

if __name__ == '__main__':
    pbar = tqdm(total=TOTAL_STEPS, ncols=100)
    
    for k, model_class in enumerate(models):
        model_name = model_names[k]
        model_result_path = os.path.join(path, model_name)
        model_analysis_path = os.path.join(analysis_path, model_name, "loss_cosine_similarity")
        loss_functions = loss_function_lists[k]
        os.makedirs(model_analysis_path, exist_ok=True)

        rmae_list = []
        for seed in INIT_SEEDS:
            seed_folder_path = os.path.join(model_result_path, f"seed_{seed}")

            model = model_class(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)

            cosine_loss_sim_history = history_loss_cosine_similarity(model=model, weight_history_path=seed_folder_path, lfinal=loss_functions[0], lresidue=loss_functions[1], lboundary=loss_functions[2], linitial=loss_functions[3], pbar=pbar)
            pd.DataFrame(np.array(cosine_loss_sim_history), columns=["lfinal_grad_norm", "lresidual_grad_norm", "lboundary_grad_norm", "linitial_grad_norm", "final_residue_cossim", "final_boundary_cossim", "final_initial_cossim", "residue_boundary_cossim", "residue_initial_cossim", "boundary_initial_cossim"]).to_csv(os.path.join(model_analysis_path, f"loss_cosine_history_seed_{seed}.csv"), index = False)

            #cosine_temp_sim_history = history_temp_cosine_similarity(model=model, mesh=mesh, weight_history_path=seed_folder_path, pbar=pbar)
            #pd.DataFrame(np.array(cosine_temp_sim_history), columns=temporal_locations).to_csv(os.path.join(model_analysis_path, f"cosine_sim_history_seed_{seed}.csv"), index = False)

