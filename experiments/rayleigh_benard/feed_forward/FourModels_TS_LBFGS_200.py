from pinnsform.util import *
from pinnsform.model import PINN, FLS, FLW, FullWavelet

from torchviz import make_dot

script_execution_start = time.time()

########################################################################################################
########################################################################################################

# making a new folder to save the script and the results 
script_name = os.path.basename(__file__)[:-3] # remove the ".py"
script_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join(script_dir, "results", script_name, f"{timestamp}")
os.makedirs(result_dir, exist_ok=True)

# storing the script
with open(__file__, 'r') as file:
    script_content = file.read()
with open(os.path.join(result_dir, f"{script_name}_executed.py"), 'w') as file:
    file.write("\n"+"#"*100+"\n#\tTHIS SCRIPT HAS BEEN EXECUTED ALREADY\n#\tTHIS IS A COPY OF THE ORIGINAL SCRIPT\n#\tTHIS SCRIPT IS NOT MEANT TO BE EXECUTED AGAIN\n#\tIT EXISTS ONLY FOR THE PURPOSE OF GIVING CONTEXT TO THE DATA IN THIS FOLDER\n"+"#"*100+"\n\n"+script_content)

########################################################################################################
########################################################################################################


#####   SETUP   #####

torch.set_default_dtype(torch.float32)

device = 'cuda'

# torch.set_default_device(device)

#####   PROBLEM   #####

BETA = 3.0


def loss_fn(model, mesh, b_left, b_right, initial, initial_values):
    # pde
    pde_residue = df(model, mesh, wrt=1, order=2) - BETA*df(model, mesh, wrt=0, order=2)
    pde_loss = pde_residue.pow(2).mean()

    # boundary
    bleft_residue = f(model, b_left)
    bright_residue = f(model, b_right)
    boundary_loss = bleft_residue.pow(2).mean() + bright_residue.pow(2).mean()

    # initial value
    initialV_residue = f(model, initial) - initial_values
    
    # initial derivative
    initialD_residue = df(model, initial, wrt=1)

    initialV_loss = initialV_residue.pow(2).mean()
    initialD_loss = initialD_residue.pow(2).mean()

    return pde_loss, boundary_loss, initialV_loss, initialD_loss

def intial_value_function(x):
    return torch.sin(torch.pi*x) + 1./2.*torch.sin(BETA*torch.pi*x)

def u_ana(x,t):
    return np.sin(np.pi*x)*np.cos(2*np.pi*t) + 1./2.*np.sin(BETA*np.pi*x)*np.cos(2*BETA*np.pi*t)

problem_domain = ([0, 1], [0, 2*np.pi], [0, 60])

#####   COLLOCATION POINTS   #####

initial_memory = torch.cuda.memory_allocated(device)

train_points = (16, 24, 61) 

# 51x51 mesh as list
np_mesh = generate_mesh(train_points, problem_domain)
# 51x51 mesh as list with temporal sequence for every point
sequence_mesh = make_temporal_sequence(np_mesh, num_step=5, step=0.2)
# flatten the temporal sequence for normal models
np_mesh_sequence = listify_sequence(sequence_mesh)
# make Mesh object of torch tensors
mesh = torchify(np_mesh_sequence, device, True)
boundaries = torchified_borders(np_mesh_sequence, problem_domain, device, True)

#mesh, boundaries = generate_mesh_object(train_points, domain=problem_domain, device=device, full_requires_grad=True, border_requires_grad=False)

b_left = boundaries[0][0]
b_right = boundaries[0][1]
initial = boundaries[1][0]

with torch.no_grad():
    initial_values = intial_value_function(initial.part[0])   #torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))

loss_function = partial(loss_fn, mesh=mesh, b_left=b_left, b_right=b_right, initial=initial, initial_values=initial_values)

allocated_memory_data = torch.cuda.memory_allocated(device) - initial_memory

# TEST
test_points = (201, 201)
test_mesh, _ = generate_mesh_object(test_points, domain=problem_domain, device=device, full_requires_grad=False, border_requires_grad=False)
analytic_solution = u_ana(test_mesh.part[0].cpu().numpy(), test_mesh.part[1].cpu().numpy())


#####   TRAINING LOOP   ######

def train_model(
    model:nn.Module,
    loss_fn,
    max_epochs,
    optimizer_fn,
    pbar,
    folder
) -> nn.Module:

    optimizer = optimizer_fn(model.parameters(), line_search_fn='strong_wolfe')

    all_data = {}
    all_data["pde_train_loss"] = np.zeros(max_epochs)
    all_data["boundary_loss"] = np.zeros(max_epochs)
    all_data["initialV_loss"] = np.zeros(max_epochs)
    all_data["initialD_loss"] = np.zeros(max_epochs)
    all_data["time"] = np.zeros(max_epochs)
    all_data["closure_calls"] = np.zeros(max_epochs)
    all_data["gpu_memory"] = np.zeros(max_epochs)

    for epoch in range(0, max_epochs):
        if epoch < 5 or all_data["initialV_loss"][epoch-1] != all_data["initialV_loss"][epoch-2]:
            torch.save(model.state_dict(), os.path.join(folder,f"model_at_{epoch}.pth"))
        epoch_start = time.time()

        def closure():
            optimizer.zero_grad()
            pde_loss, boundary_loss, initialV_loss, initialD_loss = loss_fn(model)
                
            loss = pde_loss + boundary_loss + initialV_loss + initialD_loss
            
            if not all_data["closure_calls"][epoch]:
                with torch.no_grad():
                    all_data["pde_train_loss"][epoch] = pde_loss.item()
                    all_data["boundary_loss"][epoch] = boundary_loss.item()
                    all_data["initialV_loss"][epoch] = initialV_loss.item()
                    all_data["initialD_loss"][epoch] = initialD_loss.item()
                all_data["gpu_memory"][epoch] = torch.cuda.memory_allocated(device)

            all_data["closure_calls"][epoch] += 1
            loss.backward()
            return loss

        optimizer.step(closure)
        
        all_data["time"][epoch] = (time.time() - epoch_start)

        pbar.update(1)

    return model, all_data


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


#def init_weights(m):
#    if isinstance(m, nn.Linear):
#        torch.nn.init.xavier_uniform_(m.weight)
#        m.bias.data.fill_(0.01)

NUM_SEEDS = 100
INIT_SEEDS = np.array(range(NUM_SEEDS))
MODELS = [PINN, FLS, FLW, FullWavelet]
model_names = ["PINN", "FLS", "FLW", "FullWavelet"]
#ModelParams = [(512, 4)]#, (193, 16)]
optimizer = LBFGS
MAX_EPOCHS = 200

TOTAL_EPOCHS = NUM_SEEDS * MAX_EPOCHS * len(MODELS)

if __name__ == '__main__':
    pbar = tqdm(total=TOTAL_EPOCHS, ncols=100)

    for j, model_class in enumerate(MODELS):
        model_name = model_names[j]

        for init_seed in INIT_SEEDS:
            pbar.set_description(f"Processing {model_name} seed {init_seed}/{NUM_SEEDS-1}")

            base_model = model_class(in_dim=3, hidden_dim=512, out_dim=4, num_layer=4).to(device)
            set_random_seed(init_seed)
            base_model.apply(init_weights)
            
            seed_folder_name = os.path.join(result_dir, model_name, f"seed_{init_seed}")
            os.makedirs(seed_folder_name, exist_ok=True)

            trained_model, train_data = train_model(base_model, loss_function, MAX_EPOCHS, optimizer, pbar, seed_folder_name)
            
            ###   STORE   ###
            torch.save(trained_model.state_dict(), os.path.join(seed_folder_name,f"model_at_{MAX_EPOCHS}.pth"))

            # train data
            stacked_train_data = np.stack([train_data["pde_train_loss"], train_data["boundary_loss"], train_data["initialV_loss"], train_data["initialD_loss"], train_data["time"], train_data["closure_calls"], train_data["gpu_memory"]], axis=1)
            pd.DataFrame(stacked_train_data, columns=["pde_train_loss", "boundary_loss", "initialV_loss", "initialD_loss", "time", "closure_calls", "gpu_memory"]).to_csv(os.path.join(seed_folder_name, "train_data.csv"), index = False)
#
            ## relative prediction error
            prediction = f(trained_model, test_mesh).detach().cpu().numpy() 
            rmae = rMAE(prediction, analytic_solution)
            rrmse = rRMSE(prediction, analytic_solution)
            pd.DataFrame(np.stack([[rmae], [rrmse]], axis=1), columns=["rMAE", "rRMSE"]).to_csv(os.path.join(seed_folder_name, "error.csv"), index = False)


    with open(os.path.join(result_dir, f"{script_name}_executed.py"), 'a') as file:
        file.write("\n\n"+"#"*100+f"\n#\tSCRIPT EXECUTION TIME (HH:MM:SS)\n#\t{datetime.timedelta(seconds = int(time.time()-script_execution_start))}")
