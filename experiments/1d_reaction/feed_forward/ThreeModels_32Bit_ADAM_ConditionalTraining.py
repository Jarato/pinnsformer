from pinnsform.util import *
from pinnsform.model import PINN, FLS, FLW
from torch.optim import Adam

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
#torch.set_num_threads(1)
#torch.set_default_device(device)

#####   PROBLEM   #####

RHO = 5.0


def intial_value_function(x):
    return torch.exp(- (x - torch.pi)**2 / (2*(torch.pi/4.0)**2))

def h(x):
    return np.exp( - (x-np.pi)**2 / (2 * (np.pi/4)**2))

def u_ana(x,t):
    return h(x) * np.exp(RHO*t) / ( h(x) * np.exp(RHO*t) + 1 - h(x))


                #    t          x
problem_domain = ([0, 2*np.pi], [0, 1])

#####   COLLOCATION POINTS   #####

initial_memory = torch.cuda.memory_allocated(device) 

train_points = (101, 101)
mesh, boundaries = generate_mesh_object(train_points, domain=problem_domain, device=device, full_requires_grad=True, border_requires_grad=False)

#print()
#print('mesh')
#print(mesh.full)

b_left = boundaries[0][0]
b_right = boundaries[0][1]
initial = boundaries[1][0]

with torch.no_grad():
    initial_values = intial_value_function(initial.part[0])   #torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))

def shift_ones(vector, first_entry):
    shifted = torch.roll(torch.cumprod(vector, 0)*first_entry, 1)
    shifted[0] = first_entry
    return shifted

def loss_fn(model, mesh, b_left, b_right, initial, initial_values, threshold = 0.01):

    # initial
    initial_residue = f(model, initial) - initial_values
    initial_loss = initial_residue.pow(2)

    initial_mean = torch.mean(initial_loss.detach())
    #print(initial_mean)

    # boundary
    boundary_residue = f(model, b_left) - f(model, b_right)
    boundary_loss = boundary_residue.pow(2)

    boundary_mean = torch.sum(boundary_loss.detach(), axis=1)
    #print(boundary_mean)

    # pde
    #print(mesh.full)
    u = f(model, mesh)

    pde_residue = df(model, mesh, wrt=1) - RHO*u*(1.0-u)
    #print(pde_residue)
    pde_loss = torch.reshape(pde_residue.pow(2), train_points)
    #print()
    pde_array = pde_loss.detach()

    # time slices
    time_slice_loss = (torch.sum(pde_array, axis=0) + boundary_mean)/train_points[0]
    print(initial_mean)
    print(time_slice_loss)
    active_time_slices = shift_ones(time_slice_loss < threshold, initial_mean < threshold)

    #print(active_time_slices)
    #print(pde_loss)
    # masking
    boundary_loss = boundary_loss.reshape(train_points[-1]) * active_time_slices
    pde_loss = pde_loss * active_time_slices

    active_slices = torch.sum(active_time_slices)
    if active_slices == 0:
        active_collocation_points = 1
        active_border_points = 1
    else:
        active_collocation_points = active_slices*train_points[0]
        active_border_points = active_slices*2
    #print(active_slices, active_collocation_points, active_border_points)

    return torch.sum(pde_loss)/active_collocation_points, torch.sum(boundary_loss)/active_border_points, initial_loss.mean(), active_slices

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
    pbar
) -> nn.Module:

    optimizer = optimizer_fn(model.parameters())
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimizer=optimizer, factor=1 - 0.1, patience=5, threshold=1e-8, cooldown=5
    #)

    all_data = {}
    all_data["pde_train_loss"] = np.zeros(max_epochs)
    all_data["boundary_loss"] = np.zeros(max_epochs)
    all_data["initial_loss"] = np.zeros(max_epochs)
    all_data["time_slices"] = np.zeros(max_epochs)
    all_data["time"] = np.zeros(max_epochs)
    all_data["gpu_memory"] = np.zeros(max_epochs)
    #all_data["learning_rate"] = np.zeros(max_epochs)

    for epoch in range(0, max_epochs):
        epoch_start = time.time()

        def closure():
            optimizer.zero_grad()
            pde_loss, boundary_loss, initial_loss, time_slices = loss_fn(model)
            
            loss = pde_loss + boundary_loss + initial_loss

            all_data["pde_train_loss"][epoch] = pde_loss.item()
            all_data["boundary_loss"][epoch] = boundary_loss.item()
            all_data["initial_loss"][epoch] = initial_loss.item()
            all_data["gpu_memory"][epoch] = torch.cuda.memory_allocated(device)
            all_data["time_slices"][epoch] = time_slices.item()
            #all_data["learning_rate"][epoch] = np.array([pg['lr'] for pg in scheduler.optimizer.param_groups])[0]
            #graph = make_dot(loss)
            #graph.save(os.path.join(result_dir, f"computation_graph_epoch_{epoch}.dot"))
            loss.backward()
            return loss

        loss = closure()

        optimizer.step()
        #scheduler.step(loss)
        pbar.update(1)

        all_data["time"][epoch] = (time.time() - epoch_start)
    
    return model, all_data


#def init_weights(m):
#    if isinstance(m, nn.Linear):
#        torch.nn.init.xavier_uniform_(m.weight)
#        torch.nn.init.zeros_(m.bias)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

NUM_SEEDS = 1
INIT_SEEDS = np.array(range(NUM_SEEDS))
MODELS = [PINN]#, FLS, FLW]
model_names = ["PINN", "FLS", "FLW"]
optimizer = Adam
MAX_EPOCHS = 1


TOTAL_EPOCHS = NUM_SEEDS * len(MODELS) * MAX_EPOCHS

if __name__ == '__main__':
    pbar = tqdm(total=TOTAL_EPOCHS, ncols=100)

    for j, model_class in enumerate(MODELS):
        model_name = model_names[j]

        for init_seed in INIT_SEEDS:
            #pbar.set_description(f"Processing {model_name} seed {init_seed}/{NUM_SEEDS-1}")

            set_random_seed(init_seed)

            base_model = model_class(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
            base_model.apply(init_weights)

            #for param in base_model.parameters():
            #    print(param)

            trained_model, train_data = train_model(base_model, loss_function, MAX_EPOCHS, optimizer, pbar)

            ###   STORE   ###

            seed_folder_name = os.path.join(result_dir, model_name, f"seed_{init_seed}")
            os.makedirs(seed_folder_name, exist_ok=True)

            # model weights
            torch.save(trained_model.state_dict(), os.path.join(seed_folder_name,"trained_model.pth"))

            # train data
            stacked_train_data = np.stack([train_data["pde_train_loss"], train_data["boundary_loss"], train_data["initial_loss"], train_data["time_slices"], train_data["time"], train_data["gpu_memory"]], axis=1)
            pd.DataFrame(stacked_train_data, columns=["pde_train_loss", "boundary_loss", "initial_loss", "time_slices", "time", "gpu_memory"]).to_csv(os.path.join(seed_folder_name, "train_data.csv"), index = False)

            # relative prediction error
            prediction = f(trained_model, test_mesh).detach().cpu().numpy() 
            rmae = rMAE(prediction, analytic_solution)
            rrmse = rRMSE(prediction, analytic_solution)
            pd.DataFrame(np.stack([[rmae], [rrmse]], axis=1), columns=["rMAE", "rRMSE"]).to_csv(os.path.join(seed_folder_name, "error.csv"), index = False)


    with open(os.path.join(result_dir, f"{script_name}_executed.py"), 'a') as file:
        file.write("\n\n"+"#"*100+f"\n#\tSCRIPT EXECUTION TIME (HH:MM:SS)\n#\t{datetime.timedelta(seconds = time.time()-script_execution_start)}")
