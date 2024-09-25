from pinnsform.util import *
from pinnsform.model import PINN, FLS, FLW, FLWC

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

RHO = 5.0


#   pred_res = model(x_res, t_res)
#   pred_left = model(x_left, t_left)
#   pred_right = model(x_right, t_right)
#   pred_upper = model(x_upper, t_upper)
#   pred_lower = model(x_lower, t_lower)
#   u_x = torch.autograd.grad(pred_res, x_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
#   u_t = torch.autograd.grad(pred_res, t_res, grad_outputs=torch.ones_like(pred_res), retain_graph=True, create_graph=True)[0]
#   loss_res = torch.mean((u_t - 5 * pred_res * (1-pred_res)) ** 2)
#   loss_bc = torch.mean((pred_upper - pred_lower) ** 2)
#   loss_ic = torch.mean((pred_left[:,0] - torch.exp(- (x_left[:,0] - torch.pi)**2 / (2*(torch.pi/4)**2))) ** 2)
#   loss_track.append([loss_res.item(), loss_bc.item(), loss_ic.item()])
#   loss = loss_res + loss_bc + loss_ic


def loss_fn(model, mesh, b_left, b_right, initial, initial_values):
    # pde
    u = f(model, mesh)
    pde_residue = df(model, mesh, wrt=1) - RHO*u*(1.0-u)
    pde_loss = pde_residue.pow(2).mean()

    # boundary
    boundary_residue = f(model, b_left) - f(model, b_right)
    boundary_loss = boundary_residue.pow(2).mean()

    # initial
    initial_residue = f(model, initial) - initial_values
    initial_loss = initial_residue.pow(2).mean()

    return pde_loss, boundary_loss, initial_loss

def intial_value_function(x):
    return torch.exp(- (x - torch.pi)**2 / (2*(torch.pi/4.0)**2))

def h(x):
    return np.exp( - (x-np.pi)**2 / (2 * (np.pi/4)**2))

def u_ana(x,t):
    return h(x) * np.exp(RHO*t) / ( h(x) * np.exp(RHO*t) + 1 - h(x))

problem_domain = ([0, 2*np.pi], [0, 1])

#####   COLLOCATION POINTS   #####

initial_memory = torch.cuda.memory_allocated(device)

train_points = (101, 101)
mesh, boundaries = generate_mesh_object(train_points, domain=problem_domain, device=device, full_requires_grad=True, border_requires_grad=False)
print(mesh.full.shape)
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
    pbar
) -> nn.Module:

    optimizer = optimizer_fn(model.parameters(), line_search_fn='strong_wolfe')

    all_data = {}
    all_data["pde_train_loss"] = np.zeros(max_epochs)
    all_data["boundary_loss"] = np.zeros(max_epochs)
    all_data["initial_loss"] = np.zeros(max_epochs)
    all_data["time"] = np.zeros(max_epochs)
    all_data["closure_calls"] = np.zeros(max_epochs)
    all_data["gpu_memory"] = np.zeros(max_epochs)

    for epoch in range(0, max_epochs):
        epoch_start = time.time()

        def closure():
            optimizer.zero_grad()
            pde_loss, boundary_loss, initial_loss = loss_fn(model)
            
            loss = pde_loss + boundary_loss + initial_loss # 1.0/train_points[0]*pde_loss
            if not all_data["closure_calls"][epoch]:
                with torch.no_grad():
                    all_data["pde_train_loss"][epoch] = pde_loss.item()
                    all_data["boundary_loss"][epoch] = boundary_loss.item()
                    all_data["initial_loss"][epoch] = initial_loss.item()
                all_data["gpu_memory"][epoch] = torch.cuda.memory_allocated(device)

            all_data["closure_calls"][epoch] += 1
            #graph = make_dot(loss)
            #graph.save(os.path.join(result_dir, f"computation_graph_epoch_{epoch}.dot"))
            loss.backward()
            return loss

        optimizer.step(closure)

        #memory = torch.cuda.memory_allocated(device)
        #print(f"epoch_{epoch} GPU memory", torch.cuda.memory_allocated(device))

        all_data["time"][epoch] = (time.time() - epoch_start)
        pbar.update(1)

    #for key in all_data:
    #    all_data[key] = np.array(all_data[key])

    #print(all_data)

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
MODELS = [PINN]#, FLS, FLW, FLW2]
model_names = ["PINN", "FLS", "FLW", "FLW2"]
optimizer = LBFGS
MAX_EPOCHS = 100


TOTAL_EPOCHS = NUM_SEEDS * MAX_EPOCHS * len(MODELS)

if __name__ == '__main__':
    pbar = tqdm(total=TOTAL_EPOCHS, ncols=100)

    for j, model_class in enumerate(MODELS):
        model_name = model_names[j]

        for init_seed in INIT_SEEDS:
            pbar.set_description(f"Processing {model_name} seed {init_seed}/{NUM_SEEDS-1}")

            set_random_seed(init_seed)

            base_model = model_class(in_dim=2, hidden_dim=128, out_dim=1, num_layer=4).to(device)
            base_model.apply(init_weights)

            print(get_n_params(base_model))
            #for param in base_model.parameters():
            #    print(param)

            trained_model, train_data = train_model(base_model, loss_function, MAX_EPOCHS, optimizer, pbar)

            ###   STORE   ###

            seed_folder_name = os.path.join(result_dir, model_name, f"seed_{init_seed}")
            os.makedirs(seed_folder_name, exist_ok=True)

            # model weights
            torch.save(trained_model.state_dict(), os.path.join(seed_folder_name,"trained_model.pth"))

            # train data
            stacked_train_data = np.stack([train_data["pde_train_loss"], train_data["boundary_loss"], train_data["initial_loss"], train_data["time"], train_data["closure_calls"], train_data["gpu_memory"]], axis=1)
            pd.DataFrame(stacked_train_data, columns=["pde_train_loss", "boundary_loss", "initial_loss", "time", "closure_calls", "gpu_memory"]).to_csv(os.path.join(seed_folder_name, "train_data.csv"), index = False)

            # relative prediction error
            prediction = f(trained_model, test_mesh).detach().cpu().numpy() 
            rmae = rMAE(prediction, analytic_solution)
            rrmse = rRMSE(prediction, analytic_solution)
            pd.DataFrame(np.stack([[rmae], [rrmse]], axis=1), columns=["rMAE", "rRMSE"]).to_csv(os.path.join(seed_folder_name, "error.csv"), index = False)


    with open(os.path.join(result_dir, f"{script_name}_executed.py"), 'a') as file:
        file.write("\n\n"+"#"*100+f"\n#\tSCRIPT EXECUTION TIME (HH:MM:SS)\n#\t{datetime.timedelta(seconds = int(time.time()-script_execution_start))}")
