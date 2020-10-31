#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.cuda.is_available())
print(torch.cuda.get_device_name())

#Hyperparams
beta_dim = 100
input_dim = 1
num_phi_rbf = 100
phi_rbf_sigma = 5
phi_hidden_layer_size = 10
z_dim = 16
num_training_funcs = 128 # Gives the numbers of betas to learn
num_eval_points = 20 # Number of points each function is evaluated at

encoder_h_dim_1 = 512
encoder_h_dim_2 = 512
encoder_h_dim_3 = 128

decoder_h_dim_1 = 128
decoder_h_dim_2 = 128
decoder_h_dim_3 = 128

function_xlims = [-5, 5]

def generate_cubic_dataset():
    x_points = np.random.uniform(low=-4, high=-2, size=(10,))
    x_points = np.append(x_points, np.random.uniform(low=2, high=4, size=(10,)))
    y_points = x_points**3 + np.random.normal(size=(20,)) * 3
    return (x_points, y_points)

# From krasserm github io
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
        
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def generate_gp_1d_dataset():
    # X = np.arange(function_xlims[0], function_xlims[1], 0.1).reshape(-1, 1)
    output = []
    for n in range(num_training_funcs):
        X = np.random.uniform(function_xlims[0], function_xlims[1],
            size=(num_eval_points,1))
        mu = np.zeros(X.shape)
        cov = kernel(X, X)
        sample = np.random.multivariate_normal(mu.ravel(), cov, 1)
        output.append((X, sample))
    return output

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.phi_rbf_centers = nn.Parameter(torch.ones(num_phi_rbf, input_dim))
        self.phi_rbf_centers = nn.Parameter(torch.tensor(
            np.random.uniform(function_xlims[0], function_xlims[1],
            size=(num_phi_rbf, input_dim))))
        self.phi_nn_1 = nn.Linear(num_phi_rbf, phi_hidden_layer_size)
        self.phi_nn_2 = nn.Linear(phi_hidden_layer_size, beta_dim)

        self.encoder_nn_1 = nn.Linear(beta_dim, encoder_h_dim_1)
        self.encoder_nn_2 = nn.Linear(encoder_h_dim_1, encoder_h_dim_2)
        self.encoder_nn_3 = nn.Linear(encoder_h_dim_2, encoder_h_dim_3)
        self.encoder_nn_4 = nn.Linear(encoder_h_dim_3, z_dim * 2)

        self.decoder_nn_1 = nn.Linear(z_dim, decoder_h_dim_1)
        self.decoder_nn_2 = nn.Linear(decoder_h_dim_1, decoder_h_dim_2)
        self.decoder_nn_3 = nn.Linear(decoder_h_dim_2, decoder_h_dim_3)
        self.decoder_nn_4 = nn.Linear(decoder_h_dim_3, beta_dim)

        # self.betas = nn.Parameter(torch.ones(num_training_funcs, beta_dim))
        self.betas = nn.Parameter(torch.tensor(
            np.random.uniform(-1, 1, size=(num_training_funcs, beta_dim))
        ))

        self.normal_sampler = torch.distributions.normal.Normal(0.0, 1.0)

    def Phi(self, input):
        # Takes input (batch x dim_in) and gives Phi(input) (batch x dim_out)
        input_expand = torch.unsqueeze(input, 1)
        phi_expand = torch.unsqueeze(self.phi_rbf_centers, 0)
        M1 = input_expand - phi_expand
        M2 = torch.sum(M1 ** 2, 2)
        M3 = torch.exp(-M2/phi_rbf_sigma)
        M4 = F.sigmoid(self.phi_nn_1(M3))
        M5 = self.phi_nn_2(M4)
        return M5

    def encoder(self, input):
        # input (batch x beta_dim) output ((batch x z_dim), (batch x z_dim))
        M1 = F.relu(self.encoder_nn_1(input))
        M2 = F.relu(self.encoder_nn_2(M1))
        M3 = F.relu(self.encoder_nn_3(M2))
        M4 = self.encoder_nn_4(M3)
        z_mean = M4[:, 0:z_dim]
        z_std = torch.exp(M4[:, z_dim:]) # needs to be positive
        return z_mean, z_std

    def decoder(self, input):
        # input (batch x z_dim) output (batch x beta_dim)
        M1 = F.relu(self.decoder_nn_1(input))
        M2 = F.relu(self.decoder_nn_2(M1))
        M3 = F.relu(self.decoder_nn_3(M2))
        M4 = self.decoder_nn_4(M3)
        return M4

    def get_loss(self, function_id, s, x, kl_factor, print_breakdown=False, 
        return_breakdown=False):
        # function_id is just to know which beta to use
        # s are the inputs (batch x dim)
        # x are the observed outputs (batch)
        batch_size = s.shape[0]

        phi_s = self.Phi(s)
        beta = self.betas[function_id, :]
        x_enc = torch.matmul(phi_s, beta)

        loss_term_1 = (x - x_enc)**2

        z_mean, z_std = self.encoder(beta.unsqueeze(0))
        z_samples = z_mean + z_std * self.normal_sampler.rsample((batch_size, z_dim)).cuda()
        # z_samples = z_mean.repeat(batch_size, 1)
        beta_hats = self.decoder(z_samples)
        x_dec = torch.sum(beta_hats * phi_s, dim=1)
        loss_term_2 = (x - x_dec)**2
        # beta_hat = self.decoder(z_mean)
        # beta_hat = beta_hat.reshape(beta.shape)
        # loss_term_2 = torch.mean((beta_hat - beta)**2)

        

        # You only get one value not batch_num values since there's only
        # one beta for the whole batch since they're all from the same function
        # But when you add all the losses together, it will get broadcasted
        # so that it is repeated for each item in the batch so the mean will
        # be ok
        loss_term_3 = 0.5 * torch.sum(z_std**2 + z_mean**2 - 1 - torch.log(z_std**2),
            dim=1)
        loss_term_3 = kl_factor * (loss_term_3/z_dim)

        if print_breakdown:
            # print("z_mean, std", z_mean, z_std)
            # print("z_samples", z_samples)
            print("1", torch.mean(loss_term_1))
            print("2", torch.mean(loss_term_2))
            print("3", loss_term_3)

        if return_breakdown == False:
            return torch.mean(loss_term_1 + loss_term_2) + loss_term_3
        else:
            return torch.mean(loss_term_1 + loss_term_2) + loss_term_3, \
                torch.mean(loss_term_1), torch.mean(loss_term_2), loss_term_3

    def eval_at_z(self, z, s, return_beta_hat=False):
        # Gives predicted x values at s points when the z value is given
        phi_s = self.Phi(s)
        beta_hat = self.decoder(z)
        x_dec = torch.matmul(phi_s, beta_hat)
        if not return_beta_hat:
            return x_dec
        else:
            return x_dec, beta_hat

def check_beta(model, id):
    test_points = torch.arange(-5, 5, 0.1).reshape(100, 1).cuda()
    phi_s = model.Phi(test_points)
    beta = model.betas[id, :]
    x_encs = torch.matmul(phi_s, beta)
    z_mean, z_std = model.encoder(beta.unsqueeze(0))
    print(z_mean, z_std)
    beta_hat = model.decoder(z_mean)
    x_decs = torch.matmul(beta_hat, torch.transpose(phi_s, 0, 1))
    plt.plot(test_points.detach().cpu().numpy(), x_encs.detach().cpu().numpy())
    plt.plot(test_points.detach().cpu().numpy().reshape(100),
        x_decs.detach().cpu().numpy().reshape(100))
    plt.scatter(dataset[id][0].reshape(num_eval_points), dataset[id][1].reshape(num_eval_points))
    plt.show()
#%%
dataset = generate_gp_1d_dataset()
        
model = Model().double().cuda()
#%%
optimizer = torch.optim.Adam(model.parameters())
#%%
num_funcs_to_consider = 1
for epoch_id in range(300):
    print(epoch_id)
    l1s = []
    l2s = []
    l3s = []
    for function_id in range(num_funcs_to_consider):
        optimizer.zero_grad()
        input_points = torch.tensor(dataset[function_id][0]).cuda()
        x_vals = torch.tensor(dataset[function_id][1]).cuda()
        loss, l1, l2, l3 = model.get_loss(function_id, input_points, x_vals,
            1.0, return_breakdown=True)
        loss.backward()
        optimizer.step()
        l1s.append(l1.detach().cpu().numpy())
        l2s.append(l2.detach().cpu().numpy())
        l3s.append(l3.detach().cpu().numpy())

    if epoch_id % 2 == 0:
        num_funcs_to_consider = min(num_funcs_to_consider+1, num_training_funcs)

    print("l1", np.mean(np.array(l1s)),
        "l2", np.mean(np.array(l2s)),
        "l3", np.mean(np.array(l3s)))

#%%
func_id = 0
input_points = torch.tensor(dataset[func_id][0])
x_vals = torch.tensor(dataset[func_id][1])
phi_s = model.Phi(input_points)
beta = model.betas[func_id, :]
x_encs = torch.matmul(phi_s, beta)
loss_1 = (x_vals - x_encs)**2
mean_loss_1 = torch.mean(loss_1)
print(mean_loss_1)
z_mean, z_std = model.encoder(beta.unsqueeze(0))
beta_hat = model.decoder(z_mean).reshape(100)
x_decs = torch.matmul(phi_s, beta_hat)
loss_2 = (x_vals - x_decs)**2
mean_loss_2 = torch.mean(loss_2)
print(mean_loss_2)

        phi_s = self.Phi(s)
        beta = self.betas[function_id, :]
        x_enc = torch.matmul(phi_s, beta)

        loss_term_1 = (x - x_enc)**2



#%%
input_points = torch.tensor(dataset[0][0]).cuda()
x_vals = torch.tensor(dataset[0][1]).cuda()
loss = model.get_loss(0, input_points, x_vals, True)


#%%
# Plot predicted function at a range of points in z space
test_points = torch.arange(-5, 5, 0.1).unsqueeze(1).cuda()
for i in range(10):
    eval_points = model.eval_at_z(i * torch.ones((z_dim,), dtype=torch.float64).cuda(), test_points)
    plt.plot(test_points.cpu().detach().numpy(), eval_points.cpu().detach().numpy())
plt.show()

#%%
# Examine the variablity in beta hat as z changes
N = 10
beta_hats = np.zeros((N, beta_dim))
test_points = torch.arange(-5, 5, 0.1).unsqueeze(1).cuda()
for n in range(N):
    eval_points, beta_hat = model.eval_at_z(
        n*torch.ones((5,), dtype=torch.float64).cuda(), test_points, True)
    beta_hats[n, :] = beta_hat.cpu().detach().numpy()
    print(torch.mean(torch.abs(beta_hat)))
print(np.std(beta_hats, axis=0))
print(np.mean(np.std(beta_hats, axis=0)))

#%%
# Print some betas from the model
print(model.betas[1, :])
for i in range(10):
    print(torch.mean(torch.abs(model.betas[i, :])))

#%%
for i in range(10):
    check_beta(model, i)
#%%

# Finite number of points
X = np.arange(-5, 5, 0.2).reshape(-1, 1)
print("X shape", X.shape)

# Mean and covariance of the prior
mu = np.zeros(X.shape)
print("mu shape", mu.shape)
print("mu ravel shape", mu.ravel().shape)
cov = kernel(X, X)
print("cov shape", cov.shape)

# Draw three samples from the prior
samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

print("samples shape", samples.shape)

# %%
samples = generate_gp_1d_dataset()
for i in range(100):
    plt.plot(np.arange(-5, 5, 0.1), samples[i, :])
plt.show()

# %%

