import time

from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

from spr_functions.base import ParameterizedRecommender
from hyperopt import fmin, hp, tpe

from scipy import sparse as sps



class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.
    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dims[:-1], self.dims[1:])]
        )
        self.drop = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != len(self.weights) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.
    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])]
        )
        self.p_layers = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])]
        )

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)

        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, : self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1] :]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE = F.binary_cross_entropy(recon_x, x)
    BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return BCE + anneal * KLD


def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())


update_count = 0


def train(
    model,
    idxlist,
    train_data,
    batch_size,
    N,
    optimizer,
    criterion,
    epoch,
    total_anneal_steps=1000000,
    anneal_cap=0.2,
    log_interval=10,
    device="cpu",
):
    # Turn on training mode
    model.train()
    train_loss = 0.0
    start_time = time.time()
    global update_count

    np.random.shuffle(idxlist)

    for batch_idx, start_idx in enumerate(range(0, N, batch_size)):
        end_idx = min(start_idx + batch_size, N)
        data = train_data[idxlist[start_idx:end_idx]]
        data = naive_sparse2tensor(data).to(device)

        if total_anneal_steps > 0:
            anneal = min(anneal_cap, 1.0 * update_count / total_anneal_steps)
        else:
            anneal = anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = criterion(recon_batch, data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

        if batch_idx % log_interval == 0 and batch_idx > 0:
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | "
                "loss {:4.2f}".format(
                    epoch,
                    batch_idx,
                    len(range(0, N, batch_size)),
                    elapsed * 1000 / log_interval,
                    train_loss / log_interval,
                )
            )

            # Log loss to tensorboard
            n_iter = (epoch - 1) * len(range(0, N, batch_size)) + batch_idx

            start_time = time.time()
            train_loss = 0.0

    return model


def evaluate(data_tr, data_te, model, N, batch_size=64, device="cpu"):
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    e_idxlist = list(range(data_tr.shape[0]))
    e_N = data_tr.shape[0]
    n20_list = []
    n100_list = []
    r20_list = []
    r50_list = []

    with torch.no_grad():
        for start_idx in range(0, e_N, batch_size):
            end_idx = min(start_idx + batch_size, N)
            data = data_tr[e_idxlist[start_idx:end_idx]]
            heldout_data = data_te[e_idxlist[start_idx:end_idx]]

            data_tensor = naive_sparse2tensor(data).to(device)

            recon_batch, mu, logvar = model(data_tensor)

            # Exclude examples from training set
            recon_batch = recon_batch.cpu().numpy()
            recon_batch[data.nonzero()] = -np.inf

            n20 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 20)
            n100 = NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
            r20 = Recall_at_k_batch(recon_batch, heldout_data, 20)
            r50 = Recall_at_k_batch(recon_batch, heldout_data, 50)

            n20_list.append(n20)
            n100_list.append(n100)
            r20_list.append(r20)
            r50_list.append(r50)

    total_loss /= len(range(0, e_N, batch_size))
    n20_list =  np.concatenate(n20_list)
    n100_list = np.concatenate(n100_list)
    r20_list = np.concatenate(r20_list)
    r50_list = np.concatenate(r50_list)

    return total_loss, np.mean(n20_list), np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)



class VAEInductiveRecommender(ParameterizedRecommender):

    def __init__(self, train_matrix, train_data, vad_data_te, vad_data_tr, loader):

        super().__init__(train_matrix, train_data,vad_data_te, vad_data_tr, loader)

        self.epochs = 30
        self.batch_size = 128

    @property
    def model_params(self):

        space = {
            "l2_norm": hp.uniform('l2_norm', 0, 10000),
        }
        return space


    def predict(self, data_tensor):
        recon_batch, mu, logvar = self.model(data_tensor)

        recon_batch = recon_batch.cpu().numpy()
        
        return recon_batch

    def fit_with_params(self, hyperparameters) -> float:

        n_items = self.loader.n_items
        train_data = self.train_data
        vad_data_tr, vad_data_te = self.vad_data_tr, self.vad_data_te
        
        N = train_data.shape[0]
        idxlist = list(range(N))

        p_dims = [200, 600, n_items]
        self.model = MultiVAE(p_dims).to('cpu')

        optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-3)
        criterion = loss_function

        best_n100 = -np.inf
        update_count = 0

        best_r50 = -np.inf
        pat = 0
        max_pat = 5

        try:
            for epoch in range(1, self.epochs + 1):
                epoch_start_time = time.time()
                self.model = train(
                    self.model,
                    idxlist,
                    train_data,
                    self.batch_size,
                    N,
                    optimizer,
                    criterion,
                    epoch,
                    total_anneal_steps=1000000,
                    anneal_cap=0.2,
                    log_interval=10,
                    device="cpu",
                )
                val_loss, n20, n100, r20, r50 = evaluate(vad_data_tr, vad_data_te, self.model, N)
                print("-" * 89)
                print(
                    "| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | "
                    "n20 {:5.4f} | n100 {:5.4f} | r20 {:5.4f} | r50 {:5.4f}".format(
                        epoch, time.time() - epoch_start_time, val_loss, n20, n100, r20, r50
                    )
                )
                print("-" * 89)

                n_iter = epoch * len(range(0, N, self.batch_size))

                if r50 > best_r50:
                    best_r50 = r50
                    pat = 0
                else:
                    pat += 1

                if pat > max_pat:
                    print("stop learning")
                    break

                # Save the model if the n100 is the best we've seen so far.

        except KeyboardInterrupt:
            print("-" * 89)
            print("Exiting from training early")

        valid_metric = self.eval()
        return valid_metric
