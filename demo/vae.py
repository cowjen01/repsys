import torch
from torch import nn
from torch.nn import functional


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def loss_function(recon_x, x, mu, log_var):
    BCE = functional.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD


def reparametrize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std).to(get_device())

    return (eps * std) + mu


class VAEModule(nn.Module):
    def __init__(self, in_dim, h1_dim, h2_dim, h3_dim, latent_dim):
        super(VAEModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, h1_dim)
        self.bn1 = nn.BatchNorm1d(num_features=h1_dim)

        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.bn2 = nn.BatchNorm1d(num_features=h2_dim)

        self.fc3 = nn.Linear(h2_dim, h3_dim)
        self.bn3 = nn.BatchNorm1d(num_features=h3_dim)

        self.fc31 = nn.Linear(h3_dim, latent_dim)
        self.fc32 = nn.Linear(h3_dim, latent_dim)

        self.fc4 = nn.Linear(latent_dim, h3_dim)
        self.bn4 = nn.BatchNorm1d(num_features=h3_dim)

        self.fc5 = nn.Linear(h3_dim, h2_dim)
        self.bn5 = nn.BatchNorm1d(num_features=h2_dim)

        self.fc6 = nn.Linear(h2_dim, h1_dim)
        self.bn6 = nn.BatchNorm1d(num_features=h1_dim)

        self.fc7 = nn.Linear(h1_dim, in_dim)
        self.bn7 = nn.BatchNorm1d(num_features=in_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def encode(self, x):
        h1 = self.dropout(self.relu(self.bn1(self.fc1(x))))
        h2 = self.dropout(self.relu(self.bn2(self.fc2(h1))))
        h3 = self.dropout(self.relu(self.bn3(self.fc3(h2))))

        return self.fc31(h3), self.fc32(h3)

    def decode(self, z):
        h4 = self.dropout(self.relu(self.bn4(self.fc4(z))))
        h5 = self.dropout(self.relu(self.bn5(self.fc5(h4))))
        h6 = self.dropout(self.relu(self.bn6(self.fc6(h5))))

        return self.sigmoid(self.bn7(self.fc7(h6)))

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparametrize(mu, log_var)

        return self.decode(z), mu, log_var
