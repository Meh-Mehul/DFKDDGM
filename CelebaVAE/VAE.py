import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# Define hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-5
LATENT_DIM = 128
IMAGE_SIZE = 512
CHANNELS = 3  # Assuming RGB images


# Define the VAE model with exactly 4 layers in encoder and decoder
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder - exactly 4 convolutional layers
        self.encoder = nn.Sequential(
            # Layer 1: 512x512 -> 256x256
            nn.Conv2d(CHANNELS, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Layer 2: 256x256 -> 128x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Layer 3: 128x128 -> 64x64
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Layer 4: 64x64 -> 32x32
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        # Flatten and project to latent space
        self.flatten_size = 512 * 32 * 32
        
        self.fc_mu = nn.Sequential(
            nn.Linear(self.flatten_size, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.flatten_size, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim, self.flatten_size),
            nn.BatchNorm1d(self.flatten_size)
        )
        
        # Decoder - exactly 4 transposed convolutional layers
        self.decoder = nn.Sequential(
            # Layer 1: 32x32 -> 64x64
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Layer 2: 64x64 -> 128x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Layer 3: 128x128 -> 256x256
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 4: 256x256 -> 512x512
            nn.ConvTranspose2d(64, CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(x.size(0), 512, 32, 32)  # Reshape
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

