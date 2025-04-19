from torch import nn, optim
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
class VAE(nn.Module):
    def __init__(self, input_channels=3, latent_space_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),            # 8x8 -> 4x4
            nn.ReLU()
        )
        
        # Latent space
        self.hidden_to_miu = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.hidden_to_sigma = nn.Linear(256 * 4 * 4, latent_space_dim)
        self.latent_to_hidden = nn.Linear(latent_space_dim, 256 * 4 * 4)        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1), # 16x16 -> 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        miu = self.hidden_to_miu(h)
        sigma = self.hidden_to_sigma(h)
        e = torch.randn_like(sigma)
        z_reparam = miu + e * sigma
        h = self.latent_to_hidden(z_reparam)
        h = h.view(h.size(0), 256, 4, 4)
        recon_x = self.decoder(h)
        return recon_x, miu, sigma
def train_vae():
    config = {
        "batch_size": 16,
        "epochs": 200,
        "learning_rate": 3e-4,
        "latent_space_dim": 128,
        "input_channels": 3,
        "image_size": 32,
        "dataset_path": "./data",
        "save_model_path": "./vae_model.pth",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    print("Model config: " , flush=True)
    print(config, flush=True)
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])
    train_dataset = datasets.CIFAR10(root=config["dataset_path"], train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    vae = VAE(input_channels=config["input_channels"], latent_space_dim=config["latent_space_dim"]).to(config["device"])
    optimizer = optim.Adam(vae.parameters(), lr=config["learning_rate"])
    mse_loss = nn.MSELoss()

    # Training loop
    for epoch in range(config["epochs"]):
        vae.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(config["device"])
            recon_images, miu, sigma = vae(images)
            recon_loss = mse_loss(recon_images, images)
            kl_loss = -0.5 * torch.sum(1 + sigma - miu.pow(2) - sigma.exp()) / images.size(0)
            loss = recon_loss + kl_loss            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {total_loss/len(train_loader):.4f}", flush=True)
    torch.save(vae.state_dict(), config["save_model_path"])
    print(f"Model saved to {config['save_model_path']}", flush=True)
if __name__ == "__main__":
    train_vae()