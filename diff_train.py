from torch.optim import Adam
import torch
from DiffusionModel.reverse_diff import get_loss
from DiffusionModel.utils import sample_plot_image
from DiffusionModel.unet import Unet
from DiffusionModel.dataset import get_dataset_and_dataloader
from DiffusionModel.noise_scheduler import precalculate_forward_diffusion




def train(epochs, dataloader, model, IMG_SIZE,timesteps,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod,BATCH_SIZE=128 ,device = "cpu"):
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            t = torch.randint(0, timesteps, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch[0], t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            loss.backward()
            optimizer.step()

        if epoch % 5 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
            sample_plot_image()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet()
    timesteps = 10
    model.to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 10
    IMG_SIZE = 32
    BATCH_SIZE = 128
    DataLoader = get_dataset_and_dataloader(IMG_SIZE, BATCH_SIZE)
    betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas , sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = precalculate_forward_diffusion(timesteps)
    train(epochs, DataLoader, model, IMG_SIZE, timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, BATCH_SIZE, device)
    
