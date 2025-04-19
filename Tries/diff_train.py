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
            loss = get_loss(model, batch[0], t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
            # print("Backpropping...")
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ", flush = True)
            # sample_plot_image()
            torch.save(model.state_dict(),f'./ckpt/unet-cifar10-{epoch}')



device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device: ", device, flush = True)
# device = "cpu"
model = Unet()
print(flush = True)
timesteps = 500
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 250
IMG_SIZE = 64
BATCH_SIZE = 128
DataLoader = get_dataset_and_dataloader(IMG_SIZE, BATCH_SIZE)
betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas , sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = precalculate_forward_diffusion(timesteps)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush = True)
# print(model.state_dict(), flush = True)
train(epochs, DataLoader, model, IMG_SIZE, timesteps, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, BATCH_SIZE, device)
print("Training Done", flush = True)    
print("Saving Model", flush = True)
torch.save(model.state_dict(),f'unet-cifar10-{epochs}')
