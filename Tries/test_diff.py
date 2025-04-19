import torch
from DiffusionModel.unet import Unet
from DiffusionModel.reverse_diff import sample_timestep
from DiffusionModel.noise_scheduler import precalculate_forward_diffusion
from DiffusionModel.utils import sample_plot_image
import matplotlib.pyplot as plt

def load_model(model_path, device="cpu"):
    """Load the trained model from the specified path."""
    model = Unet()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model

def denoise_and_plot(model, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device="cpu"):
    """Perform reverse diffusion and plot the denoised images."""
    # Start with random noise
    x = torch.randn((1, 3, 64, 64), device=device)  # Assuming 32x32 images with 3 channels
    for t in reversed(range(timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        x = sample_timestep(x, t_tensor, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, model, posterior_variance)
    
    # Plot the final denoised image
    plt.figure(figsize=(4, 4))
    plt.imshow(x.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    plt.title("Denoised Image")
    plt.axis("off")
    plt.savefig("./made.png")
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "./unet-cifar10-250" 
    timesteps = 500
    model = load_model(model_path, device)
    betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance = precalculate_forward_diffusion(timesteps)
    denoise_and_plot(model, timesteps, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance, device)