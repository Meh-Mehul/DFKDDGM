import torch.nn.functional as F
from DiffusionModel.noise_scheduler import forward_diffusion_sample, get_index_from_list
import torch

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    x_noisy, noise = x_noisy.to(device), noise.to(device)  # Ensure tensors are on the same device as the model
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

# def get_loss(model, x_0, t,sqrt_alphas_cumprod,sqrt_one_minus_alphas_cumprod, device = "cpu"):
#     x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
#     noise_pred = model(x_noisy, t)
#     return F.l1_loss(noise, noise_pred)

## As mentioned in paper, this is the reverse diffusion process
@torch.no_grad()
def sample_timestep(x, t, betas, sqrt_one_minus_alphas_cumprod,sqrt_recip_alphas, model, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 




        