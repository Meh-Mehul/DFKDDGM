import torch
from SD.models import set_seed
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
from SD.models import setup_accelerator_and_logging, load_models_and_tokenizer \
                        , prepare_uncond_embeddings
from SD.helper import generate_class_prompts, precompute_text_embeddings
from VAE.train_tvae import VAE
# save_syn_data_path = "./synthetic_data/"  
# batch_size_generation = 1 ## Batch size while synthesis
# inference_nums = 5 ## Steps for inference of SD
# guided_scale = 3 ## Hyperparameter for CFG
# ## Weights for the three losses
# oh = 1
# bn = 0
# adv = 0
# class_per_num_start = 10

def generate_images(accelerator, class_index, class_text_embeddings, uncond_embeddings, noise_scheduler, vae, unet, model,
                      generator, weight_dtype, syn_image_seed, config):
    """
    Generate synthetic images for a single class with adversarial training.
    """
    image_save_dir_path = os.path.join(config.save_syn_data_path, str(class_index))
    os.makedirs(image_save_dir_path, exist_ok=True)
    
    text_embeddings = class_text_embeddings[class_index]    
    syn_image_seed += 1
    generator.manual_seed(syn_image_seed)
    set_seed(syn_image_seed)
    torch.cuda.empty_cache()
    with accelerator.accumulate(unet):
        latents_shape = (config.batch_size_generation, unet.config.in_channels, 64, 64)
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device="cpu",
            dtype=weight_dtype
        ).to(unet.device)
        latents = latents * noise_scheduler.init_noise_sigma
        noise_scheduler.set_timesteps(config.inference_nums)
        timesteps_tensor = noise_scheduler.timesteps.to(latents.device)
        timestep_nums = 0

        for timesteps in timesteps_tensor[:-1]:
            # Enable gradients for necessary tensors
            text_embeddings.requires_grad_(True)
            uncond_embeddings.requires_grad_(True)
            latents.requires_grad_(True)

            input_embeddings = torch.cat([uncond_embeddings, text_embeddings], dim=0)
            latent_model_input = torch.cat([latents] * 2)

            model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
            uncond_pred, text_pred = model_preds.chunk(2)
            model_pred = uncond_pred + config.guided_scale * (text_pred - uncond_pred)
            # Calculate original latents
            with torch.no_grad():
                ori_latents = noise_scheduler.step(
                    model_pred,
                    timesteps.cpu(),
                    latents,
                    generator=generator
                ).pred_original_sample.half()

            input_latents = 1 / 0.18215 * ori_latents
            image = vae.decode(input_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)

            loss = 0.0
            loss_m = torch.tensor(0.0, device=unet.device)
            loss_kl = torch.tensor(0.0, device=unet.device)

            if (config.m + config.kl) > 0:
                image = F.interpolate(image, size=(32, 32), mode='bilinear', align_corners=False)
                recon_image, miu, sigma = model(image)
                recon_image = recon_image.squeeze(0)
                recon_image = recon_image.unsqueeze(0)
                loss_m = F.mse_loss(recon_image, image, reduction="sum")
                # loss_kl = -0.5 * torch.sum(1 + sigma - miu.pow(2) - sigma.exp()) 
                sigma = F.softplus(sigma)
                loss_kl = -0.5 * torch.sum(1 + torch.log(sigma ** 2 + 1e-8) - miu.pow(2) - sigma.pow(2))
                # loss_kl_forward = 0.5 * torch.sum(torch.log(sigma) + (1 + miu.pow(2)) / sigma.exp() - 1)
                # loss_kl = loss_kl_forward
                # print("Pred mean: ", miu, " Pred sig: ", sigma, "losskl: ", loss_kl)
                loss = loss_m*config.m + loss_kl*config.kl 
                # Backpropagate loss
                cond_grad = torch.autograd.grad(loss, latents, allow_unused=True, retain_graph=True)[0]
                if cond_grad is None:
                    cond_grad = torch.zeros_like(latents)
                latents = latents - cond_grad
                # Save intermediate images
                if (timestep_nums + 1) % (config.inference_nums // 5) == 0 and (timestep_nums + 1) != config.inference_nums:
                    for i in range(config.batch_size_generation):
                        image_name = os.path.join(
                            image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s:{timesteps.item():.0f}_m:{loss_m.item():.3f}_kl:{loss_kl.item():.3f}_{i}.jpg"
                        )
                        torchvision.utils.save_image(image[i], image_name)
            else:
                # Save intermediate images without loss
                if (timestep_nums + 1) % (config.inference_nums // 5) == 0 and (timestep_nums + 1) != config.inference_nums:
                    for i in range(config.batch_size_generation):
                        image_name = os.path.join(
                            image_save_dir_path,
                            f"{syn_image_seed}_{class_index}_s:{timesteps.item():.0f}_bn:0.0_oh:0.0_adv:0.0_{i}.jpg"
                        )
                        torchvision.utils.save_image(image[i], image_name)

            timestep_nums += 1

            with torch.no_grad():
                # Apply custom augmentation
                # if customer_aug >= 1 and (timestep_nums % (inference_nums // 5) == 0) and (
                #         timestep_nums != inference_nums):
                #     latents, _, _ = customer_aug_data(latents, customer_aug=customer_aug)
                #     latents = latents.to(dtype=weight_dtype)

                # Predict next latents
                latent_model_input = torch.cat([latents] * 2)
                model_preds = unet(latent_model_input, timesteps, input_embeddings).sample.half()
                uncond_pred, text_pred = model_preds.chunk(2)
                model_pred = uncond_pred + config.guided_scale * (text_pred - uncond_pred)
                # Update latents
                latents = noise_scheduler.step(
                    model_pred,
                    timesteps.cpu(),
                    latents,
                    generator=generator
                ).prev_sample.half()

            # Clear gradients
            unet.zero_grad()
            vae.zero_grad()
            model.zero_grad()
            torch.cuda.empty_cache()

        # Save final images
        with torch.no_grad():
            ori_latents = noise_scheduler.step(
                model_pred,
                timesteps.cpu(),
                latents,
                generator=generator
            ).pred_original_sample.half()
            input_latents = 1 / 0.18215 * ori_latents.detach()
            image = vae.decode(input_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            for i in range(config.batch_size_generation):
                image_name = os.path.join(
                    image_save_dir_path,
                    f"{syn_image_seed}_{class_index}_m:{loss_m.item():.3f}_kl:{loss_kl.item():.3f}_{i}.jpg"
                )
                resized = F.interpolate(image[i].unsqueeze(0), size=(32, 32), mode="bilinear", align_corners=False).squeeze(0)
                torchvision.utils.save_image(resized, image_name)
    return syn_image_seed


def load_teacher_model(trained_dgm_path):
    model = VAE()
    model.load_state_dict(torch.load(trained_dgm_path, weights_only=True))
    model.eval()
    return model


class Config:
    def __init__(self):
        self.SEED = 5000
        self.name = "synthetic_images"
        self.save_syn_data_path = "./synthetic_data" 
        self.checkpoints_dir = "./checkpoints" 
        self.generate_nums = 100  
        self.batch_size_generation = 1
        self.STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"  # Model name from HF
        self.SD_REVISION = None  # Revision
        self.inference_nums = 5  # Number of inference steps
        self.guided_scale = 3  # Guidance scale for classifier-free guidance
        # Loss weights
        self.m = 1.0  # Weight for MSE loss
        self.kl = 1.0  # Weight for KL divergence loss
        # Teacher model settings
        self.trained_dgm_path = "./vae_model.pth" 
        # Dataset and prompts
        self.label_name = True
        self.data_type = "cifar10" 

def main(config:Config):
    """
    Main pipeline for adversarial image generation and model training.
    """
    accelerator = setup_accelerator_and_logging(config.SEED)
    tokenizer, text_encoder, noise_scheduler, vae, unet, safety_checker, feature_extractor, generator, weight_dtype = load_models_and_tokenizer(accelerator, config.STABLE_DIFFUSION, config.SD_REVISION, config.SEED)
    uncond_embeddings = prepare_uncond_embeddings(tokenizer, text_encoder, unet, config.batch_size_generation)
    # model, model_s, hooks, transform = load_classification_models(args, accelerator)
    model = load_teacher_model(config.trained_dgm_path)
    model = model.to('cuda')
    model = model.half() ## To get it to torch.cuda.HalfTensor
    class_prompts = generate_class_prompts(config.label_name, config.data_type)
    class_text_embeddings, class_syn_nums = precompute_text_embeddings(class_prompts, tokenizer, text_encoder, unet,config.batch_size_generation)
    # Create output directories
    config.save_syn_data_path = os.path.join(config.save_syn_data_path, config.name)
    os.makedirs(config.save_syn_data_path, exist_ok=True)
    config.train_data_path = config.save_syn_data_path
    os.makedirs(config.checkpoints_dir, exist_ok=True)
    syn_image_seed = config.SEED
    # best_acc = -1.0
    for class_per_num in tqdm(range(int(config.generate_nums))):
        for class_index in range(len(class_prompts)):
            syn_image_seed = generate_images(accelerator, class_index, class_text_embeddings, uncond_embeddings, noise_scheduler, vae,
                                                unet, model, generator, weight_dtype, syn_image_seed, config)
        # Train and evaluate every 10 generations
        # if (class_per_num + 1) % 10 == 0:
        #     train_dataset, test_dataset = get_dataset(args)
        #     current_best_acc = train_and_evaluate(model, model_s, train_dataset, test_dataset, args, accelerator)
        #     if current_best_acc > best_acc:
        #         best_acc = current_best_acc
        #     if (class_per_num + 1) % 100 == 0:
        #         torch.save(
        #             model_s.state_dict(),
        #             os.path.join(args.checkpoints_dir, f'model-epoch:{class_per_num + 1}-{best_acc:.2f}.pt')
        #         )
    accelerator.wait_for_everyone()
    accelerator.end_training()
    # print(f"Final Best Accuracy: {best_acc:.2f}")

if __name__ == "__main__":
    config = Config()
    main(config)

    