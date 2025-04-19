import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from train_tvae import VAE  # Import the VAE class from the training script

def sample_from_vae():
    config = {
        "latent_space_dim": 128,
        "input_channels": 3,
        "image_size": 32,  # Should match the size used during training
        "save_model_path": "./vae_model.pth",  # Path to the trained model
        "output_image_path": "./reconstructed_image.png",  # Path to save the reconstructed image
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Load the trained VAE model
    vae = VAE(input_channels=config["input_channels"], latent_space_dim=config["latent_space_dim"]).to(config["device"])
    vae.load_state_dict(torch.load(config["save_model_path"], map_location=config["device"]))
    vae.eval()

    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])
    cifar10 = datasets.CIFAR10(root="./data", train=False, transform=transform, download=True)
    dataloader = torch.utils.data.DataLoader(cifar10, batch_size=1, shuffle=True)

    # Get a single image from CIFAR-10
    original_image, _ = next(iter(dataloader))
    original_image = original_image.to(config["device"])

    # Pass the image through the VAE
    with torch.no_grad():
        h = vae.encoder(original_image)  # Encode the image
        h = h.view(h.size(0), -1)  # Flatten the feature maps
        miu = vae.hidden_to_miu(h)  # Compute mean
        sigma = vae.hidden_to_sigma(h)  # Compute variance
        z = miu + torch.randn_like(sigma) * sigma  # Reparameterization trick
        h = vae.latent_to_hidden(z)  # Map latent vector to hidden space
        h = h.view(h.size(0), 256, 4, 4)  # Reshape to match decoder input
        reconstructed_image = vae.decoder(h)  # Decode to reconstruct the image
        reconstructed_image = reconstructed_image.squeeze(0).cpu()  # Remove batch dimension and move to CPU

    # Plot and save the original and reconstructed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_image.squeeze(0).permute(1, 2, 0).cpu().numpy())  # Original image
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_image.permute(1, 2, 0).numpy())  # Reconstructed image
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")

    plt.savefig(config["output_image_path"], bbox_inches="tight", pad_inches=0)
    print(f"Reconstructed image saved to {config['output_image_path']}")

if __name__ == "__main__":
    sample_from_vae()