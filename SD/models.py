import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor
from pathlib import Path
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
check_min_version("0.10.0.dev0")
logger = get_logger(__name__)
# STABLE_DIFFUSION = "runwayml/stable-diffusion-v1-5"
# SD_REVISION = None
# SEED = 5000

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    """Import appropriate text encoder class based on model architecture
    
    :
        pretrained_model_name_or_path: Name/path of pretrained model
        revision: Model revision version
    
    Returns:
        Text encoder class
    """
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        torch_dtype=torch.float16,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    else:
        raise ValueError(f"Unsupported model class: {model_class}")


def load_models_and_tokenizer(accelerator, STABLE_DIFFUSION, SD_REVISION, SEED):
    """
    Load all necessary models and the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="tokenizer",
        revision=SD_REVISION,
        use_fast=False,
        torch_dtype=torch.float16
    )
    text_encoder_cls = import_model_class_from_model_name_or_path(
        STABLE_DIFFUSION, SD_REVISION
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="scheduler",
        torch_dtype=torch.float16
    )
    text_encoder = text_encoder_cls.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="text_encoder",
        revision=SD_REVISION,
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="vae",
        revision=SD_REVISION,
        torch_dtype=torch.float16
    )
    unet = UNet2DConditionModel.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="unet",
        revision=SD_REVISION,
        torch_dtype=torch.float16
    )
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="safety_checker",
        revision=SD_REVISION,
        torch_dtype=torch.float16
    )
    feature_extractor = CLIPFeatureExtractor.from_pretrained(
        STABLE_DIFFUSION,
        subfolder="feature_extractor",
        revision=SD_REVISION,
        torch_dtype=torch.float16
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)  # Set seed for reproducibility
    # Freeze models
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Optimize CUDA settings
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # Prepare for distributed training
    unet, text_encoder, tokenizer, generator, noise_scheduler, vae, safety_checker, feature_extractor = accelerator.prepare(
        unet, text_encoder, tokenizer, generator, noise_scheduler, vae, safety_checker, feature_extractor
    )
    
    # Set weight data type
    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    return tokenizer, text_encoder, noise_scheduler, vae, unet, safety_checker, feature_extractor, generator, weight_dtype



def setup_accelerator_and_logging(SEED):
    """
    Initialize the accelerator and set up logging.
    """
    logging_dir = Path("./output_dir", "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="tensorboard",
        project_dir=logging_dir,
    )
    if SEED is not None:
        set_seed(SEED)
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth", config=vars())
    return accelerator

def prepare_uncond_embeddings(tokenizer, text_encoder, unet, batch_size_generation):
    """
    Prepare unconditional embeddings.
    """
    uncond_inputs = tokenizer(
        ['' for _ in range(batch_size_generation)],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    uncond_input_ids = uncond_inputs.input_ids.to(unet.device)
    uncond_embeddings = text_encoder(uncond_input_ids)[0]
    return uncond_embeddings
