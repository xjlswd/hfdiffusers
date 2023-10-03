import torch
from diffusers import DDIMScheduler
from diffusers.models.unet_2d_class_condition import UNet2DClassConditionModel
from tqdm.auto import tqdm
from diffusers.utils import randn_tensor
from torchvision.utils import save_image

device = "cuda"
img_size = 32
guidance_scale = 7

unet = UNet2DClassConditionModel(
    28,
    in_channels=1,
    out_channels=1,
    cross_attention_dim=512,
    attention_head_dim=16,
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
    block_out_channels=(256, 512, 1024),
    num_class_embeds=10,
    class_condition=True,
)
unet.load_state_dict(torch.load("results/mnist/checkpoint-15000/pytorch_model.bin"))
unet.to(device=device)
noise_scheduler = DDIMScheduler()
class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 4
batch_size = len(class_labels)
latent_size = 28
latent_channels = 1
latents = randn_tensor(
    shape=(batch_size, latent_channels, latent_size, latent_size),
    device=device,
)
latent_model_input = torch.cat([latents] * 2)

class_labels = torch.tensor(class_labels, device=device).reshape(-1)
class_null = torch.tensor([10] * batch_size, device=device)
class_labels_input = torch.cat([class_labels, class_null], 0)

# set step values
noise_scheduler.set_timesteps(250)

for t in tqdm(noise_scheduler.timesteps):
    half = latent_model_input[: len(latent_model_input) // 2]
    latent_model_input = torch.cat([half, half], dim=0)
    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

    with torch.no_grad():
        timesteps = t
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = latent_model_input.device.type == "mps"
            if isinstance(timesteps, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor(
                [timesteps], dtype=dtype, device=latent_model_input.device
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(latent_model_input.device)

        timesteps = timesteps.expand(latent_model_input.shape[0])
        # predict noise model_output
        noise_pred = unet(
            latent_model_input, timestep=timesteps, class_labels=class_labels_input
        ).sample

    # perform guidance
    eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

    half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)

    noise_pred = torch.cat([eps, rest], dim=1)
    model_output = noise_pred

    latent_model_input = noise_scheduler.step(
        model_output, t, latent_model_input
    ).prev_sample

latents, _ = latent_model_input.chunk(2, dim=0)
image = (latents / 2 + 0.5).clamp(0, 1).cpu()

save_image(image, "out.jpg")
