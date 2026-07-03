import torch
import torch.nn as nn
from diffusers import UNet2DModel

class NoiseSchedule(nn.Module):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        super().__init__()
        self.num_timesteps = num_timesteps

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas', alphas)

    def get_noise_at_t(self, x_0, t, noise):
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        sqrt_alpha = sqrt_alpha.reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, 1, 1, 1)

        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

class ConditionEmbedding(nn.Module):
    def __init__(self, num_objects=24, img_size=64):
        super().__init__()
        self.num_objects = num_objects
        self.img_size = img_size

        self.fc = nn.Sequential(
            nn.Linear(num_objects, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * img_size * img_size)
        )

    def forward(self, condition):
        batch_size = condition.shape[0]

        cond_map = self.fc(condition)
        cond_map = cond_map.reshape(batch_size, 3, self.img_size, self.img_size)
        return cond_map

class ConditionalUNet(nn.Module):
    def __init__(self, num_objects=24):
        super().__init__()
        self.cond_emb = ConditionEmbedding(num_objects=num_objects, img_size=64)

        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=6,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x, t, condition):

        cond_map = self.cond_emb(condition)
        x_cond = torch.cat([x, cond_map], dim=1)
        out = self.unet(x_cond, t).sample
        
        return out

class DDPM(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super().__init__()
        self.model = model
        self.noise_schedule = NoiseSchedule(num_timesteps)
        self.num_timesteps = num_timesteps

    def forward(self, x_0, t, condition):
        noise = torch.randn_like(x_0)
        x_t = self.noise_schedule.get_noise_at_t(x_0, t, noise)
        noise_pred = self.model(x_t, t, condition)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, condition, shape=(1, 3, 64, 64), device='cuda', return_intermediates=False):
        x = torch.randn(shape).to(device)
        intermediates = []
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.tensor([t] * shape[0], dtype=torch.long).to(device)
            noise_pred = self.model(x, t_tensor, condition)

            alpha_t = self.noise_schedule.alphas_cumprod[t]
            alpha_t_prev = self.noise_schedule.alphas_cumprod[max(0, t-1)]
            posterior_variance = (1 - alpha_t_prev) / (1 - alpha_t) * (1 - self.noise_schedule.alphas[t])

            coef1 = 1 / torch.sqrt(self.noise_schedule.alphas[t])
            coef2 = (1 - self.noise_schedule.alphas[t]) / torch.sqrt(1 - alpha_t)

            mean = coef1 * (x - coef2 * noise_pred)

            if t > 0:
                z = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_variance) * z
            else:
                x = mean

            if return_intermediates:
                intermediates.append(x.clone().detach().cpu())

        if return_intermediates:
            return x, intermediates
        return x