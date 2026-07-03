import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import iCLEVRDataset
from GAN import Generator, Discriminator


def train():
    batch_size = 64
    num_epochs = 200
    lr = 1e-4
    noise_dim = 128
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = iCLEVRDataset('train.json', './iclevr', 'objects.json', transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    G = Generator(noise_dim=noise_dim, num_objects=24).to(device)
    D = Discriminator(num_objects=24).to(device)

    criterion = nn.BCEWithLogitsLoss()
    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        g_loss_sum = 0.0
        d_loss_sum = 0.0

        for real_images, conditions in dataloader:
            real_images = real_images.to(device)
            conditions = conditions.to(device)

            batch_size_now = real_images.size(0)
            real_targets = torch.ones(batch_size_now, 1, device=device)
            fake_targets = torch.zeros(batch_size_now, 1, device=device)

            # Train Discriminator
            z = torch.randn(batch_size_now, noise_dim, device=device)
            fake_images = G(z, conditions).detach()

            real_logits = D(real_images, conditions)
            fake_logits = D(fake_images, conditions)

            d_loss_real = criterion(real_logits, real_targets)
            d_loss_fake = criterion(fake_logits, fake_targets)
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            z = torch.randn(batch_size_now, noise_dim, device=device)
            generated_images = G(z, conditions)
            gen_logits = D(generated_images, conditions)
            g_loss = criterion(gen_logits, real_targets)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()

        avg_g_loss = g_loss_sum / len(dataloader)
        avg_d_loss = d_loss_sum / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

        torch.save(
            {
                'G': G.state_dict(),
                'D': D.state_dict(),
                'epoch': epoch + 1,
            },
            'gan.pth'
        )


if __name__ == '__main__':
    train()