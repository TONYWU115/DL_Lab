import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from dataset import iCLEVRDataset
from ddpm import DDPM, ConditionalUNet

def train():
    batch_size = 64
    num_epochs = 200
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = iCLEVRDataset('train.json', './iclevr', 'objects.json', transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    unet = ConditionalUNet(num_objects=24).to(device)
    ddpm = DDPM(unet, num_timesteps=1000).to(device)

    optimizer = optim.Adam(ddpm.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        total_loss = 0
        for images, conditions in dataloader:
            images = images.to(device)
            conditions = conditions.to(device)

            t = torch.randint(0, 1000, (images.shape[0],)).to(device)

            loss = ddpm(images, t, conditions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()

    torch.save({'model': ddpm.state_dict()}, 'ddpm_model.pth')
    tqdm.write("Model saved!")

if __name__ == '__main__':
    train()