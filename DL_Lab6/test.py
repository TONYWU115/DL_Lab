import json
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from ddpm import DDPM, ConditionalUNet
from evaluator import evaluation_model
import os

torch.manual_seed(42)
def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    unet = ConditionalUNet(num_objects=24).to(device)
    ddpm = DDPM(unet).to(device)
    checkpoint = torch.load('ddpm_model_epoch_200.pth')
    ddpm.load_state_dict(checkpoint['model'])
    ddpm.eval()

    evaluator = evaluation_model()

    with open('objects.json', 'r') as f:
        objects = json.load(f)

    special_labels = ["red sphere", "cyan cube","cyan cylinder"]
    one_hot = torch.zeros(1, 24).to(device)
    for obj in special_labels:
        one_hot[0, objects[obj]] = 1

    with torch.no_grad():

        final_img, intermediates = ddpm.sample(one_hot, device=device, return_intermediates=True)

    step_indices = torch.linspace(0, len(intermediates)-1, steps=10).long()
    grid_imgs = torch.cat([intermediates[i] for i in step_indices], dim=0)
    grid_imgs = (grid_imgs + 1) / 2
    os.makedirs('./images', exist_ok=True)
    vutils.save_image(grid_imgs, './images/denoise_process_grid.png', nrow=10, normalize=False)

    for test_file in ['test.json', 'new_test.json']:
        with open(test_file, 'r') as f:
            test_data = json.load(f)

        os.makedirs(f'./images/{test_file.split(".")[0]}', exist_ok=True)

        all_images = []
        all_labels = []

        for idx, obj_list in enumerate(test_data):
            one_hot = torch.zeros(1, 24).to(device)
            for obj in obj_list:
                one_hot[0, objects[obj]] = 1

            with torch.no_grad():
                generated_image = ddpm.sample(one_hot, device=device)

            generated_image = (generated_image + 1) / 2

            vutils.save_image(generated_image,
                            f'./images/{test_file.split(".")[0]}/{idx}.png')

            all_images.append(generated_image)
            all_labels.append(one_hot)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_images_norm = transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))(all_images)

        accuracy = evaluator.eval(all_images_norm, all_labels)
        print(f"{test_file} Accuracy: {accuracy:.4f}")

        grid = vutils.make_grid(all_images[:32], nrow=8, normalize=True)
        vutils.save_image(grid, f'./images/{test_file.split(".")[0]}_grid.png')

if __name__ == '__main__':
    test()