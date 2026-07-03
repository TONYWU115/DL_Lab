import json
import os
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms

from evaluator import evaluation_model
from GAN import Generator


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_dim = 128

    G = Generator(noise_dim=noise_dim, num_objects=24).to(device)
    checkpoint = torch.load('gan.pth', map_location=device)
    G.load_state_dict(checkpoint['G'])
    G.eval()

    evaluator = evaluation_model()

    with open('objects.json', 'r') as f:
        objects = json.load(f)

    os.makedirs('./images', exist_ok=True)

    def generate_for_json(json_file, output_dir):
        with open(json_file, 'r') as f:
            test_data = json.load(f)

        os.makedirs(output_dir, exist_ok=True)

        all_images = []
        all_labels = []

        for idx, obj_list in enumerate(test_data):
            condition = torch.zeros(1, 24, device=device)
            for obj in obj_list:
                condition[0, objects[obj]] = 1

            z = torch.randn(1, noise_dim, device=device)
            with torch.no_grad():
                fake_image = G(z, condition)

            fake_image = (fake_image + 1) / 2
            vutils.save_image(fake_image, f'{output_dir}/{idx}.png')

            all_images.append(fake_image)
            all_labels.append(condition)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_images_norm = transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )(all_images)

        acc = evaluator.eval(all_images_norm, all_labels)
        print(f'{json_file} Accuracy: {acc:.4f}')

        grid = vutils.make_grid(all_images[:32], nrow=8, normalize=True)
        vutils.save_image(grid, f'{output_dir}_grid.png')

    generate_for_json('test.json', './images/test_GAN')
    generate_for_json('new_test.json', './images/new_test_GAN')


if __name__ == '__main__':
    test()