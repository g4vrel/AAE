import numpy as np

import torch
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image

import matplotlib.pyplot as plt


@torch.no_grad()
def generation(epoch, decoder, shape=(64, 2), device='cuda'):
    decoder.eval()

    real_noise = torch.randn(shape, device=device)
    gen_data = decoder(real_noise).view(64, 1, 28, 28)

    save_image(gen_data.cpu(), 'results/generation_' + str(epoch) + '.png')


@torch.no_grad()
def draw_hiddencode(name, encoder, eval_loader, device='cuda'):
    encoder.eval()

    points = []
    labels = []

    for data, label in eval_loader:
        data = data.to(device)

        fake_noise = encoder(data)
        
        points.append(fake_noise.cpu().numpy())
        labels.append(label.cpu().numpy())
    
    positions = np.concatenate(points, axis=0)
    labels = np.concatenate(labels, axis=0)

    mnist_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    colors = [mnist_colors[int(label)] for label in labels]

    fig, ax = plt.subplots()
    
    scatter = ax.scatter(positions[:,0], positions[:,1], s=1, c=colors)
    
    unique_labels = np.unique(labels)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=mnist_colors[int(label)], markersize=5)
            for label in unique_labels]
    legend_labels = [f'{int(label)}' for label in unique_labels]
    ax.legend(handles, legend_labels, loc='best')
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.savefig('results/hiddencode_' + str(name) + '.png', dpi=300)
    plt.close(fig)


@torch.no_grad()
def make_manifold(name, num_samples, decoder, device='cuda'):
    decoder.eval()

    n = int(torch.sqrt(torch.tensor(num_samples)))

    min_x, max_x = -2, 2
    min_y, max_y = -2, 2

    x = torch.linspace(min_x, max_x, n)
    y = torch.linspace(max_y, min_y, n)

    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    zs = torch.stack([grid_x, grid_y], dim=-1)

    points = zs.reshape(num_samples, 2).to(device)

    rec_x = decoder(points).view(-1, 1, 28, 28)
    
    grid = make_grid(rec_x, nrow=17)

    im = ToPILImage()(grid)
    im.save(f'results/manifold_{name}.png')