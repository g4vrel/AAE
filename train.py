import numpy as np

import torch
from torch.nn import BCELoss
import torch.nn.functional as F

from modules import Encoder, Decoder, Discriminator
from utils import get_loaders
from eval import make_manifold, draw_hiddencode, generation

torch.manual_seed(159753)
np.random.seed(159753)


if __name__ == '__main__':
    import os

    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)

    device = 'cuda'

    config = {
        'lr_encoder': 1e-3,
        'lr_decoder': 1e-3,
        'lr_discriminator': 1e-3,
        'epochs': 150,
        'encoder_hidden': 512,
        'latent_dim': 2,
        'decoder_hidden': 512,
        'discriminator_hidden': 512,
        'input_shape': (28, 28),
        'batch_size':128,
        'num_workers':3,
        'print_freq':400,
    }

    encoder = Encoder(input_dim=np.array(config['input_shape']).prod(),
                      hidden_dim=config['encoder_hidden'],
                      output_dim=config['latent_dim']).to(device)

    decoder = Decoder(input_dim=config['latent_dim'],
                      hidden_dim=config['decoder_hidden'],
                      output_dim=np.array(config['input_shape']).prod()).to(device)

    discriminator = Discriminator(input_dim=config['latent_dim'],
                                  hidden_dim=config['discriminator_hidden']).to(device)

    train_loader, eval_loader = get_loaders(config)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=config['lr_encoder'], betas=(0.5, 0.999))
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=config['lr_decoder'], betas=(0.5, 0.999))
    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=config['lr_discriminator'], betas=(0.5, 0.999))

    adv_loss = BCELoss()

    step = 0

    ones = torch.ones((config['batch_size'], 1), device=device)
    zeros = torch.zeros((config['batch_size'], 1), device=device)

    for epoch in range(config['epochs']):
        encoder.train()
        decoder.train()
        discriminator.train()

        for data, _ in train_loader:
            # Reconstruction phase

            optim_encoder.zero_grad()
            optim_decoder.zero_grad()

            data = data.to(device)

            fake_noise = encoder(data)
            rec_data = decoder(fake_noise)

            reconstruction_loss = F.mse_loss(rec_data, data.view(-1, 784))
            reconstruction_loss.backward()

            optim_encoder.step()
            optim_decoder.step()
            
            # Regularization phase

            optim_discriminator.zero_grad()

            fake_noise = encoder(data).detach()
            real_noise = torch.randn((config['batch_size'], config['latent_dim']), device=device)

            fake = discriminator(fake_noise)
            real = discriminator(real_noise)
            
            discriminator_loss = adv_loss(fake, zeros) + adv_loss(real, ones)
            discriminator_loss.backward()

            dis_norm = torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)

            optim_discriminator.step()

            optim_encoder.zero_grad()

            fake_noise = encoder(data)
            fake = discriminator(fake_noise)
            generator_loss = adv_loss(fake, ones)
            generator_loss.backward()

            optim_encoder.step()

            if step % config["print_freq"] == 0:                
                print("Epoch: {:2.0f} | Rec = {:.6f} | Dis: {:.4f} | Reg: {:.4f}".format(
                    epoch, reconstruction_loss.item(), discriminator_loss.item(), generator_loss.item()
                ))

            step += 1
        
        if epoch % 10 == 0:
            draw_hiddencode(epoch, encoder, eval_loader)
            generation(epoch, decoder)

    draw_hiddencode('final', encoder, eval_loader)
    make_manifold('final', 289, decoder)

    os.makedirs('saved_models/2dgaussian')

    torch.save(encoder.state_dict(), f'saved_models/2dgaussian/encoder_{step}.pt')
    torch.save(decoder.state_dict(), f'saved_models/2dgaussian/decoder_{step}.pt')
    torch.save(discriminator.state_dict(), f'saved_models/2dgaussian/discriminator_{step}.pt')