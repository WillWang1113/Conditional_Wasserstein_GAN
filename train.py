import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gan import Generator, Discriminator, MyDataset
from utils import setup_seed


def fit(data_file: str, batch_size: int, lr: tuple, epochs: int,
        clip_value: float, n_critic: int, latent_dim: int, condition: bool):

    # ----------------------------------------------------------------------------------------
    #  Load preprocessed data
    #  dict: {"train_ds": traindata, "val_ds": valdata,"test_ds": testdata, "scaler": scaler}
    # ----------------------------------------------------------------------------------------
    prepared_data = pickle.load(open(data_file, 'rb'))
    dataset = MyDataset(prepared_data["train_ds"])
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=True)
    print(f"total batch:{len(dataloader)}")

    # samples = target, labels = conditions
    samples, labels = next(iter(dataloader))
    print(f"samples: {samples.shape}")
    print(f"labels: {labels.shape}")

    # set cGAN or not
    if condition:
        condition_len = labels.shape[-1]
    else:
        condition_len = 0

    # For Generator: sync input and output (can be not sync)
    generator = Generator(noise_len=latent_dim,
                          condition_len=condition_len,
                          output_feature=samples.shape[-1]).cuda()
    # For Disciminator: input is fake data
    discriminator = Discriminator(
        input_feature=samples.shape[-1],
        condition_len=condition_len,
    ).cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr[0])
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr[1])

    # ----------
    #  Training
    # ----------

    batches_done = 0
    writer = SummaryWriter()
    for epoch in range(epochs):

        for i, (X, con) in enumerate(dataloader):

            # Configure input
            real_data = X

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.normal(0, 1, (X.shape[0], latent_dim)).cuda()
            # z = torch.normal(0, 1, X.shape).cuda()

            # Generate a batch of images
            fake_data = generator(z, con)
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_data, con)) + torch.mean(
                discriminator(fake_data, con))
            writer.add_scalars(
                "Loss", {
                    'D(x)': -torch.mean(discriminator(real_data, con)),
                    'D(G(z))': torch.mean(discriminator(fake_data, con)),
                    'loss_D': loss_D
                }, batches_done)
            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in discriminator.parameters():
                # p.clamp_(clip_value, clip_value)
                p.data.clamp_(-clip_value, clip_value)

            # Train the generator every n_critic iterations
            if i % n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z, con)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs, con))

                loss_G.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                        (epoch, epochs, batches_done % len(dataloader),
                        len(dataloader), loss_D.item(), loss_G.item()))

            batches_done += 1

    # save model with names
    save_name = data_file.split('_')[0]
    torch.save(generator, f"{save_name}_C{condition}_G.pth")
    torch.save(discriminator, f"{save_name}_C{condition}_D.pth")


if __name__ == "__main__":
    setup_seed(9)
    # -----------------
    #       wind
    # -----------------
    fit(data_file='loggings/wind_preprocess.pkl',
        batch_size=16,
        lr=(5e-5, 5e-5),
        epochs=600,
        clip_value=1e-2,
        n_critic=5,
        latent_dim=96,
        condition=True)

    # -----------------
    #        pv
    # -----------------
    # fit(data_file='loggings/pv_preprocess.pkl',
    #     batch_size=16,
    #     lr=(5e-5, 5e-5),
    #     epochs=600,
    #     clip_value=1e-2,
    #     n_critic=5,
    #     latent_dim=96,
    #     condition=True)
