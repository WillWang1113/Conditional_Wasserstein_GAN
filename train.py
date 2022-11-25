import torch
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gan import Generator, Discriminator, MyDataset


def fit(data_file: str, batch_size: int, lr: tuple, epochs: int,
        clip_value: float, n_critic: int):
    cuda = True if torch.cuda.is_available() else False

    # --------------------------------------------------------------------------------------
    # Load preprocessed data
    # dict: {"train_ds": traindata, "val_ds": valdata,"test_ds": testdata, "scaler": scaler}
    # --------------------------------------------------------------------------------------
    prepared_data = pickle.load(open(data_file, 'rb'))
    dataset = MyDataset(prepared_data["train_ds"])
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    print(f"total batch:{len(dataloader)}")

    features, _ = next(iter(dataloader))
    input_shape = features.shape

    # For Generator: sync input and output (can be not sync)
    generator = Generator(input_feature=input_shape[-1],
                          output_feature=input_shape[-1]).cuda()
    # For Disciminator: input is fake data
    discriminator = Discriminator(input_feature=input_shape[-1]).cuda()

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=lr[0])
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=lr[1])

    # ----------
    #  Training
    # ----------

    batches_done = 0
    writer = SummaryWriter()
    for epoch in range(epochs):

        for i, (X, _) in enumerate(dataloader):

            # Configure input
            real_data = X

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = torch.normal(0, 1, input_shape).cuda()

            # Generate a batch of images
            fake_data = generator(z)
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_data)) + torch.mean(
                discriminator(fake_data))
            writer.add_scalar("Loss/loss_dis", loss_D, batches_done)
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
                gen_imgs = generator(z)
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))
                writer.add_scalar("Loss/loss_gen", loss_G, batches_done)


                loss_G.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                      (epoch, epochs, batches_done % len(dataloader),
                       len(dataloader), loss_D.item(), loss_G.item()))

            batches_done += 1
    torch.save(generator, "loggings/generator.pth")
    torch.save(discriminator, "loggings/discriminator.pth")
