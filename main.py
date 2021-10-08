import time

import imageio as imageio
import numpy as np
import sio as sio
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader
from Model2 import CustomModel2
from Model import CustomModel
from DecryptionModel import DecryptionModel
from DataLoader import LoadImages

from CustomModel2 import CustomModel2


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 2 every 3 epochs"""
    lr = optimizer.param_groups[0]["lr"]
    if epoch % 3 == 1:
        if epoch > 1:
            lr = optimizer.param_groups[0]["lr"] / 2

    return lr


def eval(model, device, dataset_evaluated, batch_size, nb_image_to_print):
    model.to(device)
    model.eval()
    with torch.no_grad():
        general_avg = 0.0
        avg_elapsed_time = 0.0
        i = 0
        for batch_id, batch in enumerate(dataset_evaluated):
            i += 1
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            start_time = time.time()
            out_pred = model(images)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time
            val = torch.abs(labels - out_pred).sum().cpu()
            if i < nb_image_to_print:
                fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(12, 4))
                idx = 1
                ax = fig.add_subplot(1, 3, idx, xticks=[], yticks=[])
                plt.imshow(labels[i].cpu().permute(1, 2, 0))
                ax.set_title("Labels")
                idx += 1
                ax = fig.add_subplot(1, 3, idx, xticks=[], yticks=[])
                plt.imshow(images[i].cpu().permute(1, 2, 0))
                ax.set_title("Images")
                idx += 1
                ax = fig.add_subplot(1, 3, idx, xticks=[], yticks=[])
                plt.imshow(out_pred[i].cpu().permute(1, 2, 0))
                ax.set_title("Predicted")
                idx += 1
                plt.show()
            print("Metric for a batch: " + str(val))
            print("Moyenne du batch par image: " + str(val / batch_size))
            general_avg = general_avg + (val / batch_size)
    print("Moyenne general d'une image sur le dataset évalué: " + str(general_avg / i))


def train_optim(model, device, epochs, log_frequency, learning_rate=1e-4):
    model.to(device)  # we make sure the model is on the proper device

    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        lr = adjust_learning_rate(optimizer, epoch)  # adjust the learning rate. Decreasing.
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        model.train()  # we specify that we are training the model

        start = time.time()  # start timer

        # At each epoch, the training set will be processed as a set of batches
        for batch_id, batch in enumerate(trainloader):

            images, labels = Variable(batch[0] / 255), Variable(batch[1] / 255, requires_grad=False)

            # we put the data on the same device
            images, labels = images.to(device), labels.to(device)

            y_pred = model(images)

            loss = loss_fn(y_pred, labels)

            if batch_id % log_frequency == 0:
                print(
                    "epoch: {:03d}, batch: {:03d}, loss: {:.12f}, time: {:.3f}".format(epoch + 1, batch_id + 1,
                                                                                      loss.item(),
                                                                                      time.time() - start))

            optimizer.zero_grad()  # clear the gradient before backward
            loss.backward()  # update the gradient

            optimizer.step()  # update the model parameters using the gradient


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    o_dataset = "./dataset/train_original_tiny.npy"
    t_dataset = "./dataset/train_1A_tiny.npy"
    dataset = LoadImages(t_dataset, o_dataset)
    batch_size = 4
    # dataloader = DataLoader(dataset, batch_size, shuffle=True)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    # model = CustomModel()
    model = CustomModel2()
    nb_epoch = 3
    log_frequency = 10
    learning_rate = 1e-4
    # model.load_state_dict(torch.load("./modeltrained/modelTrained_2_Tiny_3epoch.pt"))
    train_optim(model, device, nb_epoch, log_frequency, learning_rate)
    torch.save(model.state_dict(), './modeltrained/modelTrained_2_Tiny_3epoch.pt')
    eval(model, device, testloader, batch_size, 10)
