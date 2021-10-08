import time

import imageio as imageio
import numpy as np
import sio as sio
import torch
from matplotlib import pyplot as plt
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader
from Model3 import CustomModel3
from Model2 import CustomModel2
from Model import CustomModel
from DecryptionModel import DecryptionModel
from DataLoader import LoadImages


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 2 every 3 epochs"""
    lr = optimizer.param_groups[0]["lr"]
    if epoch % 3 == 1:
        if epoch > 1:
            lr = optimizer.param_groups[0]["lr"] / 2

    return lr


def eval(model, device, dataset_evaluated, batch_size, nb_image_to_print):
    if nb_image_to_print > batch_size:
        nb_image_to_print = batch_size
    model.to(device)
    model.eval()
    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)
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
            loss = loss_fn(out_pred, labels)
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
            print("Metric for a batch: " + str(val.item()) + "loss:{:.12f}", loss.item())
            print("Moyenne du batch par image: " + str(val.item() / batch_size))
            general_avg = general_avg + (val.item() / batch_size)
    print("Moyenne general d'une image sur le dataset évalué: " + str(general_avg / i))


def train_optim(model, device, epochs, log_frequency, learning_rate=1e-4):
    model.to(device)  # we make sure the model is on the proper device

    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    previous_loss = 99.0
    loss_sup_in_a_row = 0
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
                    "epoch: {:03d}, batch: {:03d}, loss: {:.3f}, time: {:.3f}".format(epoch + 1, batch_id + 1,
                                                                                      loss.item(),
                                                                                      time.time() - start))

            optimizer.zero_grad()  # clear the gradient before backward
            loss.backward()  # update the gradient

            optimizer.step()  # update the model parameters using the gradient

        # validation
        for batch_id, batch in enumerate(valloader):
            model.eval()
            images, labels = Variable(batch[0] / 255), Variable(batch[1] / 255, requires_grad=False)
            images, labels = images.to(device), labels.to(device)
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            if previous_loss < loss:
                loss_sup_in_a_row += 1
                if loss_sup_in_a_row > 3:
                    return
            else:
                loss_sup_in_a_row = 0


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    o_dataset_test = "./dataset/test_original.npy"
    t_dataset_test = "./dataset/test_2.npy"
    o_dataset = "./dataset/train_original_tiny.npy"
    t_dataset = "./dataset/train_2_tiny.npy"
    dataset = LoadImages(t_dataset, o_dataset)
    dataset_test = LoadImages(t_dataset_test, o_dataset_test)
    trainsize = int(len(dataset) * 80 / 100)
    valsize = int(len(dataset) - trainsize)
    train_set, val_set = torch.utils.data.random_split(dataset, lengths=[trainsize, valsize])
    batch_size = 16
    valloader = DataLoader(val_set, batch_size, shuffle=True)
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)
    model = DecryptionModel()
    nb_epoch = 3
    log_frequency = 10
    learning_rate = 1e-4
    # model.load_state_dict(torch.load("./modeltrained/1A_Tiny_3epoch_model.pt"))
    train_optim(model, device, nb_epoch, log_frequency, learning_rate)
    # torch.save(model.state_dict(), './modeltrained/1A_Tiny_3epoch_model.pt')
    eval(model, device, testloader, batch_size, 10)
