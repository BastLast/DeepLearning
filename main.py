import time

import imageio as imageio
import numpy as np
import sio as sio
import torch
from torch.autograd import Variable

from torch.utils.data.dataloader import DataLoader
from Model2 import CustomModel
from DataLoader import LoadImages


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR divided by 2 every 3 epochs"""
    lr = optimizer.param_groups[0]["lr"]
    if epoch % 3 == 1:
        if epoch > 1:
            lr = optimizer.param_groups[0]["lr"] / 2

    return lr

def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr

def train_optim(model, device, epochs, log_frequency, learning_rate=1e-4):
    model.to(device)  # we make sure the model is on the proper device

    loss_fn = torch.nn.L1Loss()
    loss_fn.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        lr = adjust_learning_rate(optimizer, epoch) # adjust the learning rate. Decreasing.
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
                    "epoch: {:03d}, batch: {:03d}, loss: {:.3f}, time: {:.3f}".format(epoch + 1, batch_id + 1, loss.item(),
                                                                                      time.time() - start))

            optimizer.zero_grad()  # clear the gradient before backward
            loss.backward()  # update the gradient

            optimizer.step()  # update the model parameters using the gradient

        # Model evaluation after each step computing the accuracy
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            avg_psnr_predicted = 0.0
            avg_psnr_noisy = 0.0
            avg_elapsed_time = 0.0
            ct = 0.0
            for batch_id, batch in enumerate(testloader):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                start_time = time.time()
                out_pred = model(images)
                elapsed_time = time.time() - start_time
                avg_elapsed_time += elapsed_time

                psnr_predicted = output_psnr_mse(labels, output)
                avg_psnr_predicted += psnr_predicted
                ct += 1
                output = output * 255.
                output = np.uint8(np.round(output))

                test_name = "originname" + str(batch_id) + '.png'  # .mat -> .png
                imageio.imwrite("./result/result_images/" + test_name, output)  # save result images

                print(100 * ct / (testloader.__len__()), "percent done")

        avg_psnr_predicted = avg_psnr_predicted / ct
        avg_psnr_noisy = avg_psnr_noisy / ct

        print("PSNR_noisy=", avg_psnr_noisy)
        print("PSNR_predicted=", avg_psnr_predicted)


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    o_dataset = "./dataset/train_original_tiny.npy"
    t_dataset = "./dataset/train_1A_tiny.npy"
    dataset = LoadImages(t_dataset, o_dataset)
    batch_size = 16
    #dataloader = DataLoader(dataset, batch_size, shuffle=True)
    trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = CustomModel()
    nb_epoch = 1
    log_frequency = 10
    learning_rate = 1e-4
    train_optim(model, device, nb_epoch, log_frequency, learning_rate)