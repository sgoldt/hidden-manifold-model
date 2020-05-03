#!/usr/bin/env python3
#
# Easy examples experiment
#
# Date: January 2020, March 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm import tqdm

from twolayer import FlattenTransformation, TwoLayer, identity

# root folder for the datasets
root = "./datasets"


def getDataset(dataset, randy):
    """
    Loads the dataset with the given identifier and returns training and test data set
    as well as the input dimension.
    """
    trainingset = None
    testset = None
    N = 0
    if dataset in ["mnist", "fmnist", "kmnist", "cifar10"]:
        num_channels = 3 if dataset == "cifar10" else 1
        N = 3072 if dataset == "cifar10" else 784
        if dataset == "mnist":
            ds = torchvision.datasets.MNIST
        elif dataset == "fmnist":
            ds = torchvision.datasets.FashionMNIST
        elif dataset == "kmnist":
            ds = torchvision.datasets.KMNIST
        elif dataset == "emnist":
            ds = torchvision.datasets.EMNIST
        else:
            ds = torchvision.datasets.CIFAR10

        # transform for the image datasets
        mean = (0.5,) * num_channels
        std = (0.5,) * num_channels
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                FlattenTransformation(N),
            ]
        )

        # load the images!
        trainingset = ds(
            root="~/datasets", train=True, download=True, transform=transform
        )
        testset = ds(root="~/datasets", train=False, download=True, transform=transform)

        if randy:
            randperm = torch.load("randperm%d.pt" % len(trainingset))
            trainingset.targets = torch.tensor(trainingset.targets)[randperm]
    elif dataset in ["teacher0", "teacher1", "teacher2"]:
        N = 3072
        train_xs, train_ys, test_xs, test_ys = [None, None, None, None]
        parameters = "%srelu_N3072_M8192_O10_s0" % (
            "D35_" if dataset in ["teacher1", "teacher2"] else ""
        )
        train_xs = torch.load("./%s/%s_train_%s_xs.pth" % (root, dataset, parameters))
        labels = "ys_rand" if randy else "ys"
        fname = "./%s/%s_train_%s_%s.pth" % (root, dataset, parameters, labels)
        train_ys = torch.load(fname).long()
        test_xs = torch.load("./%s/%s_test_%s_xs.pth" % (root, dataset, parameters))
        fname = "./%s/%s_test_%s_ys.pth" % (root, dataset, parameters)
        test_ys = torch.load(fname).long()

        trainingset = torch.utils.data.TensorDataset(train_xs, train_ys)
        testset = torch.utils.data.TensorDataset(test_xs, test_ys)
    else:
        raise ValueError("Invalid dataset given")

    return trainingset, testset, N


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-K",
        "--K",
        metavar="K",
        type=int,
        default=4096,
        help="size of the student's intermediate layer",
    )
    parser.add_argument(
        "--device", "-d", help="which device to run on: 'cuda:x' or 'cpu'."
    )
    parser.add_argument("--dataset", default="cifar10", help="which dataset to run.")
    parser.add_argument(
        "--randy", help="Randomise the training labels.", action="store_true"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning constant")
    # parser.add_argument("--mom", type=float, default=0.9,
    #                     help="momentum")
    # parser.add_argument("--wd", type=float, default=0,
    #                     help="weight decay constant.\nDefault=0")
    # parser.add_argument("--damp", type=float, default=0,
    #                     help="dampening.\nDefault=0")
    parser.add_argument("--bs", type=int, default=32, help="batch size. Default=32")
    parser.add_argument(
        "--epochs", type=int, default=1, help="training epochs. Default=1."
    )
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="random number generator seed. Default=0",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    N = 0
    num_cats = 10
    trainingset, testset, N = getDataset(args.dataset, args.randy)

    loaders = {
        "train": DataLoader(
            trainingset, batch_size=args.bs, shuffle=True, num_workers=2
        )
    }

    # create a student network
    # NOTE: it's absolutely crucial to put an identity in the second layer,
    # if you put a non-linearity, you'll be able to reproduce results on CIFAR10,
    # but not on synthetic data !
    gs = (F.relu, identity)
    student = TwoLayer(gs, N, args.K, num_cats)
    student.to(device)
    print(student)
    for param in student.parameters():
        print(param)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(student.parameters(), lr=args.lr)

    # prepare the log file
    dataset_desc = "%s%s" % (args.dataset, "_randy" if args.randy else "")
    filename_info = "%s_K%d_lr%g_bs%d_e%d_s%d" % (
        dataset_desc,
        args.K,
        args.lr,
        args.bs,
        args.epochs,
        args.seed,
    )
    log_fname = "easy_" + filename_info + ".log"
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Finding easy examples for memorisation in %s%s\n" % (
        args.dataset,
        " with random labels" if args.randy else "",
    )
    welcome += "# N=%d, K=%d, num_cat=%d\n" % (N, args.K, num_cats)
    welcome += "# lr=%g, bs=%d, epochs=%d, seed=%d\n" % (
        args.lr,
        args.bs,
        args.epochs,
        args.seed,
    )
    welcome += "# Using device: %s" % device
    log(welcome, logfile)

    for epoch in range(args.epochs):
        num_batches = 0
        running_loss = 0

        for data in tqdm(loaders["train"]):
            # get the inputs; data is a list of [inputs, labels]
            # labels is a list with numeric class labels, i.e. [6, 7, 9, 0]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            if num_batches == 0:
                # debugging: output element to make sure we are looking at
                # the training examples in random order at each different seed
                log(
                    "# train: i=%d, input(0, 11)=%g (should change)"
                    % (num_batches, inputs[0, 400]),
                    logfile,
                )

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # raw, un-normalised scores as required for CrossEntropyLoss
            outputs = student(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1
        log("# %d, loss: %.3f " % (epoch + 1, running_loss / num_batches), logfile)

    # re-initialise thne loader for the training data set, but without shuffling
    # to make the experiments comparable
    loaders = {
        "train": DataLoader(
            trainingset, batch_size=args.bs, shuffle=False, num_workers=2
        ),
        "test": DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2),
    }

    accuracy = {"train": 0, "test": 0}
    votes = torch.zeros(10)  # count how many times the student predicted a given class
    for mode, loader in loaders.items():
        num_correct_pred = torch.zeros(len(loader.dataset))
        with torch.no_grad():
            num_batches = 0
            for data in tqdm(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                if num_batches == 0:
                    # debugging: output element to make sure we are looking at
                    # the examples in the same order, independent of the seed
                    log(
                        "# %s: i=%d, input(0, 11)=%g (should NOT change)"
                        % (mode, num_batches, inputs[0, 11]),
                        logfile,
                    )

                outputs = student(inputs)
                _, predicted = torch.max(outputs.data, 1)

                for i in range(10):
                    votes[i] += (predicted == i).sum()

                start = num_batches * args.bs
                num_correct_pred[start : (start + args.bs)] = predicted == labels

                num_batches += 1

        votes /= len(loader.dataset)
        print("votes for ", mode, ": ", votes)
        accuracy[mode] = 100 * num_correct_pred.sum().item() / num_correct_pred.size(0)

        correct_fname = ("correct_%s_" % mode) + filename_info + ".pth"
        torch.save(num_correct_pred, correct_fname)

    # store the student
    # student_fname = "student_" + filename_info + ".pth"
    # torch.save(student.state_dict(), student_fname)

    status = "%d, %g, %g" % (args.epochs, accuracy["test"], accuracy["train"])
    log(status, logfile)
    logfile.close()
    print("Bye-bye")


if __name__ == "__main__":
    main()
