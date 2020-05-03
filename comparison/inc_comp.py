#!/usr/bin/env python3
#
# A program to check whether neural networks learn functions of increasing
# complexity
#
# Date: March 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>


from __future__ import print_function

import argparse

from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from twolayer import FlattenTransformation, TwoLayer, erfscaled


def binarise(x, groups):
    """
    Binarises the tensor, replacing every element in x that is also in group
    with 1, and all the others with  -1.
    """
    # return -1 + 2 * (x[..., None] == GROUPS).any(-1).float()
    return -1 + 2 * float(x in groups)


def normalise(x):
    """
    Normalises the tensor to have mean zero and unity variance.
    """
    x -= torch.mean(x)
    x /= torch.std(x)
    return x


def get_eg(student, loader, device):
    """
    Computes the training error of the given student.

    N.B. that this function does not change the state of the model, i.e. it
    does for example not call student.eval().
    """
    losses = []

    with torch.no_grad():
        for data, target in loader:
            target = target.unsqueeze(1).float()

            data = data.to(device)
            target = target.to(device)
            prediction = student(data)
            losses += [F.mse_loss(prediction, target)]

    return 0.5 * torch.mean(torch.tensor(losses))


def log(msg, logfile):
    print(msg)
    logfile.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50,
                        help="training epochs. Default=50.")
    parser.add_argument('-K', '--K', metavar='Ks', type=int, default=4,
                        help="size of the student's intermediate layers")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="learning constant (default 0.05)")
    parser.add_argument("--bs", type=int, default=128,
                        help="batch size (default 128)")
    parser.add_argument("-g", "--g", default="erf",
                        help="activation function: 'erf' or 'relu'")
    parser.add_argument('--dataset', default="mnist",
                        help="dataset: mnist, fmnist.")
    parser.add_argument('--device', "-d",
                        help="which device to run on: 'cuda:x' or 'cpu'.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Be quiet ! (no order parameters)")
    parser.add_argument("--normalise1", action="store_true",
                        help="normalise inputs to hidden nodes")
    parser.add_argument("--normalise2", action="store_true",
                        help="normalise inputs to output nodes")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # set up the loading of the dataset
    input_size = 0
    num_classes = 1  # because we're looking at MSE on odd-even discrimination
    groups = None
    if args.dataset == "mnist":
        ds = datasets.MNIST
        input_size = 784
        groups = [0, 2, 4, 6, 8]
    elif args.dataset == "fmnist":
        ds = datasets.FashionMNIST
        input_size = 784
        groups = [0, 2, 3, 5, 8]  # attempt at a tough division of the set
    elif args.dataset == "cifar10":
        ds = datasets.CIFAR10
        input_size = 3072
        groups = [0, 2, 4, 6, 8]
    else:
        raise ValueError("dataset code %s not recognised. Will exit now !" %
                         args.dataset)

    # load the data set
    transform = transforms.Compose([transforms.ToTensor(),
                                    FlattenTransformation(input_size),
                                    transforms.Lambda(normalise)])
    trainingset = ds(root='~/datasets', train=True, download=True,
                     transform=transform,
                     target_transform=(lambda x: -1 + 2 * float(x in groups)))
    testset = ds(root='~/datasets', train=False, download=True,
                 transform=transform,
                 target_transform=(lambda x: -1 + 2 * float(x in groups)))

    # kwargs = {"num_workers": 2, "pin_memory": True}
    kwargs = {"pin_memory": True}
    train_loader = torch.utils.data.DataLoader(trainingset, batch_size=args.bs,
                                               shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1000,
                                              shuffle=False, **kwargs)

    criterion = nn.MSELoss()
    g = (erfscaled if args.g == "erf" else F.relu)
    student = TwoLayer(g, input_size, args.K, num_classes, std0=1e-3,
                       normalise1=args.normalise1, normalise2=args.normalise2)
    student = student.to(device)

    optimiser = None
    if args.normalise1:
        # set up SGD with two different learning rates
        params = [{'params': student.fc1.parameters()},
                  {'params': student.fc2.parameters(),
                   'lr': args.lr / input_size}]
        optimiser = torch.optim.SGD(params, lr=args.lr)
    else:
        optimiser = torch.optim.SGD(student.parameters(), lr=args.lr)

    fname_root = ("inc_comp_%s_%s_K%d_lr%g_bs%d_s%d" %
                  (args.dataset, args.g,
                   args.K, args.lr, args.bs, args.seed))
    log_fname = fname_root + ".log"
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Training a two-layer net on %s\n" % (args.dataset)
    welcome += "# Training on %s \n" % str(device)
    welcome += "# time, epoch, e_g, avg loss"
    log(welcome, logfile)

    # find the steps at which to print
    total_time = args.epochs * len(train_loader.dataset) / input_size
    end = torch.log10(torch.tensor([1. * total_time])).item()
    times_to_print = list(torch.logspace(-1, end, steps=200))

    time = 0

    for epoch in range(args.epochs):
        student.train()

        total_losses = []
        loss = None

        for data, target in tqdm(train_loader):
            target = target.unsqueeze(1).float()
            data = data.to(device)
            target = target.to(device)

            optimiser.zero_grad()

            prediction = student(data)
            loss = criterion(prediction, target)

            loss.backward()
            optimiser.step()

            total_losses.append(loss.item())

            if time >= times_to_print[0].item() or time == 0:
                # compute test error and average loss
                student.eval()
                eg = get_eg(student, test_loader, device)
                avg_loss = 0.5 * torch.mean(torch.tensor(total_losses))

                # print
                msg = ("%g, %d, %g, %g, " % (time, epoch, eg, avg_loss))
                if not args.quiet:
                    Q = student.selfoverlap()
                    Qvals = [Q[i, j].item() for i in range(args.K)
                                            for j in range(i, args.K)]
                    msg += ", ".join(map(str, Qvals))
                    msg += ", "
                    v = student.fc2.weight.data
                    vvals = [v[i, j].item() for i in range(num_classes)
                                            for j in range(args.K)]
                    msg += ", ".join(map(str, vvals))

                log(msg, logfile)

                # reset things to  continue training
                student.train()
                total_losses = []
                if not time == 0:
                    while time >= times_to_print[0].item():
                        times_to_print.pop(0)

            time += data.shape[0] / input_size

    fname_model = fname_root + "_student.pt"
    torch.save(student.state_dict(), fname_model)


if __name__ == "__main__":
    main()
