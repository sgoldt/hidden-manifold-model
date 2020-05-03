#!/usr/bin/env python3
#
# Generate classification datasets using a teacher network
#
# Date: January 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

from __future__ import print_function

import argparse

import torch
import torch.nn.functional as F

from twolayer import TwoLayer, erfscaled

# root folder for the datasets
root = "./"


def log(logfile, msg):
    print(msg)
    logfile.write(str(msg) + "\n")


def main():
    # read command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--g', metavar='g', default="relu",
                        help="activation function: 'erf' or 'relu' (default)")
    parser.add_argument('-N', '--N', metavar='N', type=int, default=3072,
                        help="input dimension")
    parser.add_argument('-D', '--D', metavar='D', type=int, default=35,
                        help="latent dimension for teacher1, teacher2")
    parser.add_argument('-M', '--M', metavar='M', type=int, default=8192,
                        help="size of the student's intermediate layer")
    parser.add_argument('--mean', default=0.5,
                        help="input mean")
    parser.add_argument('--std', default=0.5,
                        help="input std")
    parser.add_argument('--device', "-d",
                        help="which device to run on: 'cuda:x' or 'cpu'.")
    parser.add_argument('--scenario', default="teacher0",
                        help="which scenario to run.")
    parser.add_argument("--bs", type=int, default=256,
                        help="batch size. Default=4")
    parser.add_argument("--train", metavar='P', type=int, default=50000,
                        help="Number of training samples. Default=50 000.")
    parser.add_argument("--test", metavar='P^*', type=int, default=10000,
                        help="Number of training samples. Default=10 000.")
    parser.add_argument("-s", "--seed", type=int, default=0,
                        help="random number generator seed. Default=0")
    args = parser.parse_args()

    (scenario, g, D, N, M, bs, seed) = \
        (args.scenario, args.g, args.D, args.N, args.M, args.bs, args.seed)
    (num_train_samples, num_test_samples) = (args.train, args.test)
    num_cats = 10  # number of categories

    D_desc = "" if (scenario == "teacher0") else ("D%d_" % D)
    log_fname = ("sampling_%s_%s_%sN%d_M%d_s%d.log" %
                 (scenario, args.g, D_desc, N, M, seed))
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Sampling from a %s teacher with Gaussian weights for classification %s\n" % (g, scenario)
    welcome += ("# N=3072, M=%d, num_cat=10\n" % M)
    welcome += ("# bs=%d, num of samples %d (train) and %d (test), seed=%d\n" %
                (bs, num_train_samples, num_test_samples, seed))
    welcome += ("# Inputs have mean=%g, std=%g" % (args.mean, args.std))
    log(logfile, welcome)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    msg = ("# Using device: %s" % device)
    log(logfile, msg)

    torch.manual_seed(seed)

    # First generate the feature matrix to guarantee that it is always the same
    # for a given seed irrespective of teacher size.
    features = None
    if scenario in ["teacher1", "teacher2"]:
        features = torch.randn(N, D, device=device)
        torch.save(features, ("%s/features_D%d_N%d_M%d_O%d_s%d.pth" %
                              (root, D, N, M, num_cats, seed)))

    # create the teacher from which to sample, and check that a valid scenario
    # was given
    teacher = None
    normalise1 = True
    normalise2 = True
    std0 = 1
    g = (F.relu if g == "relu" else erfscaled)
    if scenario in ["teacher0", "teacher1"]:
        teacher = TwoLayer(g, N, M, num_cats, std0=std0,
                           normalise1=normalise1, normalise2=normalise2)
    elif scenario == "teacher2":
        teacher = TwoLayer(g, D, M, num_cats, std0=std0,
                           normalise1=normalise1, normalise2=normalise2)
    else:
        raise ValueError("Invalid scenario given")
    teacher.freeze()
    teacher.to(device)
    torch.save(teacher.state_dict(),
               ("%s/%s_%s_N%d_M%d_O%d_s%d.pth" %
                (root, scenario, args.g, teacher.N, M, num_cats, seed)))

    for mode in ["train", "test"]:
        log(logfile, "Generating the %sing data set" % mode)
        num_samples = num_train_samples if mode == "train" else num_test_samples
        xs = torch.zeros(num_samples + num_cats * bs, N)
        cs = torch.zeros(num_samples + num_cats * bs, D)
        ys = torch.zeros(num_samples + num_cats * bs, 1)

        converged = torch.tensor([False] * num_cats)
        num_xs_per_class = torch.zeros(num_cats)
        idx_samples = 0

        iteration = 0
        xs_raw, cs_raw, outputs = None, None, None
        while not torch.all(converged):
            if scenario == "teacher0":
                xs_raw = args.mean + args.std * torch.randn(bs, N, device=device)
                outputs = teacher(xs_raw)
            elif scenario in ["teacher1", "teacher2"]:
                cs_raw = torch.randn(bs, D, device=device)
                xs_raw = torch.sign(cs_raw.mm(features.T))
                xs_raw /= torch.sqrt(torch.var(xs_raw)) / args.std
                xs_raw -= torch.mean(xs_raw) - args.mean
                outputs = teacher(xs_raw if scenario == "teacher1" else cs_raw)
            _, ys_raw = torch.max(outputs.data, 1)

            for cat in range(num_cats):
                if not converged[cat]:
                    indicators = (ys_raw == cat)
                    num_new_samples = indicators.sum().item()
                    xs[idx_samples:idx_samples + num_new_samples] = xs_raw[indicators]
                    ys[idx_samples:idx_samples + num_new_samples, 0] = ys_raw[indicators]
                    if cs_raw is not None:
                        cs[idx_samples:idx_samples + num_new_samples] = cs_raw[indicators]
                    num_xs_per_class[cat] += num_new_samples
                    idx_samples += num_new_samples

            converged = num_xs_per_class >= (num_samples / num_cats)
            iteration += 1
            if (iteration % 1000 == 0):
                log(logfile, "iter %d" % iteration)
                log(logfile, num_xs_per_class)
                log(logfile, converged)

        log(logfile, num_xs_per_class)
        log(logfile, converged)

        # Done ! Now choose num_samples inputs in random order
        perm = torch.randperm(num_samples)
        xs = xs[perm]
        ys = torch.squeeze(ys[perm])

        # and save them
        torch.save(xs, ("%s/%s_%s_%s%s_N%d_M%d_O%d_s%d_xs.pth" %
                        (root, scenario, mode, D_desc, args.g, N, M, num_cats, seed)))
        torch.save(ys, ("%s/%s_%s_%s%s_N%d_M%d_O%d_s%d_ys.pth" %
                        (root, scenario, mode, D_desc, args.g, N, M, num_cats, seed)))
        if scenario in ["teacher1", "teacher2"]:
            cs = cs[perm]
            torch.save(cs, ("%s/%s_%s_%s%s_N%d_M%d_O%d_s%d_cs.pth" %
                            (root, scenario, mode, D_desc, args.g, N, M, num_cats, seed)))

    logfile.close()
    print("Bye-bye")


if __name__ == '__main__':
    main()
