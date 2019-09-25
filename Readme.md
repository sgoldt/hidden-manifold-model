Code for "Modelling the influence of data structure on learning in neural networks"
==================================

Here we provide the code used to run all the experiments of our recent paper
analysing the influence of data structure on learning in neural networks [1]. It
is written in C++ and uses the [Armadillo library](http://arma.sourceforge.net)
for linear algebra.

Contents
---------

We split the code into two parts:

1. libscmpp.h is a library containing utility functions to create and train
   two-layer neural networks.
2. lowdim.cpp contains the actual experiments. It offers many parameters and
   flags to control experiments.

Compilation
-------

To compile locally, simply type
```
g++ --std=c++11 -I. lowdim.cpp -o lowdim.exe -O3 -larmadillo
``` 
using the compiler of your choice. This assumes that you have installed the [Armadillo
library](http://arma.sourceforge.net) on your machine.

Usage
-----

```
lowdim.exe [-h] [--g G] [-N N] [-J J] [-M M] [-K K] [--lr LR]
                     [--ts TS] [--classify] [--steps STEPS] [--uniform A]
                     [--both] [--normalise] [--quiet] [-s SEED]


optional arguments:
  -h, -?                show this help message and exit
  -s, --scenario        Training scenarios:
                          0: Ising inputs,
                          1: structured data, teacher acts on inputs (default),
                          2: structured data, teacher acts on rand. coeff,
                          3: use MNIST inputs w/ MNIST odd-even labels.
                          4: i.i.d. Gaussian inputs, teacher pre-trained on MNIST
  --f F                 data generating non-linearity: X = f(CF)
                          0-> linear, 1->erf, 2->relu, 3->sgn (Default: 3).
  --g G                 activation function for teacher and student;
                          0-> linear, 1->erf, 2->relu, 3->sgn (not implemented).
  --teacher, -z  PREFIX For scenario 4, load weights for teacher and student
                            from files with the given prefix.
  --both                train both layers of the student network.
  --uniform A           make all of the teacher's second layer weights equal to
                          this value. If the second layer of the student is not
                          trained, the second-layer output weights of the student
                          are also set to this value.
  --normalise           divides 2nd layer weights by M and K for teacher and
                          student, resp. Overwritten by --both for the student
                          (2nd layer weights of the student are initialised
                           according to --init in that case).
  -N, --N N             input dimension
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  -J, --J J             number of basis vectors
  -l, --lr LR           learning rate
  --ts TS               Training set's size in multiples of N. Default=1.
                          In any scenario but MNIST, ts=0 means online learning.
                          If using MNIST and ts=0, use maximum number of MNIST
                          training images possible.
  --et                  calculate the training error, too.
  --mix                 changes the sign of half of the teacher's second-layer
                          weights.
  --classify            teacher output is +- 1.
  --random              Randomise the labels.
  --steps STEPS         max. weight update steps in multiples of N
  -r SEED, --seed SEED  random number generator seed. Default=0
  -q --quiet            be quiet and don't print order parameters to cout.
```


References 
----------

[1] S. Goldt, M. Mézard, F. Krzakala and L. Zdeborová: Modelling the influence of
data structure on learning in neural networks (arXiv identifier to follow)
