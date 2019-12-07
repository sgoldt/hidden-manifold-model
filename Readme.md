Code for "Modelling the influence of data structure on learning in neural networks"
==================================

Here we provide the code used to run all the experiments of our recent paper on
the hidden manifold model [1]. It is written in C++ and uses the [Armadillo
library](http://arma.sourceforge.net) for linear algebra.

Contents
---------

1. libscmpp.h is a library containing utility functions to create and train
   two-layer neural networks.
2. hmm.cpp contains the experiments for independent students, with many
   options for control.
3. hmm_online.cpp performs simulations of online learing.
4. hmm_ode.cpp is an integrator for the ODE that describe online learning.

Compilation
-------

To compile locally, simply type
```
make hmm.exe hmm_online.exe hmm_ode.exe
``` 
This assumes that you have installed the [Armadillo
library](http://arma.sourceforge.net) on your machine.

Usage
-----

To see the options of each program, use the -h flag.

References 
----------

[1] S. Goldt, M. Mézard, F. Krzakala and L. Zdeborová: Modelling the influence of
data structure on learning in neural networks [arXiv:1909.11500](https://arxiv.org/abs/1909.11500)
