Code for "Modelling the influence of data structure on learning in neural networks"
==================================

Here we provide the code used to run all the experiments of our recent paper on
the hidden manifold model [1]. There are two parts:

1. A simulator for training two-layer neural networks trained on the hidden
manifold using online learning. It is written in C++ and uses the [Armadillo
library](http://arma.sourceforge.net) for linear algebra.
2. A set of python scripts to run the memorisation experiments presented in
   Sec. V. They are implemented in Python using the
   [pyTorch](http://pytorch.org/) library.


Compilation of the C++ code
-------

To compile locally, simply type
```
make hmm_online.exe hmm_ode.exe
``` 
This assumes that you have installed the [Armadillo
library](http://arma.sourceforge.net) on your machine.

Usage
-----

To see the options of each program, use the -h flag.

To run the unit tests of the Python code, go the memorisation directory and
simply type
```
nose2
``` 

References
----------

[1] S. Goldt, M. Mézard, F. Krzakala and L. Zdeborová: Modelling the influence of
data structure on learning in neural networks [arXiv:1909.11500](https://arxiv.org/abs/1909.11500)
