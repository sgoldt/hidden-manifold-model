/*
 * File:    hmm_ode.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 0.2
 *
 * Date:    December 2019
 *          March    2020
 */

#include <cmath>
#include <getopt.h>
#include <iostream>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

#include <armadillo>
#include <chrono>

#include "libscmpp.h"

using namespace std;
using namespace arma;

const int NUM_LOGPOINTS = 200;  // # of times we print something

const char * usage = R"USAGE(
This tool integrates the equations of motion that describe the generalisation
dynamics of two-layer neural networks in the hidden manifold problem.

usage: hmm_ode.exe [-h] [-M M] [-K K] [--lr LR]
                        [--dt DT] [--steps STEPS] [--quiet]


optional arguments:
  -h, -?                show this help message and exit
  --g G                 activation function for teacher and student;
                           0-> linear, 1->erf.
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  --delta               ratio of latent directions to input dimension, R/N.
  -l, --lr LR           learning rate
  -a, --steps STEPS     max. weight update steps in multiples of N
  --init INIT           weight initialisation:
                           1: large initial weights, with initial overlaps from --overlaps
                           2: small initial weights
  --prefix              file prefix to load initial conditions from
  --both                train both layers.
  --uniform A           make all of the teacher's second layer weights equal to
                          this value.
  --dt DT               integration time-step
  -r SEED, --seed SEED  random number generator seed. Default=0
  -q --quiet            be quiet and don't print order parameters to cout.
)USAGE";

mat get_Q(mat& Sigma, mat& W, double fu2, double ufu_2) {
  return (fu2 - ufu_2) * W + ufu_2 * Sigma;
}


/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b.
 */
void update_C2(mat& C2, mat& cov, int a, int b) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // return A * cov * A.t();
  
  C2(0, 0) = cov(a, a);
  C2(0, 1) = cov(a, b);
  C2(1, 0) = cov(b, a);
  C2(1, 1) = cov(b, b);
}

double J2_lin(mat& C) {
  return 1;
}

double J2_erf(mat& C) {
  return 2 / datum::pi /
      sqrt(1 + C(0, 0) + C(1, 1) - pow(C(0, 1), 2) + C(0, 0) * C(1, 1));
}

double I2_erf(mat& C) {
  return (2. / datum::pi * asin(C(0, 1)/(sqrt(1 + C(0, 0))*sqrt(1 + C(1, 1)))));
}

double I2_lin(mat& C) {
  return C(0, 1);
}

/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b,
 * and c.
 */
void update_C3(mat& C3, mat& cov, int a, int b, int c) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // A(2, c) = 1;
  // return A * cov * A.t();
  
  C3(0, 0) = cov(a, a);
  C3(0, 1) = cov(a, b);
  C3(0, 2) = cov(a, c);
  C3(1, 0) = cov(b, a);
  C3(1, 1) = cov(b, b);
  C3(1, 2) = cov(b, c);
  C3(2, 0) = cov(c, a);
  C3(2, 1) = cov(c, b);
  C3(2, 2) = cov(c, c);
}

double I3_erf(mat& C) {
  double lambda3 = (1 + C(0, 0))*(1 + C(2, 2)) - pow(C(0, 2), 2);

  return (2. / datum::pi / sqrt(lambda3) *
          (C(1, 2)*(1 + C(0, 0)) - C(0, 1)*C(0, 2)) / (1 + C(0, 0)));
}

double I3_lin(mat& C) {
  return C(1, 2);
}


/**
 * Returns the projection of the given covariance matrix C to the d.o.f. a, b,
 * c, and d.
 */
void update_C4(mat& C4, mat& cov, int a, int b, int c, int d) {
  // The code below is a brute-force implementation of the following code:
  // mat A = mat(size(cov), fill::zeros);
  // A(0, a) = 1;
  // A(1, b) = 1;
  // A(2, c) = 1;
  // A(3, d) = 1;
  // return A * cov * A.t();

  C4(0, 0) = cov(a, a);
  C4(0, 1) = cov(a, b);
  C4(0, 2) = cov(a, c);
  C4(0, 3) = cov(a, d);
  C4(1, 0) = cov(b, a);
  C4(1, 1) = cov(b, b);
  C4(1, 2) = cov(b, c);
  C4(1, 3) = cov(b, d);
  C4(2, 0) = cov(c, a);
  C4(2, 1) = cov(c, b);
  C4(2, 2) = cov(c, c);
  C4(2, 3) = cov(c, d);
  C4(3, 0) = cov(d, a);
  C4(3, 1) = cov(d, b);
  C4(3, 2) = cov(d, c);
  C4(3, 3) = cov(d, d);
}

double I4_erf(mat& C) {
  double lambda4 = (1 + C(0, 0))*(1 + C(1, 1)) - pow(C(0, 1), 2);

  double lambda0 = (lambda4 * C(2, 3)
                    - C(1, 2) * C(1, 3) * (1 + C(0, 0))
                    - C(0, 2)*C(0, 3)*(1 + C(1, 1))
                    + C(0, 1)*C(0, 2)*C(1, 3)
                    + C(0, 1)*C(0, 3)*C(1, 2));
  double lambda1 = (lambda4 * (1 + C(2, 2))
                    - pow(C(1, 2), 2) * (1 + C(0, 0))
                    - pow(C(0, 2), 2) * (1 + C(1, 1))
                    + 2 * C(0, 1) * C(0, 2) * C(1, 2));
  double lambda2 = (lambda4 * (1 + C(3, 3))
                    - pow(C(1, 3), 2) * (1 + C(0, 0))
                    - pow(C(0, 3), 2) * (1 + C(1, 1))
                    + 2 * C(0, 1) * C(0, 3) * C(1, 3));

  return (4 / pow(datum::pi, 2) / sqrt(lambda4) *
          asin(lambda0 / sqrt(lambda1 * lambda2)));
}

double I4_lin(mat& C) {
  return C(2, 3);
}


/**
 * Performs an integration step and returns increments for Q and R.

 * Parameters:
 * -----------
 * duration:
 *     the time interval for which to propagate the system
 * dt :
 *     the length of a single integration step
 * t :
 *     time at the start of the propagation
 * Q : (K, K)
 *     student-student overlap
 * R : (K, M)
 *     student-teacher overlap
 * T : (M, M)
 *     teacher-teacher overlap
 * A : vec (M)
 *     hidden unit-to-output weights of the teacher
 * v : (K)
 *     hidden unit-to-output weights of the student
 * lr : scalar
 *     learning rate of the first layer
 * wd : scalar
 *     weight decay constant
 * sigma : double
 *     std. dev. of the teacher's output noise
*/
void propagate(double duration, double dt, double& time,
               mat& Sigma, mat& W, mat& Q, mat& R, mat& T, mat& tildeT,
               vec& A, vec& v, cube& sigma, cube& r, vec& rhos,
               double(*I2)(mat&), double(*I3)(mat&), double(*I4)(mat&),
               double delta, double lr, double fu2, double ufu, double ufu_2,
               bool both) {
  int K = R.n_rows;
  int M = R.n_cols;

  double propagation_time = 0;
  double zeta = fu2 - ufu_2;
  vec d = zeta * delta + ufu_2 * rhos;
    
  mat C = zeros<mat>(K + M, K + M);  // full covariance matrix C

  while(propagation_time < duration) {
    // update the full covariance matrix of all local fields
    C.submat(0, 0, K-1, K-1) = Q;
    C.submat(0, K, K-1, K+M-1) = R;
    C.submat(K, 0, K+M-1, K-1) = R.t();
    C.submat(K, K, K+M-1, K+M-1) = T;

    // integrate r
    cube dr = cube(size(r), fill::zeros);
    int m;
    #pragma omp parallel for collapse(2) private(m)
    for (int k = 0; k < K; k++) {
      for (m = 0; m < M; m++) {
        // reduced cov matrices for 3- and 4-point correlations
        mat C3 = zeros<mat>(3, 3);
        mat C4 = zeros<mat>(4, 4);

        vec rkm = (vec) r.tube(k, m);
        vec drkm = vec(size(rkm), fill::zeros);
        for (int j = 0; j < K; j++) {
          if (j == k)
            continue;
          double det = Q(j, j) * Q(k, k) - pow(Q(k, j), 2);

          // first line
          update_C3(C3, C, k, k, j);
          drkm -= d % rkm * v(k) * v(j) * Q(j, j) * I3(C3) / det;
          update_C3(C3, C, k, j, j);
          drkm += d % rkm * v(k) * v(j) * Q(k, j) * I3(C3) / det;

          // second line
          vec rjm = (vec) r.tube(j, m);
          update_C3(C3, C, k, j, j);
          drkm -= d % rjm * v(k) * v(j) * Q(k, k) * I3(C3) / det;
          update_C3(C3, C, k, k, j);
          drkm += d % rjm * v(k) * v(j) * Q(k, j) * I3(C3) / det;
        }

        // third line
        update_C3(C3, C, k, k, k);
        drkm -= d % rkm * v(k) * v(k) / Q(k, k) * I3(C3);

        for (int n = 0; n < M; n++) {
          double det = Q(k, k) * T(n, n) - pow(R(k, n), 2);

          // fourth line
          update_C3(C3, C, k, k, K + n);
          drkm += d % rkm * v(k) * A(n) * T(n, n) * I3(C3) / det;
          update_C3(C3, C, k, K + n, K + n);
          drkm -= d % rkm * v(k) * A(n) * R(k, n) * I3(C3) / det;

          // fifth line
          update_C3(C3, C, k, K + n, K + n);
          drkm += ufu * rhos * v(k) * A(n) * tildeT(n, m) * Q(k, k) * I3(C3) / det;
          update_C3(C3, C, k, k, K + n);
          drkm -= ufu * rhos * v(k) * A(n) * tildeT(n, m) * R(k, n) * I3(C3) / det;
        }

        dr.tube(k, m) = dt * lr / delta * drkm;
      } // end of m loop
    } // end of k loop

    // integrate sigma
    cube dsigma = cube(size(sigma));
    int num_elems = K  * (K + 1) / 2;  // number of elements in triu
    #pragma omp parallel for
    for (int idx = 0; idx < num_elems; idx++) {
      // map the index to a row and a column
      // algorithm courtesy of Z boson https://stackoverflow.com/a/28483812
      int k = idx % (K + 1);
      int l = idx / (K + 1);
      if (k > l) {
        k = K - k;
        l = K  - l - 1;
      }
      
      // reduced cov matrices for 3- and 4-point correlations
      mat C3 = zeros<mat>(3, 3);
      mat C4 = zeros<mat>(4, 4);

      vec skl = (vec) sigma.tube(k, l);
      vec dskl = vec(size(skl), fill::zeros);
      for (int j = 0; j < K; j++) {
        if (j == k)
          continue;
        double det = Q(j, j) * Q(k, k) - pow(Q(k, j), 2);

        // first line
        update_C3(C3, C, k, k, j);
        dskl -= d % skl * v(k) * v(j) * Q(j, j) * I3(C3) / det;
        update_C3(C3, C, k, j, j);
        dskl += d % skl * v(k) * v(j) * Q(k, j) * I3(C3) / det;

        // second line
        vec sjl = (vec) sigma.tube(j, l);
        update_C3(C3, C, k, j, j);
        dskl -= d % sjl * v(k) * v(j) * Q(k, k) * I3(C3) / det;
        update_C3(C3, C, k, k, j);
        dskl += d % sjl * v(k) * v(j) * Q(k, j) * I3(C3) / det;
      }

      // third line
      update_C3(C3, C, k, k, k);
      dskl -= d % skl * v(k) * v(k) / Q(k, k) * I3(C3);

      for (int n = 0; n < M; n++) {
        double det = Q(k, k) * T(n, n) - pow(R(k, n), 2);

        // fourth line
        update_C3(C3, C, k, k, K + n);
        dskl += d % skl * v(k) * A(n) * T(n, n) * I3(C3) / det;
        update_C3(C3, C, k, K + n, K + n);
        dskl -= d % skl * v(k) * A(n) * R(k, n) * I3(C3) / det;

        // fifth line
        vec rln = (vec) r.tube(l, n);
        update_C3(C3, C, k, K + n, K + n);
        dskl += ufu * rhos % rln * v(k) * A(n) * Q(k, k) * I3(C3) / det;
        update_C3(C3, C, k, k, K + n);
        dskl -= ufu * rhos % rln * v(k) * A(n) * R(k, n) * I3(C3) / det;
      }

      // sigma: now starting with the conjugate terms
      vec slk = (vec) sigma.tube(l, k);

      for (int j = 0; j < K; j++) {
        if (j == l)
          continue;
        double det = Q(j, j) * Q(l, l) - pow(Q(l, j), 2);

        // first line
        update_C3(C3, C, l, l, j);
        dskl -= d % slk * v(l) * v(j) * Q(j, j) * I3(C3) / det;
        update_C3(C3, C, l, j, j);
        dskl += d % slk * v(l) * v(j) * Q(l, j) * I3(C3) / det;

        // second line
        vec sjk = (vec) sigma.tube(j, k);
        update_C3(C3, C, l, j, j);
        dskl -= d % sjk * v(l) * v(j) * Q(l, l) * I3(C3) / det;
        update_C3(C3, C, l, l, j);
        dskl += d % sjk * v(l) * v(j) * Q(l, j) * I3(C3) / det;
      }

      // third line
      update_C3(C3, C, l, l, l);
      dskl -= d % slk * v(l) * v(l) / Q(l, l) * I3(C3);

      for (int n = 0; n < M; n++) {
        double det = Q(l, l) * T(n, n) - pow(R(l, n), 2);

        // fourth line
        update_C3(C3, C, l, l, K + n);
        dskl += d % slk * v(l) * A(n) * T(n, n) * I3(C3) / det;
        update_C3(C3, C, l, K + n, K + n);
        dskl -= d % slk * v(l) * A(n) * R(l, n) * I3(C3) / det;

        // fifth line
        vec rkn = (vec) r.tube(k, n);
        update_C3(C3, C, l, K + n, K + n);
        dskl += ufu * rhos % rkn * v(l) * A(n) * Q(l, l) * I3(C3) / det;
        update_C3(C3, C, l, l, K + n);
        dskl -= ufu * rhos % rkn * v(l) * A(n) * R(l, n) * I3(C3) / det;
      }

      // multiply the whole of dskl with the linear learning rate
      dskl *= dt * lr / delta;

      // now for the quadratic part!
      vec dtlr2rho = dt * pow(lr, 2) * (zeta * rhos + ufu_2 / delta * pow(rhos, 2));
      for (int n = 0; n < M; n++) {  // teacher
        for (int m = 0; m < M; m++) {  // teacher
          update_C4(C4, C, k, l, K + n, K + m);
          dskl += dtlr2rho * v(k) * v(l) * A(n) * A(m) * I4(C4);
        }
      }

      for (int j = 0; j < K; j++) {
        for (int m = 0; m < M; m++) {
          update_C4(C4, C, k, l, j, K + m);
          dskl -= dtlr2rho * 2 * v(k) * v(l) * v(j) * A(m) * I4(C4);
        }
      }

      for (int j = 0; j < K; j++) {  // student
        for (int a = 0; a < K; a++) {  // student
          update_C4(C4, C, k, l, j, a);
          dskl += dtlr2rho * v(k) * v(l) * v(j) * v(a) * I4(C4);
        }
      }

      dsigma.tube(k, l) = dskl;
      if (k != l) {
        dsigma.tube(l, k) = dskl;
      }

      // --------------------
      // integrate W
      // --------------------

      // terms proportional to the learning rate
      for (int n = 0; n < M; n++) { // teacher
        update_C3(C3, C, k, l, K + n);
        W(k, l) += dt * lr * v(k) * A(n) * I3(C3);
        update_C3(C3, C, l, k, K + n);
        W(k, l) += dt * lr * v(l) * A(n) * I3(C3);
      }

      for (int j = 0; j < K; j++) { // student
        update_C3(C3, C, k, l, j);
        W(k, l) -= dt * lr * v(k) * v(j) * I3(C3);
        update_C3(C3, C, l, k, j);
        W(k, l) -= dt * lr * v(l) * v(j) * I3(C3);
      }
        
      // W terms quadratic in the learning rate
      double dtlr2 = dt * fu2 * pow(lr, 2);
      for (int n = 0; n < M; n++) {  // teacher
        for (int m = 0; m < M; m++) {  // teacher
          update_C4(C4, C, k, l, K + n, K + m);
          W(k, l) += dtlr2 * v(k) * v(l) * A(n) * A(m) * I4(C4);
        }
      }

      for (int j = 0; j < K; j++) {
        for (int m = 0; m < M; m++) {
          update_C4(C4, C, k, l, j, K + m);
          W(k, l) -= dtlr2 * 2 * v(k) * v(l) * v(j) * A(m) * I4(C4);
        }
      }

      for (int j = 0; j < K; j++) {  // student
        for (int a = 0; a < K; a++) {  // student
          update_C4(C4, C, k, l, j, a);
          W(k, l) += dtlr2 * v(k) * v(l) * v(j) * v(a) * I4(C4);
        }
      }
    }
    W = symmatu(W);

    // integrate v
    if (both) {
      // reduced cov matrices for 2-point correlations
      mat C2 = zeros<mat>(2, 2);

      vec dv = vec(size(v), fill::zeros);
      for (int i = 0; i < K; i++) {  // student
        for (int k = 0; k < K; k++) {  // student
          update_C2(C2, C, i, k);
          dv(i) -= dt * lr * v(k) * I2(C2);
        }

        for (int n = 0; n < M; n++) { // teacher
          update_C2(C2, C, i, K + n);
          dv(i) += dt * lr * A(n) * I2(C2);
        }
      }

      v += dv;
    }

    // putting it all together to update the order parameters
    sigma += dsigma;
    Sigma = mean(sigma, 2);

    Q = get_Q(Sigma, W, fu2, ufu_2);

    r += dr;
    R = ufu * mean(r, 2);

    time += dt;
    propagation_time += dt;
  }
}


int main(int argc, char* argv[]) {
  // flags; false=0 and true=1
  int both = 0;   // train both layers
  int quiet = 0;  // don't print the order parameters to cout
  // other parameters
  int    f         = SIGN;  // manifold-inducing function
  int    g         = ERF;   // teacher and student activation function
  int    M         = 2;  // num of teacher's hidden units
  int    K         = 2;  // num of student's hidden units
  double delta     = 0.5;  // delta = R / N
  double lr        = 0.2;  // learning rate
  double dt        = 0.01;
  int    init      = INIT_LARGE; // initialisation
  string prefix;  // file name prefix to preload the weights
  double max_steps = 1000;  // max number of gradient updates / N
  int    seed      = 0;  // random number generator seed

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"both",       no_argument, &both,           1},
    {"quiet",      no_argument, &quiet,          1},
    {"f",       required_argument, 0, 'f'},
    {"g",       required_argument, 0, 'g'},
    {"M",       required_argument, 0, 'M'},
    {"K",       required_argument, 0, 'K'},
    {"delta",   required_argument, 0, 'z'},
    {"lr",      required_argument, 0, 'l'},
    {"dt",      required_argument, 0, 'd'},
    {"init",    required_argument, 0, 'i'},
    {"prefix",  required_argument, 0, 'p'},
    {"overlap", required_argument, 0, 'o'},
    {"steps",   required_argument, 0, 'a'},
    {"seed",    required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "g:M:K:z:l:d:i:f:o:a:r",
                    long_options, &option_index);

    /* Detect the end of the options. */
    if (c == -1) {
      break;
    }

    switch (c) {
      case 0:
        break;
      case 'f':
        f = atoi(optarg);
        break;
      case 'g':
        g = atoi(optarg);
        break;
      case 'M':
        M = atoi(optarg);
        break;
      case 'K':
        K = atoi(optarg);
        break;
      case 'z':
        delta = atof(optarg);
        break;
      case 'l':
        lr = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'p':  // initialisation of the weights
        prefix = string(optarg);
        break;
      case 'd':  // integration time-step
        dt = atof(optarg);
        break;
      case 'a':  // number of steps
        max_steps = atof(optarg);
        break;
      case 'r':
        seed = atoi(optarg);
        break;
      case 'h':  // intentional fall-through
      case '?':
        cout << usage << endl;
        return 0;
      default:
        abort ();
    }
  }

  // set the seed
  arma_rng::set_seed(seed);
  
  double fu2, ufu, ufu_2;
  switch (f) {
    case ERF:
      fu2 = 1. / 3;
      ufu = sqrt(1. / datum::pi);
      ufu_2 = 1. / datum::pi;
      break;
    case RELU:
      cerr << "Haven't implemented ReLU yet, since it's mean is not 0." << endl;
      break;
    case SIGN:
      fu2 = 1.;  // <f(u)^2>
      ufu = sqrt(2. / datum::pi);  // <u f(u)>
      ufu_2 = 2. / datum::pi;  // <u f(u)>^2
      break;
    default:
      cerr << "Haven't implemented this manifold-inducing function yet" << endl;
      return 0;
  }

  double (*J2_fun)(mat&);
  double (*I2_fun)(mat&);
  double (*I3_fun)(mat&);
  double (*I4_fun)(mat&);
  mat (*g_fun)(mat&);
  switch (g) {  // find the teacher's activation function
    case LINEAR:
      J2_fun = J2_lin;
      I2_fun = I2_lin;
      I3_fun = I3_lin;
      I4_fun = I4_lin;
      g_fun = g_lin;
      break;
    case ERF:
      J2_fun = J2_erf;
      I2_fun = I2_erf;
      I3_fun = I3_erf;
      I4_fun = I4_erf;
      g_fun = g_erf;
      break;
    default:
      cerr << "g has to be linear (g=" << LINEAR << ") or erf (g=" << ERF
           << ").\n will exit now!" << endl;
      return 1;
  }
  
  FILE* logfile;
  const char* g_name = activation_name(g);
  if (prefix.empty()) {
    char* log_fname;
    asprintf(&log_fname,
             "hmm_ode_sgn_%s_%s_%sdelta%g_M%d_K%d_lr%g_i%d_steps%g_dt%g_s%d.dat",
             g_name, g_name, (both ? "both_" : ""), delta, M, K, lr, init, max_steps, dt, seed);
    logfile = fopen(log_fname, "w");
  } else {
    string log_fname = prefix;
    log_fname.append("_ode.dat");
    logfile = fopen(log_fname.c_str(), "w");
  }

  ostringstream welcome;
  welcome << "# This is the HMM ODE integrator for two-layer NN" << endl
          << "# g1=g2=" << g_name << ", M=" << M << ", K=" << K
          << ", steps/N=" << max_steps << ", seed=" << seed << endl
          << "# delta=" << delta << ", lr=" << lr << ", dt=" << dt << endl;
  if (!prefix.empty()) {
    welcome << "# took initial conditions from simulation " << prefix << endl;
  }
  welcome << "# steps / N, eg, et, diff" << endl;
  string welcome_string = welcome.str();
  cout << welcome_string;
  
  fprintf(logfile, "%s", welcome_string.c_str());

  // "original" order parameters
  int N = 10000;
  int D = round(delta * N);
  mat Sigma = mat(K, K);
  cube sigma = cube(K, K, D, fill::zeros);
  mat W = mat(K, K);
  mat Q = mat(K, K);
  mat R = mat(K, M);
  cube r = cube(K, M, D, fill::zeros);
  mat T = mat(M, M);
  mat tildeT = mat(M, M);  // self-overlap of rotated teacher vectors
  vec A = ones(M);
  vec v = ones(K);
  vec rhos;  // vector with the eigenvalues rho
  if (!prefix.empty()) {
    prefix.append("_sigma0.dat");
    bool ok = sigma.load(prefix);
    prefix.replace(prefix.end()-11, prefix.end(), "_W0.dat");
    ok = ok && W.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_v0.dat");
    ok = ok && v.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_A0.dat");
    ok = ok && A.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_r0.dat");
    ok = ok && r.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_rhos.dat");
    ok = ok && rhos.load(prefix);
    prefix.replace(prefix.end()-9, prefix.end(), "_T0.dat");
    ok = ok && T.load(prefix);
    prefix.replace(prefix.end()-7, prefix.end(), "_tildeT0.dat");
    ok = ok && tildeT.load(prefix);
    if (!ok) {
      cerr << "Error loading initial conditions from files, will exit !" << endl;
      return 1;
    } else {
      cout << "# Loaded all initial conditions successfully." << endl;
    }
  } else if (init == INIT_LARGE or init == INIT_SMALL) {
    int N = 10000;
    double prefactor = (init == INIT_LARGE) ? 1 : 1e-3;
    int D = round(delta * N);
    mat w = prefactor *  randn<mat>(K, N);
    mat B = randn<mat>(M, D);
    mat F = randn<mat>(N, D);

    mat Omega = 1. / N * F.t() * F;
    mat psis;  // vector with the eigenvalues defined earlier
    eig_sym(rhos, psis, Omega);
    // make sure to normalise, orient evectors according to the note
    psis = sqrt(D) * psis.t();  

    mat S = 1. / sqrt(N) * w * F;
    mat B_tau = 1. / sqrt(D) * B * psis.t();
    mat Gamma_tau = 1. / sqrt(D) * S * psis.t();

    for (int k = 0; k < K; k++) {
      for (int l = 0; l < K; l++) {
        sigma.tube(k, l) = Gamma_tau.row(k) % Gamma_tau.row(l);
      }
      for (int n = 0; n < M; n++) {
        r.tube(k, n) = Gamma_tau.row(k) % B_tau.row(n);
      }
    }

    W = 1. / N * w * w.t();
    T = 1. / D * B * B.t();
    tildeT = 1. / D * B_tau * diagmat(rhos) * B_tau.t();
  } else {
    cerr << "--init must be 1 (random init) or 2 (file)."
         << endl << "Will exit now !" << endl;
    return 1;
  }
  Sigma = mean(sigma, 2);
  Q = get_Q(Sigma, W, fu2, ufu_2);
  R = ufu * mean(r, 2);
  
  cout << "Begin of initial conditions" << endl;
  Sigma.print("Sigma=");
  W.print("W=");
  Q.print("Q=");
  R.print("R=");
  T.print("T=");
  tildeT.print("\\tilde T=");
  cout << "End of initial conditions" << endl;
  
  // find printing times
  vec print_times = logspace<vec>(-1, log10(max_steps), NUM_LOGPOINTS);
  print_times(0) = 0;
  vec durations = diff(print_times);

  // print the status header
  std::ostringstream header;
  header << "0 time, 1 eg, 2 et, 3 diff, " << endl;
  int column = 4;
  for (int k = 0; k < K; k++) {
    for (int l = k; l < K; l++) {
      header << column << " Q(" << k << ", " << l << "), ";
      column++;
    }
  }
  for (int k = 0; k < K; k++) {
    for (int m = 0; m < M; m++) {
      header << column << " R(" << k << ", " << m << "), ";
      column++;
    }
  }
  for (int m = 0; m < M; m++) {
    for (int n = m; n < M; n++) {
      header << column << " T(" << m << ", " << n << "), ";
      column++;
    }
  }
  for (int m = 0; m < M; m++) {
    header << column << " A(" << m << "), ";
    column++;
  }
  for (int k = 0; k < K; k++) {
    header << column << " v(" << k << "), ";
    column++;
  }
  for (int k = 0; k < K; k++) {
    for (int l = k; l < K; l++) {
      header << column << " Sigma(" << k << ", " << l << "), ";
      column++;
    }
  }
  for (int k = 0; k < K; k++) {
    for (int l = k; l < K; l++) {
      header << column << " W(" << k << ", " << l << "), ";
      column++;
    }
  }
  std::string header_string = header.str();
  cout << header_string << endl;

  chrono::steady_clock::time_point begin = chrono::steady_clock::now();

  double t = 0;
  bool converged = false;
  for (double& duration : durations) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    // string msg = status(t, eg, datum::nan, datum::nan, datum::nan, datum::nan,
    //                     Q, R, T, A, v);
    std::ostringstream msg;
    msg << t << ", " << eg << ", " << datum::nan << ", " << datum::nan << ", ";
    
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        msg << Q(k, l) << ", ";
      }
    }
    for (int k = 0; k < K; k++) {
      for (int m = 0; m < M; m++) {
        msg << R(k, m) << ", ";
      }
    }
    for (int m = 0; m < M; m++) {
      for (int n = m; n < M; n++) {
        msg << T(m, n) << ", ";
      }
    }
    for (int m = 0; m < M; m++) {
      msg << A(m) << ", ";
    }
    for (int k = 0; k < K; k++) {
      msg << v(k) << ", ";
    }
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        msg << Sigma(k, l) << ", ";
      }
    }
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        msg << W(k, l) << ", ";
      }
    }
    std::string msg_str = msg.str();
    msg_str = msg_str.substr(0, msg_str.length() - 2);
    cout << msg_str << endl;
    fprintf(logfile, "%s\n", msg_str.c_str());
    fflush(logfile);

    if (eg < 1e-14 && t > 100) {
      converged = true;
      break;
    } else {
      propagate(duration, dt, t, Sigma, W, Q, R, T, tildeT, A, v, sigma, r, rhos,
                I2_fun, I3_fun, I4_fun, delta, lr, fu2, ufu, ufu_2, both);
    }
  }
  if (!converged) {
    double eg = eg_analytical(Q, R, T, A, v, g_fun, g_fun);
    string msg = status(t, eg, datum::nan, datum::nan, datum::nan, datum::nan,
                        Q, R, T, A, v);
    cout << msg << endl;
    fprintf(logfile, "# %s\n", msg.c_str());
    fflush(logfile);
  }
  
  chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
  fprintf(logfile, "# Computation took %lld seconds\n",
          chrono::duration_cast<chrono::seconds>(end - begin).count());
  fclose(logfile);

  return 0;
}

