/*
 * File:    lowdim_online.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 0.2
 *
 * Date:    February 2019
 */

#include <cmath>
#include <getopt.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

// #define ARMA_NO_DEBUG
#include <armadillo>

using namespace arma;

#include "libscmpp.h"

const int D_DEFAULT = 10;
const int N_DEFAULT = 784;  // dimension of the basis vectors

// four different scenarios to look at
const int ISING = 0;  // Gaussian inputs
const int TEACHER_ON_INPUT = 1;  // structured inputs, teacher acting on X
const int TEACHER_ON_COEFF = 2;  // structured inputs, teacher acting on rand. coeff
const int MNIST = 3;  // MNIST inputs with odd-even labels
const int GC = 5;  // gaussian conjugate model

// constants for the Gaussian conjugate model
const double mu_0 = 0.;
const double mu_1 = 1. / sqrt(datum::pi);
const double mu_star = 1. / 3. - 1. / datum::pi;

const int NUM_TEST_SAMPLES = 10000;

const char * usage = R"USAGE(
Online learning of structured datasets by committee machines.

usage: lowdim_online.exe [-h] [--g G] [-N N] [-D D] [-M M] [-K K] [--lr LR]
                     [--ts TS] [--classify] [--steps STEPS] [--uniform A]
                     [--both] [--normalise] [--quiet] [-s SEED]


optional arguments:
  -h, -?                show this help message and exit
  -s, --scenario        Training scenarios (default is 2):
                          0: Ising inputs,
                          1: structured data, teacher acts on inputs
                          2: structured data, teacher acts on rand. coeff,
                          4: Gaussian conjugate model
  --f F                 data generating non-linearity: X = f(CF)
                          0-> linear, 1->erf, 2->relu, 3->sgn (Default: 3).
  --g G                 activation function for teacher and student;
                          0-> linear, 1->erf, 2->relu, 3->sgn (not implemented).
  --both                train both layers of the student network.
  --normalise           divides 2nd layer weights by M and K for teacher and
                          student, resp. Overwritten by --both for the student
                          (2nd layer weights of the student are initialised
                           according to --init in that case).
  --uniform A           make all of the teacher's second layer weights equal to
                          this value. If the second layer of the student is not
                          trained, the second-layer output weights of the student
                          are also set to this value.
  -N, --N N             input dimension
  -M, --M M             number of hidden units in the teacher network
  -K, --K K             number of hidden units in the student network
  -D, --D D             number of basis vectors
  -l, --lr LR           learning rate
  --steps STEPS         max. weight update steps in multiples of N
  -r SEED, --seed SEED  random number generator seed. Default=0
  --store               store initial overlap and final weight matrices.
  --fourier             Use a DCT matrix as a feature fector.
  -q --quiet            be quiet and don't print order parameters to cout.
  --dummy               dummy command that doesn't do anything but helps with parallel.
)USAGE";


int main(int argc, char* argv[]) {
  // flags; false=0 and true=1
  int both         = 0;  // train both layers
  int normalise    = 0;  // normalise the output of SCMs
  int store        = 0;  // store initial weights etc.
  int quiet        = 0;  // don't print the order parameters to cout
  int debug        = 0;  // useful for various debugging messages
  int fourier      = 0;  // useful for various debugging messages
  int dummy        = 0;  // dummy parameter
  double uniform   = 0;  // value of all weights in the teacher's second layer
  // other parameters
  int    f         = SIGN;  // data-generating non-linearity
  int    g         = ERF;  // teacher activation function
  int    scenario  = TEACHER_ON_COEFF;
  int    N         = N_DEFAULT;  // number of inputs
  int    M         = 2;  // num of teacher's hidden units
  int    K         = 2;  // num of student's hidden units
  int    D         = D_DEFAULT;  // num of basis vectors
  double lr        = 0.2;  // learning rate
  double max_steps = 10000;  // max number of gradient updates / N
  int    init      = 1;  // initialisation of student weights
  int    seed      = 0;  // random number generator seed

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"both",       no_argument, &both,           1},
    {"normalise",  no_argument, &normalise,      1},
    {"store",      no_argument, &store,          1},
    {"quiet",      no_argument, &quiet,          1},
    {"debug",      no_argument, &debug,          1},
    {"fourier",    no_argument, &fourier,        1},
    {"dummy",      no_argument, &dummy,          1},
    {"f",          required_argument, 0, 'f'},
    {"g",          required_argument, 0, 'g'},
    {"scenario",   required_argument, 0, 'm'},
    {"N",          required_argument, 0, 'N'},
    {"M",          required_argument, 0, 'M'},
    {"K",          required_argument, 0, 'K'},
    {"D",          required_argument, 0, 'D'},
    {"lr",         required_argument, 0, 'l'},
    {"init",       required_argument, 0, 'i'},
    {"uniform",    required_argument, 0, 'u'},
    {"steps",      required_argument, 0, 'a'},
    {"seed",       required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "f:g:m:N:M:K:D:t:a:s:r:i:",
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
      case 'm':
        scenario = atoi(optarg);
        break;
      case 'N':  // input dimension
        N = atoi(optarg);
        break;
      case 'M':  // number of teacher hidden units
        M = atoi(optarg);
        break;
      case 'K':  // number of student hidden units
        K = atoi(optarg);
        break;
      case 'D':  // ambient dimension of the data space
        D = atoi(optarg);
        break;
      case 'l':  // learning rate
        lr = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'u':  // value of the second layer weights of the teacher
        uniform = atof(optarg);
        break;
      case 'a':  // number of steps in multiples of N
        max_steps = atof(optarg);
        break;
      case 'r':  // random number generator seed
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

  // Generating samples or at least prepare online generation of samples
  mat F;
  if (fourier) {
    D = 1023;
    N = 1023;
    // F = mat(N, N, fill::zeros);

    // for (int i = 0; i < N; i++) {
    //   for (int r = 0; r < N; r++) {
    //     F(i, r) = cos(datum::pi / N * (i + 0.5) * r);
    //   }
    // }
    bool ok = F.load("/Users/goldt/Research/lowdim/src/hadamard.csv", csv_ascii);
    if (ok) {
      cout << "Managed to load Hadamard matrix" << endl;
    } else {
      cerr << "Problem loading Hadamard matrix" << endl;
    }
    F *= sqrt(N);
  } else {
    F = randn<mat>(N, D);  // random basis for scenarios 1, 2
  }

  // Initialise students straight away to guarantee that starting with the same
  // seed means starting with the same initial weights irrespective of the
  // scenario etc.
  mat w = mat(K, N);
  mat dw = mat(size(w));
  mat gradw = mat(size(w));
  vec v = vec(K);
  vec gradv = vec(K);
  switch (init) {
    case INIT_LARGE:
    case INIT_SMALL:
    case INIT_MIXED:
    case INIT_MIXED_NORMALISE: // intentional fall-through
      init_student_randomly(w, v, N, K, init, uniform, both, normalise, false);
      break;
    case INIT_INFORMED:
      cerr << "Not implemented yet. Will exit now !" << endl;
      return 1;
    case INIT_DENOISE:
      cerr << "Not implemented yet. Will exit now !" << endl;
      return 1;
    default:
      cerr << "Init must be within 1-6. Will exit now." << endl;
      return 1;
  }

  // get the teacher if required
  mat B = mat();  // teacher input-to-hidden weights
  vec A = vec();  // teacher hidden-to-output weights
  int dimension = ((scenario == TEACHER_ON_COEFF) or (scenario == GC)) ? D : N;
  init_teacher_randomly(B, A, dimension, M, uniform, both, normalise);

  mat (*f_fun)(mat&);
  double ufu = 0;
  double fu2 = 0;
  double ufu_2 = 0;
  switch (f) {  // find the teacher's activation function
    case LINEAR:
      f_fun = g_lin;
      break;
    case ERF:
      f_fun = g_erf;
      fu2 = 1. / 3;
      ufu = sqrt(1. / datum::pi);
      ufu_2 = 1. / datum::pi;
      break;
    case RELU:
      f_fun = g_relu;
      fu2 = 1. / 2;
      ufu = 1. / 2;
      ufu_2 = 1. / 4;
      break;
    case SIGN:
      f_fun = g_sign;
      fu2 = 1.;  // <f(u)^2>
      ufu = sqrt(2. / datum::pi);  // <u f(u)>
      ufu_2 = 2. / datum::pi;  // <u f(u)>^2
      break;
    default:
      cerr << "f has to be linear (g=" << LINEAR << "), erf (g=" << ERF <<
          "), ReLU (g=" << RELU << ") or sign (f=" << SIGN << ")." << endl;
      cerr << "will exit now!" << endl;
      return 1;
  }

  mat (*g_fun)(mat&);
  mat (*dgdx_fun)(mat&);
  switch (g) {  // find the teacher's activation function
    case LINEAR:
      g_fun = g_lin;
      dgdx_fun = dgdx_lin;
      break;
    case ERF:
      g_fun = g_erf;
      dgdx_fun = dgdx_erf;
      break;
    case RELU:
      g_fun = g_relu;
      dgdx_fun = dgdx_relu;
      break;
    default:
      cerr << "g1 has to be linear (g=" << LINEAR << "), erf (g=" << ERF <<
          ") or ReLU (g=" << RELU << ")." << endl;
      cerr << "will exit now!" << endl;
      return 1;
  }
  
  mat test_cs = mat(NUM_TEST_SAMPLES, D);  // coefficients for scenarios 1, 2
  mat test_xs = mat(NUM_TEST_SAMPLES, N);
  mat test_ys = mat(NUM_TEST_SAMPLES, 1);

  switch (scenario) {
    case ISING: {
      test_xs = randn<mat>(NUM_TEST_SAMPLES, N);
      test_ys = phi(B, A, test_xs, g_fun);
      D = 0;
      break;
    }
    case TEACHER_ON_INPUT:  // intentional fall-through
    case TEACHER_ON_COEFF: {
      test_cs = randn<mat>(NUM_TEST_SAMPLES, D);
      mat test_xs_raw = test_cs * F.t() / sqrt(D);
      test_xs = f_fun(test_xs_raw);
      if (scenario == TEACHER_ON_INPUT) {
        test_ys = phi(B, A, test_xs, g_fun);
      } else { // scenario == TEACHER_ON_INPUT
        test_ys = phi(B, A, test_cs, g_fun);
      }
      break;
    }
    case GC: {
      test_cs = randn<mat>(NUM_TEST_SAMPLES, D);
      mat test_xs_raw = test_cs * F.t() / sqrt(D);
      test_xs = mu_0 + mu_1 * test_xs_raw +  sqrt(mu_star) * randn<mat>(size(test_xs));
      test_ys = phi(B, A, test_cs, g_fun);
      break;
    }
    case MNIST:
      cerr << "did not recognise the scenario; will exit now!" << endl;
      return 1;
  }

  // find printing times
  vec steps = logspace<vec>(-2, log10(max_steps), 200);

  const char* f_name = activation_name(f);
  const char* g_name = activation_name(g);
  char* uniform_desc;
  asprintf(&uniform_desc, "u%g_", uniform);
  char* log_fname;
  asprintf(&log_fname, "hmm_online_%s_%s_%s_%sscenario%d_%s%sN%d_D%d_M%d_K%d_lr%g_i%d_s%d.dat",
           f_name, g_name, (both ? "both" : "1st"), normalise ? "norm_" : "",
           scenario, (fourier ? "fourier_" : ""), (uniform > 0 ? uniform_desc : ""),
           N, D, M, K, lr, init, seed);
  FILE* logfile = fopen(log_fname, "w");

  std::ostringstream welcome;
  welcome << "# Online learning from low-dimensional inputs, model" << endl
          << "# Scenario " << scenario << " with f=" << f_name << ", g=" << g_name
          << (normalise ? " (normalised)" : "") << endl
          << "# D=" << D << ", N=" << N << ", M=" << M << ", K=" << K
          << ", lr=" << lr << endl << "# t, eg, diff";
  if (!quiet) welcome << ", eg guess, Q, R, T, A, v";
  std::string welcome_string = welcome.str();
  cout << welcome_string << endl;
  fprintf(logfile, "%s\n", welcome_string.c_str());

  if (store) {
    if (scenario == TEACHER_ON_COEFF) {
      // sample the densities r and sigma
      cube sigma = cube(K, K, D, fill::zeros);
      cube r = cube(K, M, D, fill::zeros);

      mat Omega = 1. / N * F.t() * F;
      vec rhos;
      mat psis;
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

      mat W = 1. / N * w * w.t();
      mat T = 1. / D * B * B.t();
      mat tildeT = 1. / D * B_tau * diagmat(rhos) * B_tau.t();
      
      std::string fname = std::string(log_fname);
      // r
      fname.replace(fname.end()-4, fname.end(), "_r0.dat");
      r.save(fname, arma_binary);
      // sigma
      fname.replace(fname.end()-7, fname.end(), "_sigma0.dat");
      sigma.save(fname, arma_binary);
      // rhos
      fname.replace(fname.end()-11, fname.end(), "_rhos.dat");
      rhos.save(fname, arma_binary);
      mat W0 = w * w.t() / N;
      fname.replace(fname.end()-9, fname.end(), "_W0.dat");
      W0.save(fname, arma_binary);
      mat T0 = 1. / D * B * B.t();
      fname.replace(fname.end()-7, fname.end(), "_T0.dat");
      T0.save(fname, arma_binary);
      mat tildeT0 = 1. / D * B_tau * diagmat(rhos) * B_tau.t();
      fname.replace(fname.end()-7, fname.end(), "_tildeT0.dat");
      tildeT0.save(fname, arma_binary);

      // check whether we get the order parameters back by integrating their
      // respective densities
      if (debug) {
        mat S = 1. / sqrt(N) * w * F;
        mat Sigma = 1. / D * Gamma_tau * Gamma_tau.t();
        mat R = ufu / D * S * B.t();
        mat R_int = ufu * mean(r, 2);
        mat Sigma_int = mean(sigma, 2);
        R.print("actual R=");
        R_int.print("integrated R=");
        Sigma.print("actual Sigma=");
        Sigma_int.print("integrated Sigma=");
        return 0;
      }
    }
  }

  int step_print_next_idx = 0;
  mat xs = mat(1, N);
  mat ys = mat(1, 1);
  double dstep = 1. / N;
  
  for (double step = 0; step < max_steps; step += dstep) {
    if (step > steps(step_print_next_idx) || step == 0.) {
      bool converged = false;
      // EVALUATION
      std::ostringstream msg;
      msg << step;

      mat pred_ys = phi(w, v, test_xs, g_fun);
      double eg_numerical = mse_numerical(pred_ys, test_ys);
      double diff = step > 0 ? as_scalar(sqrt(mean(mean(pow(dw, 2))))) : 0;
      msg << ", " << eg_numerical << ", " << diff << ", ";
      if (eg_numerical < 1e-14) {
        converged = true;
      }

      // printing order parameters and second-layer weights as a check
      if (!quiet) {
        mat Q = mat(K, K);
        mat R = mat(K, M);
        mat T = mat(M, M);
        mat W = mat(K, K);       // only for scenario 2
        mat Sigma = mat(K, K);   // only for scenario 2
        
        if ((scenario == TEACHER_ON_COEFF) or (scenario == GC)) {
          mat S = 1. / sqrt(N) * w * F;
          W = 1. / N * w * w.t();
          Sigma = 1. / D * S * S.t();
          Q = (fu2 - ufu_2) * W + ufu_2 * Sigma;
          R = ufu / D * S * B.t();
          T = 1. / D * B * B.t();
        } else {
          Q = 1. / N * w * w.t();
          R = 1. / N * w * B.t();
          T = 1. / N * B * B.t();
        }

        double eg_ana = eg_analytical(Q, R, T, A, v, g_erf, g_erf);
        msg << eg_ana << ", ";

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
        if ((scenario == TEACHER_ON_COEFF) or (scenario == GC)) {
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
        }
      }

      std::string msg_str = msg.str();
      msg_str = msg_str.substr(0, msg_str.length() - 2);
      cout << msg_str << endl;
      fprintf(logfile, "%s\n", msg_str.c_str());
      fflush(logfile);

      if (converged)
        break;
      step_print_next_idx++;
    }

    // train
    // first get the sample
    switch (scenario) {
      case ISING: {
        xs = randn<mat>(1, N);
        ys = phi(B, A, xs, g_fun);
        break;
      }
      case TEACHER_ON_INPUT:  // intentional fall-through
      case TEACHER_ON_COEFF: {
        mat cs = randn<mat>(1, D);
        mat xs_pre = cs * F.t() / sqrt(D);
        xs = f_fun(xs_pre);
        if (scenario == TEACHER_ON_INPUT) {
          ys = phi(B, A, xs, g_fun);
        } else {
          ys = phi(B, A, cs, g_fun);
        }
        break;
      }
      case GC: {
        mat cs = randn<mat>(1, D);
        mat xs_pre = cs * F.t() / sqrt(D);
        xs = mu_0 + mu_1 * xs_pre +  sqrt(mu_star) * randn<mat>(size(xs_pre));
        ys = phi(B, A, cs, g_fun);
        break;
      }        
      default:
        cerr << "Did not recognise scenario, will exit now." << endl;
        return 1;
    }
    
    update_gradients(gradw, gradv, w, v, xs, ys, g_fun, dgdx_fun, both);

    dw = lr / sqrt(N) * gradw;
    w += dw;
    if (both) {
      v += lr / N * gradv;
    }
  }

  fclose(logfile);

  return 0;
}
