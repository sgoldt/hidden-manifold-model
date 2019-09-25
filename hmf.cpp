/**
 * File:    hmf.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 1.0
 *
 * Date:    September 2019
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

const int J_DEFAULT = 10;
const int N_DEFAULT = 784;  // dimension of the basis vectors
const double TS_DEFAULT = 1;  // number of samples in multiples of the teacher's dimension

// four different scenarios to look at
const int ISING = 0;  // Gaussian inputs
const int TEACHER_ON_INPUT = 1;  // structured inputs, teacher acting on X
const int TEACHER_ON_COEFF = 2;  // structured inputs, teacher acting on rand. coeff
const int MNIST = 3;  // MNIST inputs with odd-even labels
const int TEACHER_FROM_MNIST = 4;  // i.i.d. Gaussian inputs, teacher pre-trained on MNIST acting on X


const int NUM_TEST_SAMPLES = 50000;

const char * usage = R"USAGE(
Training independent committee machines on the same dataset.

usage: hmf.exe [-h] [--g G] [-N N] [-J J] [-M M] [-K K] [--lr LR]
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
)USAGE";


/**
 * Generates P inputs with their labels according to the given scenario.
 *
 * Parameters:
 * -----------
 * P, N:
 *     number of samples / of dimensions
 * F : (J, N)
 *     random basis
 * f_fun:
 *     non-linearity that creates inputs: X = f_fun(C F)
 * g_fun:
 *     activation function of the network
 * B : (M, N) or (M, J)
 *     teacher weights
 */
mat get_samples(mat& xs, mat& ys, int scenario, int P, int N, mat& F,
                mat (*f_fun)(mat&), mat (*g_fun)(mat&), mat& B, vec& A,
                mat (*output)(mat&, vec&, mat&, mat (*)(mat&))) {
  xs = mat(P, N);
  ys = mat(P, 1);

  switch (scenario) {
    case ISING: // intentional fall-through
    case TEACHER_FROM_MNIST: {
      mat raw = randn<mat>(P, N);
      xs = f_fun(raw);
      xs -= mean(mean(xs));
      xs /= as_scalar(stddev(vectorise(xs)));
      ys = output(B, A, xs, g_fun);
      break;
    }
    case TEACHER_ON_COEFF:  // intentional fall-through
    case TEACHER_ON_INPUT: {
      mat C = randn<mat>(P, F.n_rows); // random coefficients
      mat raw = C * F / sqrt(F.n_rows);
      xs = f_fun(raw);  // (P, N)
      xs -= mean(mean(xs));
      xs /= as_scalar(stddev(vectorise(xs)));
      
      if (scenario == TEACHER_ON_INPUT) {
        ys = output(B, A, xs, g_fun);
      } else if (scenario == TEACHER_ON_COEFF) {
        ys = output(B, A, C, g_fun);
      }
      break;
    }
  }

  return xs;
}


int main(int argc, char* argv[]) {
  // flags; false=0 and true=1
  int both         = 0;  // train both layers
  int calculate_et = 0;  // calculate the training error
  int random       = 0;  // randomise the labels
  int do_classify  = 0;  // calculate classification errors, too
  int normalise    = 0;  // normalise the output of SCMs
  int mix          = 0;  // flip the sign of half of teacher's 2nd-layer weights
  int quiet        = 0;  // don't print the order parameters to cout
  double uniform   = 0;  // value of all weights in the teacher's second layer
  // other parameters
  int    f         = SIGN;  // data-generating non-linearity
  int    g         = ERF;  // teacher activation function
  int    scenario  = TEACHER_ON_INPUT;
  int    N         = N_DEFAULT;  // number of inputs
  int    M         = 2;  // num of teacher's hidden units
  int    K         = 2;  // num of student's hidden units
  int    J         = J_DEFAULT;  // num of basis vectors
  double lr        = 0.5;  // learning rate
  double ts        = TS_DEFAULT;  // size of the training set in multiples of N
  double max_steps = 10000;  // max number of gradient updates / N
  int    init      = 1;  // initialisation of student weights
  std::string teacher;   // name of file containing teacher weights
  int    seed      = 0;  // random number generator seed

  // parse command line options using getopt
  int c;

  static struct option long_options[] = {
    // for documentation of these options, see the definition of the
    // corresponding variables
    {"both",       no_argument, &both,           1},
    {"et",         no_argument, &calculate_et,   1},
    {"normalise",  no_argument, &normalise,      1},
    {"mix",        no_argument, &mix,            1},
    {"quiet",      no_argument, &quiet,          1},
    {"random",     no_argument, &random,         1},
    {"classify",   no_argument, &do_classify,    1},
    {"teacher",    required_argument, 0, 'z'},
    {"f",          required_argument, 0, 'f'},
    {"g",          required_argument, 0, 'g'},
    {"scenario",   required_argument, 0, 'm'},
    {"N",          required_argument, 0, 'N'},
    {"M",          required_argument, 0, 'M'},
    {"K",          required_argument, 0, 'K'},
    {"J",          required_argument, 0, 'J'},
    {"lr",         required_argument, 0, 'l'},
    {"ts",         required_argument, 0, 't'},
    {"init",       required_argument, 0, 'i'},
    {"uniform",    required_argument, 0, 'u'},
    {"steps",      required_argument, 0, 'a'},
    {"seed",       required_argument, 0, 'r'},
    {0, 0, 0, 0}
  };

  while (true) {
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "f:g:m:N:M:K:J:t:a:r:i:",
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
      case 'J':  // ambient dimension of the data space
        J = atoi(optarg);
        break;
      case 'l':  // learning rate
        lr = atof(optarg);
        break;
      case 'i':  // initialisation of the weights
        init = atoi(optarg);
        break;
      case 'z':  // pre-load teacher weights from file with this prefix
        teacher = std::string(optarg);
        break;
      case 'u':  // value of the second layer weights of the teacher
        uniform = atof(optarg);
        break;
      case 't':  // size of the training set
        ts = atof(optarg);
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

  // Initialise students straight away to guarantee that starting with the same
  // seed means starting with the same initial weights irrespective of the
  // scenario etc.
  mat w [2];
  mat dw [2];
  mat gradw [2];
  vec v [2];
  vec gradv [2];
  for (int i = 0; i < 2; i++) {
    w[i] = mat(K, N);
    v[i] = vec(K);
    switch (init) {
      case INIT_LARGE:
      case INIT_SMALL:
      case INIT_MIXED:
      case INIT_MIXED_NORMALISE: // intentional fall-through
        init_student_randomly(w[i], v[i], N, K, init, uniform, both, normalise, false);
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
  }

  // get the teacher
  mat B0 = mat();  // teacher input-to-hidden weights
  vec A0 = vec();  // teacher hidden-to-output weights
  bool success = false;
  if (scenario == ISING
      or scenario == TEACHER_ON_INPUT
      or scenario == TEACHER_ON_COEFF) {
    int dimension = (scenario == TEACHER_ON_COEFF) ? J : N;
    success = init_teacher_randomly(B0, A0, dimension, M, uniform, both, normalise, 0, mix);
  } else if (scenario == TEACHER_FROM_MNIST) {
    if (teacher.empty()) {
      cerr << "Have to provide teacher weights via --teacher for this scenario."
           << " Will exit now!" << endl;
      return 1;
    } else{
      teacher.append("_w.dat");
      success = B0.load(teacher);
      teacher.replace(teacher.end()-6, teacher.end(), "_v.dat");
      success = success && A0.load(teacher);
      M = B0.n_rows;
      N = B0.n_cols;
      J = 0;
    }
  } else if (scenario == MNIST) {
    // no need to initialise a teacher; will load MNIST data set next
    success = true;
  }
  if (!success) {
    cerr << "Failed to initialise the teacher correctly, will exit now. " << endl;
    return 1;
  }

  const int P = round(ts * N);
  if (P == 0 && scenario == MNIST) {
    cerr << "Cannot online learn MNIST. Will exit now ! " << endl;
    return 1;
  }

  mat (*f_fun)(mat&);
  switch (f) {  // find the teacher's activation function
    case LINEAR:
      f_fun = g_lin;
      break;
    case ERF:
      f_fun = g_erf;
      break;
    case RELU:
      f_fun = g_relu;
      break;
    case SIGN:
      f_fun = g_sign;
      break;
    case QUAD:
      f_fun = g_quad;
      break;
    default:
      cerr << "f has to be linear (f=" << LINEAR << "), erf (f=" << ERF <<
          "), ReLU (f=" << RELU << ") or sign (f=" << SIGN << ") or quad (f="
           << QUAD << "). " << endl;
      cerr << "Will exit now!" << endl;
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
    case QUAD:
      g_fun = g_quad;
      dgdx_fun = dgdx_quad;
      break;
    default:
      cerr << "g has to be linear (g=" << LINEAR << "), erf (g=" << ERF <<
          "), ReLU (g=" << RELU << ") or sign (g=" << SIGN << ") or quad (g="
           << QUAD << "). " << endl;
      cerr << "Will exit now!" << endl;
      return 1;
  }

  // find the teachers output function
  auto teacher_output = (do_classify ?
      static_cast< mat (*)(mat&, vec&, mat&, mat (*)(mat&))>(classify) :
      static_cast< mat (*)(mat&, vec&, mat&, mat (*)(mat&))>(phi));

  // Generating samples or at least prepare online generation of samples
  mat F = randn<mat>(J, N);  // random basis for scenarios 1, 2
  mat x_train;
  mat y_train;
  mat y_train_class;
  mat x_test;
  mat y_test;
  mat y_test_class;
  switch (scenario) {
    case ISING: 
    case TEACHER_ON_INPUT:  // intentional fall-through
    case TEACHER_ON_COEFF:
    case TEACHER_FROM_MNIST: {
      if (ts > 0) {
        get_samples(x_train, y_train,
                    scenario, P, N, F, f_fun, g_fun, B0, A0, teacher_output);
      }
      get_samples(x_test, y_test, scenario, NUM_TEST_SAMPLES, N, F,
                  f_fun, g_fun, B0, A0, teacher_output);
      y_train_class = classify(y_train);
      y_test_class = classify(y_test);
      if (scenario == ISING or scenario == TEACHER_FROM_MNIST) {
        J = 0;  // clean up variables for filenames
      }
      break;
    }
    case MNIST: {
      if (J > 10) {
        cerr << "Cannot have J > 10 for MNIST, will exit now!" << endl;
        return 1;
      }
      if (J % 2 != 0) {
        cerr << "Can only have even number of bases for MNIST, will exit now!" << endl;
        return 1;
      }
      M = 0; // there is no teacher in MNIST, so set M=0 for the filename.
      
      // choose digits to look at
      vec even = shuffle(regspace<vec>(0, 2, 8));
      vec odd = shuffle(regspace<vec>(1, 2, 9));
      vec digits;
      if (J < 10) {
        digits = vec(J);
        digits.head(round(J / 2)) = even.head(round(J / 2));
        digits.tail(round(J / 2)) = odd.head(round(J / 2));
      }

      // load MNIST
      bool loaded = load_mnist(x_train, y_train, digits, MNIST_TRAIN, true);
      loaded = loaded && load_mnist(x_test, y_test, digits, MNIST_TEST, true);
      if (!loaded) {
        cerr << "Problem loading MNIST training data" << endl;
        return 1;
      }
      N = x_train.n_cols;

      // sub-sample if possible
      if (ts > 0) {
        if (P > x_train.n_rows) {
          cerr << "Cannot have " << P << " training images when using "
               << J << " digits!" << endl;
          return 1;
        }
        uvec arange = regspace<uvec>(0, x_train.n_rows - 1);
        arange = shuffle(arange);  // slow... Armadillo needs a choice function!
        uvec indices = arange.head(P);
        x_train = x_train.rows(indices);
        y_train = y_train.rows(indices);
        y_train_class = y_train;
        y_test_class = y_test;
      } else {
        ts = round(x_train.n_rows / N);
      }

      break;
    }
    default:
      cerr << "did not recognise the scenario; will exit now!" << endl;
      return 1;
  }
  // white test data for student-student comparisons
  mat x_white = randn<mat>(size(x_test));
  mat y_white;
  mat y_white_class;
  if (B0.n_cols == N) {
    y_white = teacher_output(B0, A0, x_white, g_fun);
    y_white_class = classify(y_white);
  }

  if (random) {
    calculate_et = true;
    y_train = shuffle(y_train);
  }

  // find printing times
  vec steps = logspace<vec>(-1, log10(max_steps), 200);

  const char* f_name = scenario == MNIST ? "mnist" : activation_name(f);
  const char* g_name = activation_name(g);
  char* uniform_desc;
  asprintf(&uniform_desc, "u%g_", uniform);
  char* log_fname;
  asprintf(&log_fname, "hmf_%s_%s_%s_%sscenario%d%s%s_J%d_%s%sN%d_M%d_K%d_ts%g_lr%g_i%d_s%d.dat",
           f_name, g_name, (both ? "both" : "1st"), normalise ? "norm_" : "",
           scenario, do_classify ? "class" : "", random ? "r" : "", J, 
           (uniform > 0 ? uniform_desc : ""), (mix > 0 ? "mix_" : ""), N, M, K, ts, lr, init, seed);
  FILE* logfile = fopen(log_fname, "w");

  std::ostringstream welcome;
  welcome << "# This is the low-dimensional model simulator 0.2" << endl
          << "# Scenario " << scenario << " with f=" << f_name << ", g=" << g_name
          << (normalise ? " (normalised)" : "")
          << (random ? " and random labels " : "") << endl
          << "# J=" << J << ", N=" << N << ", M=" << M << ", K=" << K
          << ", P=" << P << ", lr=" << lr << endl;
  if (scenario == TEACHER_FROM_MNIST and !teacher.empty()) {
    welcome << "# Loaded teacher from " << teacher.c_str() << endl;
  }
  welcome << "# t, "
          << "(1) eg_mse w1, (2) eg_mse w2, (3) mse w1 vs w2 (struc), (4) mse w1 vs w2 (Gauss), (5) eg_mse_gauss w1 vs B, (6) eg_mse_gauss w2 vs B, (7) et_mse w1, (8) et_mse w2, (9) et_mse w1 vs w2"
          << "(10) eg_frac w1, (11) eg_frac w2, (12) frac w1 vs w2 (struc), (13) frac w1 vs w2 (Gauss), (14) eg_frac_gauss w1 vs B, (15) eg_frac_gauss w2 vs B, (16) et_frac w1, (17) et_frac w2, (18) et_frac w1 vs w2";
  if (!quiet) welcome << ", A0, v1";
  std::string welcome_string = welcome.str();
  cout << welcome_string << endl;
  fprintf(logfile, "%s\n", welcome_string.c_str());

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

      for (int i = 0; i < 2; i++) {
        // i=0: regression, i=1: classification
        auto student_output = (i == 0 ?
                               static_cast< mat (*)(mat&, vec&, mat&, mat (*)(mat&))>(phi) :
                               static_cast< mat (*)(mat&, vec&, mat&, mat (*)(mat&))>(classify));
        // how do we quantify the error?
        auto error = (i == 0 ?
                      static_cast< double (*)(mat&, mat&)>(mse_numerical) :
                      static_cast< double (*)(mat&, mat&)>(frac_error));

        mat y_pred1 = student_output(w[0], v[0], x_test, g_fun);
        mat y_pred2 = student_output(w[1], v[1], x_test, g_fun);
        mat y_pred1_white = student_output(w[0], v[0], x_white, g_fun);
        mat y_pred2_white = student_output(w[1], v[1], x_white, g_fun);

        // Generalisation w.r.t. teacher
        if (random) {
          msg << ", " << datum::nan << ", " << datum::nan;
        } else {
          double eg0 = error(y_pred1, i == 0 ? y_test : y_test_class); // w1 vs MNIST / B0 on struc. inputs
          double eg1 = error(y_pred2, i == 0 ? y_test : y_test_class); // w2 vs MNIST / B0 on struc. inputs
          msg << ", " << eg0 << ", " << eg1;
          // if ((eg0 + eg1) < 1e-14) {
          //   converged = true;
          // }
        }
        // Generalisation w.r.t. each other
        msg << ", "
            << error(y_pred1, y_pred2) << ", " // w1 vs w2 on struc. inputs
            << error(y_pred1_white, y_pred2_white); // and on white inputs

        // Generalisation w.r.t. teacher on white inputs
        if (!random && !B0.is_empty() && size(B0) == size(w[0])) {
          // can compare the student and the teacher on white inputs
          msg << ", " << error(y_pred1_white, i == 0 ? y_white : y_white_class)
              << ", " << error(y_pred2_white, i == 0 ? y_white : y_white_class);
        } else {
          msg << ", " << datum::nan << ", " << datum::nan;
        }

        // Training errors
        mat y_trainpred1 = student_output(w[0], v[0], x_train, g_fun);
        mat y_trainpred2 = student_output(w[1], v[1], x_train, g_fun);
        double et1 = error(y_trainpred1, i == 0 ? y_train : y_train_class);
        double et2 = error(y_trainpred2, i == 0 ? y_train : y_train_class);
        double et12 = error(y_trainpred1, y_trainpred2);
        msg << ", " << et1 << ", " << et2 << ", " << et12;
      }
      // add a comma at the end to make sure we end with ', ' before printing
      // (see below)
      msg << ", ";

      // printing second-layer weights as a check
      if (!quiet) {
        for (int m = 0; m < M; m++) {
          msg << A0(m) << ", ";
        }
        for (int k = 0; k < K; k++) {
          msg << v[0](k) << ", ";
        }
      }

      std::string msg_str = msg.str();
      msg_str = msg_str.substr(0, msg_str.length() - 2);
      cout << msg_str << endl;
      fprintf(logfile, "%s\n", msg_str.c_str());
      fflush(logfile);

      if (converged) break;
      step_print_next_idx++;
    }

    // train each student independently on the outputs
    for (int i = 0; i < 2; i++) {
      if (ts > 0) {
        int idx = randi(distr_param(0, x_train.n_rows-1));
        xs = x_train.row(idx);
        ys = y_train.row(idx);
      } else { // online learning
        // first the inputs
        get_samples(xs, ys,
                    scenario, 1, N, F, f_fun, g_fun, B0, A0, teacher_output);
      }
      
      update_gradients(gradw[i], gradv[i], w[i], v[i], xs, ys, g_fun, dgdx_fun, both);
      w[i] += lr / sqrt(N) * gradw[i];
      if (both) {
        v[i] += lr / N * gradv[i];
      }
    }
  }

  fclose(logfile);

  return 0;
}
