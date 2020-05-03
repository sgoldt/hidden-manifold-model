/**
 * File:    libscmpp.cpp
 *
 * Author:  Sebastian Goldt <goldt.sebastian@gmail.com>
 *
 * Version: 1.0
 *
 * Date:    September 2019
 */

#ifndef LIBSCMPP
#define LIBSCMPP

#include <cmath>
#include <getopt.h>
#include <stdexcept>
#include <string.h>
#include <unistd.h>

// #define ARMA_NO_DEBUG
#include <armadillo>
#include <chrono>
using namespace arma;

// MATHEMATICAL CONSTANTS
const double ONE_OVER_SQRT2 = 1.0/datum::sqrt2;
const double SQRT_2_OVER_PI = datum::sqrt2/sqrt(datum::pi);

// codes for activation functions
const int LINEAR = 0;
const int ERF = 1;
const int RELU = 2;
const int SIGN = 3;
const int QUAD = 4;

// weight initialisations
const int INIT_LARGE = 1;
const int INIT_SMALL = 2;
const int INIT_INFORMED = 3;
const int INIT_DENOISE = 4;
const int INIT_MIXED = 5;
const int INIT_MIXED_NORMALISE = 6;
const int INIT_NATI = 7;
const int INIT_NATI_MF = 8;

// MNIST data sets
const int MNIST_TRAIN = 1;
const int MNIST_TEST = 2;


vec v_empty = vec();

/**
 * A wrapper around Armadillos trapz function that performs numerical
 * integration for all the vectors in the given field w.r.t. the given x.
 *
 * Parameters:
 * -----------
 * x : vec
 *     the spacing of x with respect to which the integral is computed
 * f : cube(r, c, s)
 *     cube where the integral is done over the last dimension
 *
 * Returns:
 * --------
 * result : mat (r, c)
 *     result of the trapezoidal integrals along z
 */
mat trapz(vec& x, cube& cu) {
  mat result = zeros(cu.n_rows, cu.n_cols);

  for (int i = 0; i < cu.n_rows; i++) {
    for (int k = 0; k < cu.n_cols; k++) {
      result(i, k) = as_scalar(trapz(x, (vec) cu.tube(i, k)));
    }
  }

  return result;
}


/**
 * Returns the moving average of the given vector x with window-size w.
 */
vec moving_average(vec& x, int w) {
  vec window = vec(w, fill::ones);
  vec avg = 1. / w * conv(x, window, "same");
  return avg;
}


/**
 * Returns a random rotation matrix, drawn from the Haar distribution
 * (the only uniform distribution on SO(n)).
 *
 * The algorithm is described in the paper Stewart, G.W., "The efficient
 * generation of random orthogonal matrices with an application to condition
 * estimators", SIAM Journal on Numerical Analysis, 17(3), pp. 403-409, 1980.
 * For more information see
 * https://en.wikipedia.org/wiki/Orthogonal_matrix#Randomization
 *
 * This implementation is a translation from the SciPy code, to be found at
 * https://github.com/scipy/scipy/blob/v1.2.1/scipy/stats/_multivariate.py,
 * which in turn is a wrapping of wrapping the random_rot code from the MDP
 * Toolkit, https://github.com/mdp-toolkit/mdp-toolkit
 */
mat special_ortho_group(int N) {
  mat H = eye<mat>(N, N);
  vec D = ones<vec>(N, 1);  // vec = colvec, i.e. dense matrix with one column
  for (int n = 1; n < N; n++) {
    vec x = randn<vec>(N-n+1);
    D(n-1) = x(0) >= 0 ? 1 : -1;
    x(0) -= D(n-1) * as_scalar(sqrt(sum(pow(x, 2))));
    // Householder transformation
    mat Hx = eye<mat>(N-n+1, N-n+1) - 2.* (x * x.t()) / as_scalar(sum(pow(x, 2)));
    mat asdf = eye<mat>(N, N);
    asdf.submat(n-1, n-1, N-1, N-1) = Hx;
    H = H * asdf;
  }
  // fix the last sign s.t. the determinant is 1
  D(D.n_elem - 1) = pow(-1, 1 - (N % 2)) * prod(D);
  H = diagmat(D) * H;

  return H;
}

/**
 * Prints a status update with the generalisation error and elements of Q and
 * R.
 *
 * Parameters:
 * -----------
 * t :
 *     time
 * eg : scalar
 *     generalisation error
 * et : scalar
 *     training error
 * eg_frac : scalar
 *     fractional generalisation error
 * et_frac : scalar
 *     fractional testing error
 * diff : scalar
 *     mean absolute change in weights
 * Q, R, T:
 *     order parameters
 * quiet : bool
 *     if True, output reduced information
 */
std::string status(double t, double eg, double et, double eg_frac,
                   double et_frac, double diff,
                   mat& Q, mat& R, mat&T, vec& A, vec& v, bool quiet=false) {
  std::ostringstream msg;

  msg << t << ", " << eg << ", " << et << ", " << eg_frac << ", "
      << et_frac << ", " << diff << ", ";

  if (!quiet) {
    int M = R.n_cols;
    int K = Q.n_rows;

    // print elements of Q
    for (int k = 0; k < K; k++) {
      for (int l = k; l < K; l++) {
        msg << Q(k, l) << ", ";
      }
    }
    if (!R.is_empty()) {
      // print elements of R
      for (int k = 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          msg << R(k, m) << ", ";
        }
      }
    }

    if(!A.is_empty()) {
      //print elements of A
      for (int n = 0; n < M; n++) {
        msg << A(n) << ", ";
      }
    }
    for (int i = 0; i < K; i++) {
      msg << v(i) << ", ";
    }
  }

  std::string msg_str = msg.str();
  return msg_str.substr(0, msg_str.length() - 2);
}


const char* activation_name(int g) {
  switch (g) {
    case LINEAR:
      return "lin";
    case ERF:
      return "erf";
    case RELU:
      return "relu";
    case SIGN:
      return "sgn";
    case QUAD:
      return "quad";
    default:
      return "";
  }
}


/**
 * Returns the fraction of entries where these two arrays do not have the same
 * entry.
 *
 * Parameters:
 * -----------
 * mat a, b : 
 *     the two matrices, which each must be of the size (N, 1)
 */
double frac_error(mat& a, mat& b) {
  if (a.n_elem != b.n_elem) {
    throw std::invalid_argument("vectors a and b must have the same length.");
  }
     
  bool polar = (a.min() < 0) || (b.min() < 0);

  a /= a.max();
  b /= b.max();

  if (polar) {  // ie encoded with -1, 1
    return 1 - 0.5 * mean(mean((a % b) + 1));  // % is the element-wise product
  } else {  // ie encoded with 0, const
    return mean(mean(abs(a - b)));
  }
}


/**
 * Randomises each label in the matrix a with probability ``p_randomise``.
 *
 * Parameters:
 * -----------
 * a : (N, 1)
 *     an (N, 1) matrix of labels
 * p_randomise :
 *     the probability that any label is replaced by a randomly drawn one.
 */
void randomise(mat& a, const double p_randomise) {
  const double max_value = a.max();

  a.transform([p_randomise, max_value] (double val) {
      if (randu() < p_randomise) {
        return max_value * (randu() > 0.5 ? 1. : -1.);
      } else {
        return val;
      }
    });
}

/**
 * Element-wise application of g(x) to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_lin(mat& x) {
  return x;
}


/**
 * Element-wise application of the derivative of Erf(x/sqrt(2)) to the matrix x
 *
 * @param  x  some matrix
 */
mat dgdx_lin(mat& x) {
  return ones(size(x));
}

/**
 * Element-wise application of g(x)=x^2 to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_quad(mat& x) {
  return pow(x, 2);
}

/**
 * Element-wise application of g(x)=x^2 to the matrix x.
 *
 * @param  x  some matrix
 */
mat dgdx_quad(mat& x) {
  return 2 * x;
}

/**
 * Element-wise application of erf(x/sqrt(2)) to the matrix x.
 *
 * @param  x  some matrix
 */
mat g_erf(mat& x) {
  return erf(ONE_OVER_SQRT2 * x);
}

/**
 * Element-wise application of the derivative of Erf(x/sqrt(2)) to the matrix x
 *
 * @param  x  some matrix
 */
mat dgdx_erf(mat& x) {
  return SQRT_2_OVER_PI*exp(-0.5*pow(x, 2));
}

/**
 * Rectified linear unit activation function.
 *
 * @param  x  some matrix
 */
mat g_relu(mat& x) {
  double max = x.max();
  return max > 0 ? clamp(x, 0, max) : x.fill(0);
}

/**
 * Derivative of the rectified linear unit.
 *
 * @param  x  some matrix
 */
mat dgdx_relu(mat& x) {
  mat dgdx = zeros<mat>(size(x));
  dgdx.elem(find(x > 0)).ones();
  return dgdx;
}

/**
 * Rectified linear unit activation function.
 *
 * @param  x  some matrix
 */
mat g_sign(mat& x) {
  return sign(x);
}


/**
 * Computes the generalisation error for the given overlap and self-overlap
 * matrices.
 * 
 * Parameters
 * ----------
 * Q, T: mat (K, K), mat (M, M)
 *     student's and teacher's self-overlap, resp.
 * R: mat (K, M)
 *     student-teacher overlap matrix
 * A, v: second-layer weights of the teacher and student, resp.
 * g1, g2:
 *     teacher's and student's activation function, resp.
 */
double eg_analytical(mat& Q, mat& R, mat& T, vec& A, vec& v,
                     mat (*g1)(mat&), mat (*g2)(mat&)) {
  double epsilon = 0;

  if (g1 == g2 and g1 == g_erf) {
    // student-student overlaps
    vec sqrtQ = sqrt(1 + Q.diag());
    mat normalisation = sqrtQ * sqrtQ.t();
    epsilon += 1. / M_PI * as_scalar(accu(v * v.t() % asin(Q / normalisation)));
  
    // teacher-teacher overlaps
    vec sqrtT = sqrt(1 + T.diag());
    normalisation = sqrtT * sqrtT.t();
    epsilon += 1. / M_PI * as_scalar(accu(A * A.t() % asin(T / normalisation)));

    // student-teacher overlaps
    normalisation = sqrtQ * sqrtT.t();
    epsilon -= 2. / M_PI * as_scalar(accu(v * A.t() % asin(R / normalisation)));
  } else if (g1 == g2 && g1 == g_lin) {
    epsilon = 0.5 * (accu(v * v.t() % Q) + accu(A * A.t() % T)
                     - 2. * accu(v * A.t() % R));
  }

  return epsilon;
}


/**
 * Computes the generalisation error for the given teacher and student network.
 * 
 * Parameters
 * ----------
 * B: mat (M, N)
 *     teacher's weight matrix
 * A : vec (M)
 *     hidden unit-to-output weights of the teacher
 * w : (K, N)
 *     input-to-hidden unit weights of the student
 * v : (K)
 *     hidden unit-to-output weights of the student
 * g1, g2:
 *     teacher's and student's activation function, resp.
 */
double eg_analytical(mat& B, vec& A, mat& w, vec& v,
                     mat (*g1)(mat&), mat (*g2)(mat&)) {
  const int N = B.n_cols;

  mat Q = w * w.t() / N;
  mat R = w * B.t() / N;
  mat T = B * B.t() / N;

  return eg_analytical(Q, R, T, A, v, g1, g2);
}

/**
 * Computes the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w :
 *     input-to-hidden unit weights of the scm
 * v :
 *     hidden unit-to-output weights of the scm
 * xis : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 */
mat phi(mat& w, vec& v, mat& xis, mat (*g)(mat&)) {
  mat act = xis * w.t() / sqrt(w.n_cols);  // activation of the hidden units
  mat hidden = (*g)(act);  // apply the non-linearity point-wise
  return hidden * v;  // and sum up!
}

/**
 * Computes the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w :
 *     input-to-hidden unit weights of the scm
 * xis : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 */
mat phi(mat& w, mat& xis, mat (*g)(mat&)) {
  mat act = xis * w.t() / sqrt(w.n_cols);  // activation of the hidden units
  mat hidden = (*g)(act);  // apply the non-linearity point-wise
  return sum(hidden, 1); // and sum up !
}

/**
 * Numerically computes the mse of an SCM with the given weights as half the
 * squared difference between the SCM's output and the given ys.
 *
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * v : (K)
 *     hidden unit-to-output weights of the scm
 * xis : (bs, N)
 *     input matrix of size (bs, N) on which the mse is evaluated.
 * ys : (bs, 1)
 *     the "true" outputs for the given xis.
 * g :
 *     activation function of the SCM to use.
 */
double mse_numerical(mat& w, vec& v, mat& xis, mat& ys, mat (*g)(mat&)) {
  return 0.5 * as_scalar(mean(pow(ys - phi(w, v, xis, g), 2)));
}


/**
 * Numerically computes the mse between the scalars y1 and y2.
 */
double mse_numerical(mat& y1, mat& y2) {
  if (!(size(y1) == size(y2))) {
    throw std::invalid_argument("matrices y1 and y2 must have the same size.");
  }
  return 0.5 * as_scalar(mean(pow(y1 - y2, 2)));
}


/**
 * Classifies the given outputs of an SCM.
 *
 * Parameters:
 * ----------
 * ys : (bs, 1)
 *     SCM outputs
 *
 * returns:
 * --------
 * classes (bs, 1)
 *     class labels in polar encoding \pm 1
 */
mat classify(mat& ys) {
  mat classes = sign(ys);
  classes.replace(0, 1);
  return classes;
}


/**
 * Classifies the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w : (K, N)
 *     input-to-hidden layer weights of the scm
 * v : (K)
 *     hidden-to-output layer weights 
 * xis : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 *
 * Returns:
 * --------
 * classes (mat)
 *     class labels in polar encoding \pm 1
 */
mat classify(mat& w, vec& v, mat& xis, mat (*g)(mat&)) {
  mat phis = phi(w, v, xis, g);
  return classify(phis);
}

/**
 * Classifies the output of the neural network with the given weights.
 *
 * Parameters:
 * ----------
 * w : (K, N)
 *     input-to-hidden layer weights of the scm
 * xis : (bs, N)
 *     SCM inputs
 * g :
 *     activation function
 * boundary :
 *     decision boundary that separates the two classes if the output function
 *     is ReLU.
 *
 * Returns:
 * --------
 * classes (mat)
 *     binary (0, 1) if the given activation is ReLU, else \pm 1
 */
mat classify(mat& w, mat& xis, mat (*g)(mat&)) {
  mat phis = phi(w, xis, g);
  return classify(phis);
}

/**
 * Computes the weight increment of the input-to-hidden unit weights for
 * gradient descent on the mean squared error.
 *
 * Parameters:
 * -----------
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * v : (K)
 *     hidden unit-to-output weights of the scm
 * scale : double
 *     scalar that rescales the output(s) of the network; useful when using
 *     dropout, for example.
 */
void update_gradients(mat& gradw, vec& gradv, mat& w, vec& v, mat& xis, mat& ys,
                      mat(*g)(mat&), mat(*dgdx)(mat&),
                      bool both, double scale=1) {
  const int N = w.n_cols;
  const int bs = xis.n_rows;  // mini-batch size

  mat act = w * xis.t() / sqrt(N);
  vec error = ys - scale * phi(w, v, xis, g);
  mat deriv = dgdx(act);
  deriv.each_col() %= v;  

  gradw = 1. / bs * deriv * diagmat(error) * xis;

  if (both) {
    gradv = 1. / bs * g(act) * error;
  }
}

/**
 * Computes the weight increment of the input-to-hidden unit weights for
 * gradient descent on the mean squared error.
 *
 * Parameters:
 * -----------
 * w : (K, N)
 *     input-to-hidden unit weights of the scm
 * scale : double
 *     scalar that rescales the output(s) of the network; useful when using
 *     dropout, for example.
 */
void update_gradient(mat& gradw, mat& w, mat& xis, mat& ys,
                     mat(*g)(mat&), mat(*dgdx)(mat&), double scale=1) {
  const int N = w.n_cols;
  const int bs = xis.n_rows;  // mini-batch size

  mat act = w * xis.t() / sqrt(N);
  vec error = ys - scale * phi(w, xis, g);
  gradw = 1. / bs * dgdx(act) * diagmat(error) * xis;
}


/**
 * Sets the teacher weights
 *
 * Returns:
 * --------
 * true if initialisation was successful, false in case of an error.
 */
bool init_teacher_randomly(mat& B0, vec& A0, int N, int M, double uniform,
                           bool both, bool normalise=false, bool meanfield=0,
                           int mix = 0, double sparse=0) {
  B0 = randn<mat>(M, N);   // teacher input-to-hidden weights
  A0 = vec(M, fill::ones);  // teacher hidden-to-output weights
  if (abs(uniform) > 0) {
    A0 *= uniform;
  } else if (both) {
    A0 = vec(M, fill::randn);
  }
  if (normalise) {
    A0 /= M;
  } else if (meanfield) {
    A0 /= sqrt(M);
  }
  if (sparse > 0) {
    // hide a fraction sparse of first-layer teacher weights
    if (sparse > 1) {
      cerr << "Cannot have sparse > 1. Will exit now " << endl;
      return false;
    }
    mat mask = randu<mat>(size(B0));
    mask.elem(find(mask > sparse)).ones();
    mask.elem(find(mask < sparse)).zeros();
    B0 %= mask;
  }

  if (mix) {
    // flip the sign of half of the teacher's second-layer weights
    vec mask = ones<vec>(M);
    mask.head(round(M/2.)) *= -1;
    A0 %= mask;
  }

  return true;
}

/**
 * Randomly initialises the student weights.
 */
void init_student_randomly(mat& w, vec& v, int N, int K, int init,
                           double uniform, bool both, bool normalise,
                           bool meanfield) {
  // INIT_LARGE:
  double prefactor_w = 1;
  double prefactor_v = 1;
  if (init == INIT_SMALL) {
    prefactor_w = 1e-3;
    prefactor_v = 1e-3;
  } else if (init == INIT_MIXED) {
    prefactor_w = 1. / sqrt(N);
    prefactor_v = 1. / sqrt(K);
  } else if (init == INIT_MIXED_NORMALISE) {
    prefactor_w = 1. / N;
    prefactor_v = 1. / K;
  }
  
  w = prefactor_w * randn<mat>(K, N);
  if (both) {
    v = prefactor_v * randn<vec>(K);
  } else {
    v = vec(K, fill::ones);
    if (normalise) {
      if (abs(uniform) > 0) {
        v.fill(uniform / K);
      } else {
        v.fill(1. / K);
      }
    } else if (meanfield) {
      if (abs(uniform) > 0) {
        v.fill(uniform / sqrt(K));
      } else {
        v.fill(1. / sqrt(K));
      }
    } else if (abs(uniform) > 0) {
      v.fill(uniform);
    }
  }
}

/**
 * Performs the online learning.
 *
 * Parameters:
 * -----------
 * B0, w0 :
 *     initial weights of teacher and student, resp. B0 can also be None when
 *     learning from real data.
 * lr :
 *     learning rate
 * lr2 :
 *     learning rate of the second layer
 * wd :
 *     weight decay constant
 * g1, g2 :
 *     teacher's / student's activation function, resp.
 * sigma: scalar
 *     std. dev. of the noise of the teacher's output
 * bs: int
 *     number of samples used to average the gradient per step
 * train_xs, train_ys : (P, N); (P, 1)
 *     if not empty, samples are taken randomly from this finite set.
 * test_xs, test_ys : (bs, N), (bs, 1)
 *     labelled dataset used to compute the generalisation error, even if the
 *     teacher weights are given.
 *
 * Returns:
 *  --------
 * weigths : (w0.shape)
 *     final weights
 */
void learn(mat& B0, vec& A0, mat& w, vec& v,
           mat (*g1)(mat&), mat (*g2)(mat&), mat(*dgdx)(mat&),
           double lr, double lr2, double wd, double dropout_p, double sigma,
           vec steps, int bs,
           mat& train_xs, mat& train_ys,
           mat& test_xs, mat& test_ys,
           FILE* logfile, bool both=false,
           bool classification=false, bool quiet=true, double step=0,
           bool store=false, std::string log_fname="") {
  const int N = w.n_cols;
  const int K = w.n_rows;

  // find the right rescaling of the output units if we're using dropout
  const double dropout_scale = dropout_p > 0 ? 1./(1.-dropout_p) : 1;

  mat gradw = mat(size(w));
  vec gradv = vec(size(v));
  mat dw = mat(size(w), fill::zeros);
  vec dv = vec(size(v), fill::zeros);

  double dstep = 1.0 / N;
  uword step_print_next_idx = 0;
  bool done = false;
  unsigned long batch_idx = 0;
  uvec indices = shuffle(regspace<uvec>(0, train_xs.n_rows - 1));
  bool batch = (bs == (int) train_xs.n_rows);

  // inputs and labels used in an actual sgd step
  mat xis = mat(bs, N);
  mat ys = mat(bs, 1);
  if (batch) {
    xis = train_xs;
    ys = train_ys;
  }

  while(!done) {
    if (step > steps(step_print_next_idx) || step == 0) {
      double eg, et, eg_frac, et_frac;
      
      mat Q = w * w.t() / N;
      mat R = B0.is_empty() ? mat() : w * B0.t() / N;
      mat T = B0.is_empty() ? mat() : B0 * B0.t() / N;

      // compute the TEST error
      if (!test_xs.is_empty()) {
        eg = mse_numerical(w, v, test_xs, test_ys, g2);
        if (classification) {
          mat classes = classify(w, v, test_xs, g2);
          eg_frac = frac_error(classes, test_ys);
        } else {
          eg_frac = datum::nan;
        }
      } else if (!B0.is_empty() && g1 != g_relu && g2 != g_relu) {
        eg = eg_analytical(Q, R, T, A0, v, g1, g2);
        eg_frac = datum::nan;
        if (eg < 1e-14 && step > 1000) {
          done = true;
        }
      } else {
        eg = datum::nan;
        eg_frac = datum::nan;
      }

      // and the TRAINING error
      if (!train_xs.is_empty()) {
        et = mse_numerical(w, v, train_xs, train_ys, g2);
        if (classification) {
          mat classes = classify(w, v, train_xs, g2);
          et_frac = frac_error(classes, train_ys);
        } else {
          et_frac = datum::nan;
        }
      } else {
        et = datum::nan;
        et_frac = datum::nan;
      }

      double diff = as_scalar(mean(mean(abs(dw))));

      std::string msg = status(step, eg, et, eg_frac, et_frac, diff,
                               Q, R, T, A0, v, quiet);
      cout << msg << endl;
      fprintf(logfile, "%s\n", msg.c_str());
      fflush(logfile);

      if (store) {    // store the final teacher/student weights
        std::string fname = std::string(log_fname);
        fname.replace(fname.end()-4, fname.end(), "_w.dat");
        w.save(fname.c_str(), csv_ascii);
        fname.replace(fname.end()-6, fname.end(), "_v.dat");
        v.save(fname.c_str(), csv_ascii);
      }
      

      while (!done && step > steps(step_print_next_idx)) {
        step_print_next_idx++;
        if (step_print_next_idx == steps.n_elem) {
          done = true;
        }
      }
    }

    // TRAINING: first, generate or select the samples for the next step
    if (train_xs.is_empty()) {  // online learning !
      xis = randn<mat>(bs, N);  // random input
      ys = phi(B0, A0, xis, g1);  // output given by the teacher
      if (classification) {
        // turn the output of the teacher into a class label
        ys = classify(ys);
        if (sigma > 0) {
           randomise(ys, sigma);  // and randomise some of the outputs
        }
      } else if (sigma > 0) {  // add some white output noise
        ys += sigma * randn(size(ys));
      }
    } else if (!batch) {
      // select a mini-batch from the fixed training set
      xis = train_xs.rows(indices(span(batch_idx * bs,
                                        (batch_idx + 1) * bs - 1)));
      ys = train_ys.rows(indices(span(batch_idx * bs,
                                        (batch_idx + 1) * bs - 1)));
      batch_idx++;

      if ((batch_idx+1) * bs - 1 >= indices.n_elem) {
        // we have gone through the whole training set once; need to reshuffle!
        indices = shuffle(indices);
        batch_idx = 0;
      }
    } // or we're doing batch gradient descent and there is nothing to choose.

    // (s)gd step
    if (dropout_p > 0) {
      dw.zeros();
      // find the indices of the hidden units that are UPDATED
      vec r = randu<vec>(K);
      uvec idx = find(r > dropout_p);
      if (idx.n_elem == 0) {
        continue;
      }
      mat w_small = w.rows(idx);
      vec v_small = v.elem(idx);
      mat gradw = mat(size(w_small));
      vec gradv = vec(size(v_small));
      update_gradients(gradw, gradv, w_small, v_small, xis, ys, g2, dgdx,
                       both, dropout_scale);

      // compute update for the active hidden units
      dw.zeros();
      dw.rows(idx) = - wd / N * w_small + lr / sqrt(N) * gradw;

      if (both) {
        dv.zeros();
        dv.elem(idx) = -wd / N * v_small + lr2 / N * gradv;
      }
    } else {
      update_gradients(gradw, gradv, w, v, xis, ys, g2, dgdx, both);
      dw = - wd / N * w + lr / sqrt(N) * gradw;

      if (both) {
        dv = -wd / N * v + lr2 / N * gradv;
      }
    }
    w += dw;
    v += dv;
    step += dstep;
  }
}

/**
 * Store MNIST images and labels in the given matrices x and y.
 *
 * Parameters:
 * -----------
 * x, y: mat, mat
 *     matrices to store the images and their labels, resp.
 * digits: vec
 *     if not empty, load only images of the given digits
 */
bool load_mnist(mat& x, mat& y, vec& digits,
                int mode=MNIST_TRAIN, bool oe=false) {
  char* mnist_fname;
  if (mode == MNIST_TRAIN) {
    asprintf(&mnist_fname, "%s/datasets/mnist/mnist_train.csv", getenv("HOME"));
  } else {
    asprintf(&mnist_fname, "%s/datasets/mnist/mnist_test.csv", getenv("HOME"));
  }
  
  mat raw;
  bool loaded = raw.load(mnist_fname, csv_ascii);

  // filter the digits
  if (!digits.empty()) {
    x.reset();
    vec y_raw = raw.col(0);
    for (int i = 0; i < digits.n_elem; i++) {
      uvec indices = find(y_raw == digits(i));
      if (x.empty()) {
        x = raw.rows(indices);
      } else {
        x = join_cols(x, raw.rows(indices));
      }
    }
  } else {
    x = raw;
  }
  // shuffle the images so that all the digits are not in order
  x = shuffle(x);

  // extract the labels into a separate matrix
  y = x.col(0);
  x.shed_col(0);
  x -= mean(mean(x));
  x /= as_scalar(stddev(vectorise(x)));

  if (oe) {
    for (double i = 0.; i < 10; i += 2.) {
      y.replace(i, -1.);
      y.replace(i + 1., 1);
    }
  }
  return loaded;
}


#endif
