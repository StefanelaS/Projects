data {
  int<lower=1> n; // total number of data points
  int<lower=1> k; // number of predictors
  matrix[n, k] X; // independent variables
  vector[n] y;    // dependent variable
}

parameters {
  real alpha;
  vector[k] beta;         // slope
  real<lower=0> lambda; // precision parameter
}

model {
  //Priors
  beta ~ cauchy (0, 2.5);
  
  //Likelihood
  vector[n] mu;           // expected values
  mu = exp(alpha + X * beta);
  y ~ gamma(mu .* lambda, lambda);
  
}

generated quantities {
  vector[n] pred;
  for (i in 1:n) {
    pred[i] = gamma_rng(exp(alpha + X[i] * beta).*lambda, lambda);
  }
}

