data {
  int<lower=1> n; // total number of data points
  int<lower=1> k; // number of predictors
  matrix[n, k] X; // independent variables
  vector[n] y;    // dependent variable
}

parameters {
  vector[k] b;         // slope
  real<lower=0> sigma; // stdev
}

model {
  // model
  y ~ normal(X * b, sigma);
  sigma ~ cauchy (0, 3);
}

generated quantities {
  vector[n] pred;    // predictions for each data point
  for (i in 1:n) {
    pred[i] = normal_rng(X[i] * b, sigma);
  }
}
