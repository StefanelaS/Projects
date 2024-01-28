data {
  int<lower=0> n;          // total number of measurements
  int<lower=0> m;          // number of subjects
  vector[n] x;             // sequence/time
  vector[n] y;             // response
  array[n] int<lower=0> s; // subject ids
}

parameters {
  // subject level parameters
  vector[m] alpha;
  vector[m] beta;
  real<lower=0> sigma;
  
  // global parameters
  real mu_a;
  real mu_b;
  real<lower=0> sigma_a;
  real<lower=0> sigma_b;
}

model {
  // hierarchical links between subjects
  alpha ~ normal(mu_a, sigma_a);
  beta ~ normal(mu_b, sigma_b);

  
  // subject level modelling - normal linear regression
  for (i in 1:n)
    y[i] ~ normal(alpha[s[i]] + beta[s[i]] * x[i], sigma);
}

generated quantities {
  vector[n] pred;    // predictions for each data point
  for (i in 1:n) {
    pred[i] = normal_rng(alpha[s[i]] + x[i] * beta[s[i]], sigma);
  }
}
