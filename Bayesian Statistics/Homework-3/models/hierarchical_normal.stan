data {
  int<lower=0> n; // number of observations
  int<lower=0> m; // number of subjects
  vector[n] y;    // observations
  array[n] int s; // subject indexes
}

parameters {
  real<lower=0> sigma_mu;
  real mu_mu;
  real<lower=0> sigma_sigma;
  real mu_sigma;
  vector[m] sigma;
  vector[m] mu;
}

model {
  // hierarchical link
  mu ~ normal(mu_mu, sigma_mu);
  sigma ~ normal(mu_sigma, sigma_sigma);
  
  // normal model for each group
  for (i in 1:n) {
      y[i] ~ normal(mu[s[i]], sigma[s[i]]);
  }
}
