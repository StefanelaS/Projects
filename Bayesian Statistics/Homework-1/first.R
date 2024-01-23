# load the libraries
library(cmdstanr)  # for interfacing Stan
library(ggplot2)   # for visualizations
library(ggdist)    # for distribution visualizations
library(tidyverse) # for data prep
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse)    # for comparing samples and calculating MCSE
library(shinystan) # for visual diagnostics
set.seed(89)

getwd()
setwd("link to working directory")

# prepare the data
data <- read.csv("crop.csv", sep = ",")

ggplot(data, aes(x = yield, group = fertilizer, fill = factor(fertilizer))) + 
  geom_histogram(bins = 12, color = 'black', position = "dodge") +
  labs(x = 'Yield', y = 'Count') +
  scale_fill_manual(values = c("blue", "green", "red"), name = "", 
                    label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) +
  theme_minimal()

fert1_data <- data %>% filter(fertilizer == 1)
fert2_data <- data %>% filter(fertilizer == 2)
fert3_data <- data %>% filter(fertilizer == 3)

stan_data_1<- list(n = nrow(fert1_data), y = fert1_data$yield)
stan_data_2<- list(n = nrow(fert2_data), y = fert2_data$yield)
stan_data_3<- list(n = nrow(fert3_data), y = fert3_data$yield)

# compile model 
model <- cmdstan_model("normal_minimal.stan")

# fitting and diagnostics
fit_1 <- model$sample(
  data = stan_data_1,
  seed = 1
)

fit_2 <- model$sample(
  data = stan_data_2,
  seed = 1
)

fit_3 <- model$sample(
  data = stan_data_3,
  seed = 1
)

# summary
fit_1$summary()
fit_2$summary()
fit_3$summary()

# traceplots
mcmc_trace(fit_1$draws("mu"))
mcmc_trace(fit_1$draws("sigma"))
mcmc_trace(fit_2$draws("mu"))

# additional automated diagnostics
fit_1$cmdstan_diagnose()
fit_2$cmdstan_diagnose()
fit_3$cmdstan_diagnose()

# convert draws to data frame
df_1 <- as_draws_df(fit_1$draws())
df_2 <- as_draws_df(fit_2$draws())
df_3 <- as_draws_df(fit_3$draws())

# probability mu1 > mu2
mcse(df_1$mu - df_2$mu)             
mcse(df_1$mu > df_2$mu)             

# probability mu2 > mu1
mcse(df_2$mu - df_1$mu)               
mcse(df_2$mu > df_1$mu)               
                                                  # 85.525% +- 0.659%
# probability mu1 > mu3
mcse(df_1$mu > df_3$mu)                           # 0.375% +- 0.170%

# probability mu3 > mu1
mcse(df_3$mu > df_1$mu)                           # 99.675% +- 0.170%

# probability mu2 > mu3
mcse(df_2$mu > df_3$mu)                           # 0.9% +- 0.311%

# probability mu3 > mu2
mcse(df_3$mu > df_2$mu)                           # 99.1% +- 0.311%

# probability taht mu1 > mu2 and mu1 > mu3
mcse((df_1$mu > df_2$mu) & (df_1$mu > df_3$mu))   # 0.1% +- 0.05%

# probability taht mu2 > mu1 and mu2 > mu3
mcse((df_2$mu > df_1$mu) & (df_2$mu > df_3$mu))   # 0.825% +- 0.219% 

# probability taht mu3 > mu1 and mu3 > mu2
mcse((df_3$mu > df_1$mu) & (df_3$mu > df_2$mu))   # 99.075% +- 0.312%

# visualize all three models
df_1$fertilizer <- 1
df_2$fertilizer <- 2
df_3$fertilizer <- 3
new_df <- data.frame(mu = c(df_1$mu, df_2$mu, df_3$mu), fertilizer = c(df_1$fertilizer, df_2$fertilizer, df_3$fertilizer))
plot1 <- ggplot(data = new_df, aes(x = mu, group = fertilizer, fill = factor(fertilizer))) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3"))
plot1    
ggsave("plot1.png", plot = plot1, width = 5, height = 5)

# number of draws
n <- 100000

# draws from the distributions
draws_1 <- rnorm(n, mean = df_1$mu, sd = df_1$sigma)
draws_2 <- rnorm(n, df_2$mu, df_2$sigma)
draws_3 <- rnorm(n, df_3$mu, df_3$sigma)

# compare
mcse((draws_1 > draws_2) & (draws_1 > draws_3))      # 13.93% +- 0.110%
mcse((draws_2 > draws_3) & (draws_2 > draws_1))      # 17.153% +- 0.119%
mcse((draws_3 > draws_1) & (draws_3 > draws_2))      # 68.91% +- 0.148%

# visualize
df_draws <- data.frame(value = c(draws_1, draws_2, draws_3), fertilizer = rep(1:3, each = length(draws_1)))
plot2 <- ggplot(data = df_draws, aes(x = value, group = fertilizer, fill = factor(fertilizer))) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) +
  scale_x_continuous(limits = c(173,181)) +
  labs(x = "yield")

ggsave("plot2.png", plot = plot2, width = 5, height = 5)


set.seed(1)
n <- 4000
better_1 <- vector()
better_2 <- vector()
better_3 <- vector()
for (i in 1:n) {
  next_1 <- rnorm(1, df_1$mu[i], df_1$sigma[i])
  next_2 <- rnorm(1, df_2$mu[i], df_2$sigma[i])
  next_3 <- rnorm(1, df_3$mu[i], df_3$sigma[i])
  if ((next_3 > next_2) & (next_3 > next_1)){
    better_3 <- c(better_3, 1)
  } else {
    better_3 <- c(better_3, 0)
  }
  if ((next_2 > next_3) & (next_2 > next_1)){
    better_2 <- c(better_2, 1)
  } else {
    better_2 <- c(better_2, 0)
  }
  if ((next_1 > next_3) & (next_1 > next_2)){
    better_1 <- c(better_1, 1)
  } else {
    better_1 <- c(better_1, 0)
  }
}
mcse(better_1)
mcse(better_2)
mcse(better_3)


set.seed(1)
n <- 4000
better_1 <- vector()
better_2 <- vector()
better_3 <- vector()
for (i in 1:n) {
  next_1 <- rnorm(10, df_1$mu[i], df_1$sigma[i])
  next_2 <- rnorm(10, df_2$mu[i], df_2$sigma[i])
  next_3 <- rnorm(10, df_3$mu[i], df_3$sigma[i])
  for (i in 1:10){
    if ((next_3[i] > next_2[i]) & (next_3[i] > next_1[i])){
      better_3<- c(better_3, 1)
    } else {
      better_3 <- c(better_3, 0)
    }
    if ((next_2[i] > next_3[i]) & (next_2[i] > next_1[i])){
      better_2 <- c(better_2, 1)
    } else {
      better_2 <- c(better_2, 0)
    }
    if ((next_1[i] > next_3[i]) & (next_1[i] > next_2[i])){
      better_1 <- c(better_1, 1)
    } else {
      better_1 <- c(better_1, 0)
    }
  }
}
mcse(better_1)
mcse(better_2)
mcse(better_3)

better_df <- data.frame(mu = c(df_1$mu, df_2$mu, df_3$mu), fertilizer = c(df_1$fertilizer, df_2$fertilizer, df_3$fertilizer))
ggplot(data = new_df, aes(x = mu, group = fertilizer, fill = factor(fertilizer))) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) 
scale_fill_brewer(type = "qual", palette = 3) +
  theme_minimal()