library(cmdstanr)  # for interfacing Stan
library(ggplot2)   # for visualizations
library(ggdist)    # for distribution visualizations
library(tidyverse) # for data prep
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse)    # for comparing samples and calculating MCSE
library(shinystan) # for visual diagnostics
library(HDInterval)
library(cowplot)
set.seed(89)

getwd()
setwd("set working directory")

# prepare the data
data <- read.csv("crop.csv", sep = ",")
n_fert = max(data$fertilizer)

############################################## NORMAL MODEL #####################################################

fert1_data <- data %>% filter(fertilizer == 1)
fert2_data <- data %>% filter(fertilizer == 2)
fert3_data <- data %>% filter(fertilizer == 3)

stan_data_1<- list(n = nrow(fert1_data), y = fert1_data$yield)
stan_data_2<- list(n = nrow(fert2_data), y = fert2_data$yield)
stan_data_3<- list(n = nrow(fert3_data), y = fert3_data$yield)

# compile model 
model <- cmdstan_model("models/normal_minimal.stan")

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

rm(stan_data_1, stan_data_2, stan_data_3)
rm(fert1_data,fert2_data, fert3_data)

# convert draws to data frame
df_1 <- as_draws_df(fit_1$draws())
df_2 <- as_draws_df(fit_2$draws())
df_3 <- as_draws_df(fit_3$draws())

rm(model, fit_1, fit_2, fit_3)

# probability mu1 > mu2
mcse(df_1$mu > df_2$mu)                           # 14.345% +- 0.659% 

# probability mu2 > mu1
mcse(df_2$mu > df_1$mu)                           # 85.525% +- 0.659%

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
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) +
  ggtitle("Simple normal model") +
  theme(plot.title = element_text(hjust = 0.5))
plot1
#ggsave("plot1.png", plot = plot1, width = 5, height = 5)

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

mcse(better_1)                     # 14.75% +- 0.561%
mcse(better_2)                     # 16.95% +- 0.593%
mcse(better_3)                     # 68.30% +- 0.736%


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
  labs(x = "yield") +
  ggtitle("Simple normal model") +
  theme(plot.title = element_text(hjust = 0.5))
plot2

#ggsave("plot2.png", plot = plot2, width = 5, height = 5)

rm(next_1,next_2,next_3, better_1, better_2, better_3, i, n)
rm(new_df, df_draws, draws_1, draws_2, draws_3)


############################################### HIERARCHICAL MODEL ##################################################

stan_data <- list(
  n = nrow(data),
  m = max(data$fertilizer),
  y = data$yield,
  s = data$fertilizer
)

# hierarchical normal model ----------------------------------------------------
model_h <- cmdstan_model("models/hierarchical_normal.stan")

# fit
fit_h <- model_h$sample(
  data = stan_data,
  parallel_chains = 4,
  seed = 1,
  adapt_delta = 0.99,
  max_treedepth = 30,
)
rm(stan_data)

# diagnostics
mcmc_trace(fit_h$draws())
fit_h$summary()

# samples
df_h <- as_draws_df(fit_h$draws(c("sigma", "mu", "mu_mu", "sigma_mu")))
df_h <- df_h %>% select(-.draw, -.chain, -.iteration)

# probability mu1 > mu2
mcse(df_h$'mu[1]' > df_h$'mu[2]')                # 16.525% +- 0.881%
           
# probability mu2 > mu1
mcse(df_h$'mu[2]' > df_h$'mu[1]')                # 83.325% +- 0.881%         

# probability mu3 > mu1
mcse(df_h$'mu[3]' > df_h$'mu[1]')                # 98.40% +- 0.294%         

# probability mu2 > mu3
mcse(df_h$'mu[2]' > df_h$'mu[3]')                # 3.025% +- 0.452%           

# probability mu3 > mu2
mcse(df_h$'mu[3]' > df_h$'mu[2]')                # 96.90% +- 0.458%          

# probability taht mu1 > mu2 and mu1 > mu3
mcse((df_h$'mu[1]' > df_h$'mu[2]') & ((df_h$'mu[1]' > df_h$'mu[3]')))   # 0.825% +- 0.187%    

# probability taht mu2 > mu1 and mu2 > mu3
mcse((df_h$'mu[2]' > df_h$'mu[1]') & (df_h$'mu[2]' > df_h$'mu[3]'))     # 2.725% +- 0.415%    

# probability taht mu3 > mu1 and mu3 > mu2
mcse((df_h$'mu[3]' > df_h$'mu[1]') & (df_h$'mu[3]' > df_h$'mu[2]'))     # 96.40% +- 0.555%     

# visualize all three models
new_df <- data.frame(mu = c(df_h$'mu[1]', df_h$'mu[2]', df_h$'mu[3]'), fertilizer = rep(1:3, each = length(df_h$'mu[1]')))
plot3 <- ggplot(data = new_df, aes(x = mu, group = fertilizer, fill = factor(fertilizer))) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) +
  ggtitle("Hierarchical model") +
  theme(plot.title = element_text(hjust = 0.5))
plot3
#ggsave("plot3.png", plot = plot3, width = 5, height = 5)

n <- 4000
better_1 <- vector()
better_2 <- vector()
better_3 <- vector()
for (i in 1:n) {
  next_1 <- rnorm(1, df_h$'mu[1]', df_h$'sigma[1]')
  next_2 <- rnorm(1, df_h$'mu[2]', df_h$'sigma[2]')
  next_3 <- rnorm(1, df_h$'mu[3]', df_h$'sigma[3]')
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

mcse(better_1)                     # 13.10% +- 0.533%
mcse(better_2)                     # 34.325% +- 0.751%
mcse(better_3)                     # 52.575% +- 0.787%


# number of draws
n <- 100000
# draws from the distributions
draws_1 <- rnorm(n, mean = df_h$`mu[1]`, sd = df_h$`sigma[1]`)
draws_2 <- rnorm(n, df_h$'mu[2]', df_h$'sigma[2]')
draws_3 <- rnorm(n, df_h$'mu[3]', df_h$'sigma[3]')

# compare
mcse((draws_1 > draws_2) & (draws_1 > draws_3))      # 16.224% +- 0.117%             # 13.930% +- 0.110%
mcse((draws_2 > draws_3) & (draws_2 > draws_1))      # 19.197% +- 0.125%             # 17.153% +- 0.119%
mcse((draws_3 > draws_1) & (draws_3 > draws_2))      # 64.579% +- 0.156%             # 68.910% +- 0.148%


# visualize
df_draws <- data.frame(value = c(draws_1, draws_2, draws_3), fertilizer = rep(1:3, each = length(draws_1)))
plot4 <- ggplot(data = df_draws, aes(x = value, group = fertilizer, fill = factor(fertilizer))) +
  geom_density(alpha = 0.5, color = NA) +
  scale_fill_manual(values = c("gold", "navy", "maroon"), name = "",label = c("Fertilizer 1", "Fertilizer 2", "Fertilizer 3")) +
  scale_x_continuous(limits = c(173,181)) +
  labs(x = "yield")+
  ggtitle("Hierarchical model") +
  theme(plot.title = element_text(hjust = 0.5))
plot4
#ggsave("plot4.png", plot = plot4, width = 5, height = 5)

rm(next_1,next_2,next_3, better_1, better_2, better_3, i, n)
rm(new_df, df_draws, draws_1, draws_2, draws_3)


################################## cOMPARE GROUP LEVEL MEANS #########################################################


df_group <- data.frame(
  Mean = numeric(),
  HDI5 = numeric(),
  HDI95 = numeric(),
  Model = character()
)

# sample
sample_mean <- mean(data$yield)
df_group <- rbind(df_group, data.frame(
  Mean = sample_mean,
  HDI5 = sample_mean,
  HDI95 = sample_mean,
  Model = "Sample"
))


# simple normal model
df_n <- data.frame(mu = c(df_1$mu, df_2$mu, df_3$mu))
normal_mean <- mean(df_n$mu)
normal_90_hdi <- hdi(df_n$mu, credMass = 0.9)
df_group <- rbind(df_group, data.frame(
  Mean = normal_mean,
  HDI5 = normal_90_hdi[1],
  HDI95 = normal_90_hdi[2],
  Model = "Simple normal"
))

# hierarchical model
hierarchical_mean <- mean(df_h$mu_mu)
hierarchical_90_hdi <- hdi(df_h$mu_mu, credMass = 0.9)
df_group <- rbind(df_group, data.frame(
  Mean = hierarchical_mean,
  HDI5 = hierarchical_90_hdi[1],
  HDI95 = hierarchical_90_hdi[2],
  Model = "Hierarchical"
))


# plot
# set model factors so the colors are the same
df_group$Model <- factor(df_group$Model,
                         levels = c("Simple normal", "Hierarchical", "Sample")
)


ggplot(
  data = df_group,
  aes(
    x = Model,
    y = Mean,
    ymin = HDI5,
    ymax = HDI95,
    colour = Model
  )
) +
  geom_point() +
  geom_errorbar(width = 0.3, linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  ggtitle("a)")+
  #ylim(0, 6) +
  theme(plot.title = element_text(size = 18), axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), axis.title.x = element_text(size = 14),  
        axis.title.y = element_text(size = 14),  
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14))
        #legend.position = 'top',
        #legend.justification = "left",
        #legend.box = 'vertical')


############################################# COMPARE SUBJECT LEVEL MEANS ###########################################

df_subject <- data.frame(
  Mean = numeric(),
  Q5 = numeric(),
  Q95 = numeric(),
  Model = character(),
  fert = numeric()
)

# sample means
df_mu_sample <- data %>%
  group_by(fertilizer) %>%
  summarise(mean_weight = mean(yield))
df_subject <- rbind(df_subject, data.frame(
  Mean = df_mu_sample$mean_weight,
  HDI5 = df_mu_sample$mean_weight,
  HDI95 = df_mu_sample$mean_weight,
  Model = "Sample",
  fert = seq(1:n_fert)
))

# subject means
df_s <- data.frame(df_1$mu, df_2$mu, df_3$mu)
df_mu_s <- df_s 
s_means <- colMeans(df_mu_s)
s_90_hdi <- apply(df_mu_s, 2, hdi, credMass = 0.9)
df_subject <- rbind(df_subject, data.frame(
  Mean = s_means,
  HDI5 = s_90_hdi[1, ],
  HDI95 = s_90_hdi[2, ],
  Model = "Simple normal",
  fert = seq(1:n_fert)
))


# hierarchical means
df_mu_h <- df_h %>% select(4:(3 + n_fert))
h_means <- colMeans(df_mu_h)
h_hdi90 <- apply(df_mu_h, 2, hdi, credMass = 0.9)
df_subject <- rbind(df_subject, data.frame(
  Mean = h_means,
  HDI5 = h_hdi90[1, ],
  HDI95 = h_hdi90[2, ],
  Model = "Hierarchical",
  fert = seq(1:n_fert)
))

# plot
# set model factors so the colors are the same
df_subject$Model <- factor(df_subject$Model,
                           levels = c("Simple normal","Hierarchical", "Sample")
)


# plot
ggplot(
  data = df_subject,
  aes(
    x = Model,
    y = Mean,
    ymin = HDI5,
    ymax = HDI95,
    colour = Model
  )
) +
  geom_hline(yintercept = mean(data$yield), color = "grey75") +
  geom_point() +
  geom_errorbar(width = 0.3, linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  ggtitle("b)") +
  #ylim(0, 6) +
  facet_wrap(. ~ fert, ncol=3) +
  theme(plot.title = element_text(size = 18), axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), axis.title.x = element_text(size = 14),  
        axis.title.y = element_text(size = 14),  
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.box = 'horizontal')




