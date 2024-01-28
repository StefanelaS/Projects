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
library(psych) # for independent variables correlation plot
library(vioplot)
library(grid)
library(gridExtra)
library(dplyr)
library(loo)

set.seed(89)
setwd("C:/Users/Korisnik/Desktop/FAKS/Bayesian Statistics/Homework-4")


############################################## DATA PREPARATION ####################################################

# load the world happiness data report
data <- read.csv("whr_2023.csv", sep = ",")
data <- data %>% select(-Negative.affect, -Positive.affect)
colnames(data) <- c('country', 'year', 'score', 'economy','social_support','health','freedom', 'generosity', 'perceived_corruption')

# filter to keep only years 2016 - 2022
data <- data %>% filter(year > 2015)
data <- data %>% filter(year < 2023)
data <- data %>% na.omit()


# correlation plot
pairs.panels(data,
             method = "pearson", # correlation method
             hist.col = "darksalmon",
             density = TRUE, # show density plots
             ellipses = TRUE # show correlation ellipses
)


# define X and Y
X <- data %>% select(economy, social_support, freedom, perceived_corruption, generosity)
y <- data %>% select(score)
n <- dim(X)[1]
m_max <- dim(X)[2] 
X <- X %>% mutate(intercept = 1) %>% select(intercept, everything())


# normalization
normalize <- function(x, na.rm = TRUE) {
  return((x- min(x)) /(max(x)-min(x)))
}

X$economy <- normalize(X$economy)
X$social_support <- normalize(X$social_support)
X$freedom <- normalize(X$freedom)
X$perceived_corruption <- normalize(X$perceived_corruption)
X$generosity <- normalize(X$generosity)


# plotting the data
plot_data <- X %>% select(economy, social_support, freedom, perceived_corruption, generosity)
plot_data$score <- y$score
colnames(plot_data) <- c('economy', 'social_support', 'freedom', 'perceived_corruption', 'generosity', 'happiness_score')
my_plots <- lapply(names(plot_data), function(var_x){
  p <- ggplot(plot_data) + aes_string(var_x)
  if(is.numeric(plot_data[[var_x]])) {
    p <- p + geom_density(fill = "mediumblue", alpha = 0.3)  + theme(text = element_text(size = 25))
    
  } else {
    p <- p + geom_bar() + theme(text = element_text(size = 25))
  } 
})

plot_data <- plot_grid(plotlist = my_plots)
plot_data
ggsave("plot_data.png", plot = plot_data, width = 14, height = 8)


############################################## MODELS ##########################################################

# compile model
model <- cmdstan_model("models/linear_deviance.stan")

# stan_data
stan_data <- list(
  n = n,
  m_max = m_max,
  X = X,
  y = y$score
)

# storages
log_lik <- list()
df_aic <- data.frame(AIC = numeric(), order = factor())

# linear regression with different number of predictors (1-5)
for (m in 1:m_max) {
  # set order
  stan_data$m <- m
  
  # fit
  fit <- model$sample(
    data = stan_data,
    parallel_chains = 4,
    iter_warmup = 500,
    iter_sampling = 500,
    seed = 1
  )
  
  # uncomment lines below for diagnostic purposes
  # traceplot
  # mcmc_trace(fit$draws(c("b", "sigma")))
  # summary
  # fit$summary(c("b", "sigma"))
  
  # extract
  log_lik[[m]] <- fit$draws(c("log_lik"))
  df_ll <- as_draws_df(fit$draws(c("log_lik")))
  
  # remove unwanted columns
  # also cast to regular data frame to avoid some warnings later on
  df_ll <- data.frame(df_ll %>% select(-.chain, -.iteration, -.draw))
  
  # average per row and store
  df_aic <- rbind(
    df_aic,
    data.frame(AIC = -2 * rowSums(df_ll) + 2 * (m + 1), Order = as.factor(m))
  )
}


# linear regression with interactions ---------------------------------------------------------

# model 6 -------------------------------------------------------------------------------------
X_new <- X %>% select(intercept,economy, social_support, freedom, perceived_corruption)
X_new$ec <- data$economy*data$perceived_corruption
X_new$ec <- normalize(X_new$ec)
m_max <- dim(X_new)[2] - 1

# stan_data
stan_data1 <- list(
  n = n,
  m = m_max,
  m_max = m_max,
  X = X_new,
  y = y$score
)

# fit
fit1 <- model$sample(
  data = stan_data1,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  seed = 1
)

fit1$summary(c("b", "sigma"))

# extract
log_lik[[6]] <- fit1$draws(c("log_lik"))

# model 7 -----------------------------------------------------------------------------------
X_new <- X %>% select(intercept, economy, social_support, freedom, perceived_corruption)
X_new$ef <- data$economy*data$freedom
X_new$ef <- normalize(X_new$ef)
m_max <- dim(X_new)[2] - 1

# stan_data
stan_data2 <- list(
  n = n,
  m = m_max,
  m_max = m_max,
  X = X_new,
  y = y$score
)

# fit
fit2 <- model$sample(
  data = stan_data2,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  seed = 1
)

fit2$summary(c("b", "sigma"))

# extract
log_lik[[7]] <- fit2$draws(c("log_lik"))


# model 8 ------------------------------------------------------------------------------------
X_new <- X %>% select(intercept, economy, social_support, freedom, perceived_corruption, generosity)
X_new$ec <- data$economy*data$perceived_corruption
X_new$es <- data$economy*data$social_support
X_new$ef <- data$economy*data$freedom
X_new$eg <- data$economy*data$generosity
X_new$ec <- normalize(X_new$ec)
X_new$es <- normalize(X_new$es)
X_new$ef <- normalize(X_new$ef)
X_new$eg <- normalize(X_new$eg)
X_new <- X_new %>% select(-economy, -social_support, -freedom, -perceived_corruption, -generosity)
m_max <- dim(X_new)[2] - 1


# stan_data
stan_data3 <- list(
  n = n,
  m = m_max,
  m_max = m_max,
  X = X_new,
  y = y$score
)

# fit
fit3 <- model$sample(
  data = stan_data3,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  seed = 1
)

fit3$summary(c("b", "sigma"))

# extract
log_lik[[8]] <- fit3$draws(c("log_lik"))

# model 9 ---------------------------------------------------------------------------------------
X_new <- X %>% select(intercept, economy, social_support, freedom, perceived_corruption, generosity)
X_new$es <- data$economy*data$social_support
X_new$ef <- data$economy*data$freedom
X_new$ec <- data$economy*data$perceived_corruption
X_new$eg <- data$economy*data$generosity
X_new$cs <- data$perceived_corruption*data$social_support
X_new$cf <- data$perceived_corruption*data$freedom
X_new$cg <- data$perceived_corruption*data$generosity

X_new$ec <- normalize(X_new$ec)
X_new$es <- normalize(X_new$es)
X_new$ef <- normalize(X_new$ef)
X_new$eg <- normalize(X_new$eg)
X_new$cs <- normalize(X_new$cs)
X_new$cf <- normalize(X_new$cf)
X_new$cg <- normalize(X_new$cg)

X_new <- X_new %>% select(-economy, -social_support, -freedom, -perceived_corruption, -generosity)
m_max <- dim(X_new)[2] - 1

# stan_data
stan_data4 <- list(
  n = n,
  m = m_max,
  m_max = m_max,
  X = X_new,
  y = y$score
)

# fit
fit4 <- model$sample(
  data = stan_data4,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  seed = 1
)

fit4$summary(c("b", "sigma"))

# extract
log_lik[[9]] <- fit4$draws(c("log_lik"))

# model 10 ---------------------------------------------------------------------------------------
X_new <- X %>% select(intercept, economy, social_support, freedom, perceived_corruption, generosity)
X_new$es <- data$economy*data$social_support
X_new$ef <- data$economy*data$freedom
X_new$ec <- data$economy*data$perceived_corruption
X_new$eg <- data$economy*data$generosity
X_new$fs <- data$freedom*data$social_support
X_new$fc <- data$perceived_corruption*data$freedom
X_new$fg <- data$freedom*data$generosity

X_new$ec <- normalize(X_new$ec)
X_new$es <- normalize(X_new$es)
X_new$ef <- normalize(X_new$ef)
X_new$eg <- normalize(X_new$eg)
X_new$fs <- normalize(X_new$fs)
X_new$fc <- normalize(X_new$fc)
X_new$fg <- normalize(X_new$fg)
X_new <- X_new %>% select(-economy, -social_support, -freedom, -perceived_corruption, -generosity)
m_max <- dim(X_new)[2] - 1

# stan_data
stan_data5 <- list(
  n = n,
  m = m_max,
  m_max = m_max,
  X = X_new,
  y = y$score
)

# fit
fit5 <- model$sample(
  data = stan_data5,
  parallel_chains = 4,
  iter_warmup = 500,
  iter_sampling = 500,
  seed = 1
)

fit5$summary(c("b", "sigma"))

# extract
draws5 <- fit5$draws(c("log_lik"))
log_lik[[10]] <- draws5


# PLOT LOOIC ------------------------------------------------------------------------
df_looic <- data.frame(looic = numeric(), SE = numeric(), Order = factor())

for (i in 1:length(log_lik)) {
  r_eff <- relative_eff(log_lik[[i]])
  loo <- loo(log_lik[[i]], r_eff = r_eff)
  df_looic <- rbind(df_looic, data.frame(
    looic = loo$estimates[3, 1],
    SE = loo$estimates[3, 2],
    Order = as.factor(i)
  ))
}

# plot
plot_looic <- ggplot(data = df_looic, aes(x = Order, y = looic)) +
  geom_point(shape = 16, size = 2) +
  geom_linerange(aes(ymin = (looic - SE), ymax = (looic + SE)), alpha = 0.3) +
  xlab("Model") +
  ylab("LOOIC")
plot_looic
ggsave("plot_looic.png", plot = plot_looic, width = 6, height = 4)

# PLOT AKAIKE WEIGHTS for model combination -----------------------------------------

# calculate delta_looic
df_looic$delta_looic <- abs(df_looic$looic - min(df_looic$looic))

# calculate weights
df_looic$weight <-
  exp(-0.5 * df_looic$delta_looic) / sum(exp(-0.5 * df_looic$delta_looic))
df_looic$weight <- round(df_looic$weight, 2)

# calculate worst and best case scenario for each model in terms of +/- SE
# worst case: looic + SE for the choosen model, looic - SE for the rest
# best case: looic - SE for the choosen model, looic + SE for the rest
df_looic$weight_plus_se <- 0
df_looic$weight_minus_se <- 0
for (i in seq_len(nrow(df_looic))) {
  # worst case
  # best variant of other models
  looic_plus_se <- df_looic$looic - df_looic$SE
  # worst of current model
  looic_plus_se[i] <- df_looic$looic[i] + df_looic$SE[i]
  # weights
  delta_looic <- abs(looic_plus_se - min(looic_plus_se))
  weights <- exp(-0.5 * delta_looic) / sum(exp(-0.5 * delta_looic))
  df_looic$weight_plus_se[i] <- round(weights[i], 2)
  
  # best case
  # worst variant of other models
  looic_minuse_se <- df_looic$looic + df_looic$SE
  # best of current model
  looic_minuse_se[i] <- df_looic$looic[i] - df_looic$SE[i]
  # weights
  delta_looic <- abs(looic_minuse_se - min(looic_minuse_se))
  weights <- exp(-0.5 * delta_looic) / sum(exp(-0.5 * delta_looic))
  df_looic$weight_minus_se[i] <- round(weights[i], 2)
}

# plot
akaike_plot <- ggplot(data = df_looic, aes(x = Order, y = weight)) +
  geom_errorbar(
    aes(ymin = weight_minus_se, ymax = weight_plus_se),
    width = 0.25,
    color = "mediumblue",
    alpha = 0.8
  ) +
  geom_point(shape = 16, color = "mediumblue", size = 2) +
  xlab("Model") +
  ylab("Akaike weight") +
  #theme_minimal() +
  ylim(0, 1)
akaike_plot
ggsave("akaike.png", plot = akaike_plot, width = 6, height = 4)

# print
df_looic