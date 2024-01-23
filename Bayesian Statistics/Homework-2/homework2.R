library(cmdstanr)  # for interfacing Stan
library(ggplot2)   # for visualizations
library(ggdist)    # for distribution visualizations
library(tidyverse) # for data prep
library(posterior) # for extracting samples
library(bayesplot) # for some quick MCMC visualizations
library(mcmcse)    # for comparing samples and calculating MCSE
library(shinystan) # for visual diagnostics
library(psych) # for independent variables correlation plot
library(vioplot)
library(grid)
library(gridExtra)

set.seed(89)

getwd()
setwd('link to working directory')

data <- read.csv('crime.csv')
colnames(data)[1] <- 'crime_rate'

# CORRELATION plot

pairs.panels(data,
             method = "pearson", # correlation method
             hist.col = "darksalmon",
             density = TRUE, # show density plots
             ellipses = TRUE # show correlation ellipses
)


data <- data %>% select (-X25_plus_4_years_college)

y1 <- data$crime_rate
y2 <- data$violent_crime_rate
X <- data %>% select(-crime_rate, -violent_crime_rate)


########################################FEATURE ENGINEERING #######################################################


X$high_school <- ((X$X25_plus_high_school)*0.93 + (100-X$X16_19_no_high_school)*0.07)
X$college <- X$X18_24_college

X <- select(X, -X16_19_no_high_school, -X25_plus_high_school, -X18_24_college)


X <- scale(X)
X <- as.data.frame(X)

X$police_funding <- (X$police_funding - min(X$police_funding)) / (max(X$police_funding) - min(X$police_funding))
X$high_school <- (X$high_school - min(X$high_school)) / (max(X$high_school) - min(X$high_school))
X$college <- (X$college - min(X$college)) / (max(X$college) - min(X$college))


########################################## NORMAL - CRIME RATE #######################################################

# compile the model
model1 <- cmdstan_model("linear_nointercept.stan")

stan_data1 <- list(n = length(y1), k = ncol(X), y = y1, X = X)

# fit
fit <- model1$sample(
  data = stan_data1,
  parallel_chains = 4,
  seed = 1
)


# diagnostics ------------------------------------------------------------------
fit$summary()

# additional automated diagnostics
fit$cmdstan_diagnose()

# extract samples --------------------------------------------------------------
df1_cr <- as_draws_df(fit$draws())


# create a data frame with MCSE values
coeff1_cr <- data.frame(
  b1 = df1_cr$`b[1]`,
  b2 = df1_cr$`b[2]`,
  b3 = df1_cr$`b[3]`
)

colnames(coeff1_cr) <- c('police_funding', 'high_school', 'college')

# to long format
coeff1_cr_lf <- coeff1_cr %>% gather(
  Variable,
  Value,
  c(police_funding, high_school, college)
)


# generate predictions using posterior samples
pred1_cr <- data.frame(matrix(NA, nrow = 50, ncol = 50))

for (i in 1:50) {
  predictions <- coeff1_cr[2000+i, 1] * X$police_funding + coeff1_cr[2000+i, 2] * X$high_school + coeff1_cr[2000+i, 3] * X$college
  pred1_cr[i, ] <- predictions
}


########################################## NORMAL - VIOLENT CRIME RATE #######################################################

stan_data2 <- list(n = length(y2), k = ncol(X), y = y2, X = X)

# fit
fit <- model1$sample(
  data = stan_data2,
  parallel_chains = 4,
  seed = 1
)

# diagnostics ------------------------------------------------------------------
fit$summary()


# additional automated diagnostics
fit$cmdstan_diagnose()


# extract samples --------------------------------------------------------------
df1_vcr <- as_draws_df(fit$draws())

# create a data frame with MCSE values
coeff1_vcr <- data.frame(
  b1 = df1_vcr$`b[1]`,
  b2 = df1_vcr$`b[2]`,
  b3 = df1_vcr$`b[3]`
)

colnames(coeff1_vcr) <- c('police_funding', 'high_school', 'college')


# to long format
coeff1_vcr_lf <- coeff1_vcr %>% gather(
  Variable,
  Value,
  c(police_funding, high_school, college)
)

# generate predictions using posterior samples
pred1_vcr <- data.frame(matrix(NA, nrow = 50, ncol = 50))

for (i in 1:50) {
  predictions <- coeff1_vcr[2000+i, 1] * X$police_funding + coeff1_vcr[2000+i, 2] * X$high_school + coeff1_vcr[2000+i, 3] * X$college
  
  pred1_vcr[i, ] <- predictions
}


########################################## NORMAL - PLOTTING #######################################################

coeff1_cr_lf$Type <- 'Crime Rate'
coeff1_vcr_lf$Type <- 'Violent Crime Rate'

coeff1 <- rbind(coeff1_cr_lf, coeff1_vcr_lf)


# PLOTTING COEFFICIENTS
ggplot(data = coeff1, aes(x = Value, y = Variable)) +
  stat_eye(fill = "seagreen", alpha = 0.5) +
  facet_grid(Type ~ .) +
  labs(x = "Beta Value", y = "Variable") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darksalmon", linewidth = 1) +
  theme(strip.text = element_text(size = 14))


# transpose predictions
normal_cr <- as.data.frame(t(as.matrix(pred1_cr)))


# histogram
histogram_cr <- ggplot(data, aes(x = crime_rate)) +
  geom_histogram(aes(y = ..density..),bins = 22, alpha = 0.5 , color = 'black', fill = 'bisque', linewidth = 0.8) 

normal_plot <- histogram_cr + 
  geom_line(data = normal_cr, aes(x = V1), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)

normal_plot

# extract column names 
col_names <- colnames(normal_cr)


# loop to add density lines for each column
for (col_name in col_names) {
  normal_plot <- normal_plot +
    geom_line(data = normal_cr, aes(x = .data[[col_name]]), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)
}


# transpose predictions
normal_vcr <- as.data.frame(t(as.matrix(pred1_vcr)))


# histogram
histogram_vcr <- ggplot(data, aes(x = violent_crime_rate)) +
  geom_histogram(aes(y = ..density..),bins = 22, alpha = 0.5 , color = 'black', fill = 'bisque', linewidth = 0.8) 

normal_plot_vcr <- histogram_vcr + 
  geom_line(data = normal_vcr, aes(x = V1), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)

# extract column names 
col_names <- colnames(normal_vcr)

# loop to add density lines for each column
for (col_name in col_names) {
  normal_plot_vcr <- normal_plot_vcr +
    geom_line(data = normal_vcr, aes(x = .data[[col_name]]), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)
}

grid.arrange(normal_plot, normal_plot_vcr, ncol(1))


X <- cbind(data$crime_rate, data$violent_crime_rate, X)
colnames(X)[1] <- "crime_rate"
colnames(X)[2] <- "violent_crime_rate"



########################################## GAMMA - CRIME RATE #######################################################

model2 <- cmdstan_model("gamma.stan")

stan_data3 <- list(n = length(y2), k = ncol(X), y = y1, X = X)


# fit
fit <- model2$sample(
  data = stan_data3,
  parallel_chains = 4,
  seed = 1
)

# diagnostics ------------------------------------------------------------------

# summary of betas
fit$summary("beta")

# additional automated diagnostics
fit$cmdstan_diagnose()

# extract parameters
coeff2_cr <- as_draws_df(fit$draws("beta"))
pred2_cr <- as_draws_df(fit$draws("pred"))
coeff2_cr <- coeff2_cr %>% select(-.chain, -.iteration, -.draw)
pred2_cr <- pred2_cr %>% select(-.chain, -.iteration, -.draw)

# plot betas -------------------------------------------------------------------
colnames(coeff2_cr) <- c('police_funding','high_school', 'college')

# to long format
coeff2_cr <- coeff2_cr %>% gather(Beta, Value)


#################################### GAMMA - VIOLENT CRIME RATE ############################################################

stan_data4 <- list(n = length(y2), k = ncol(X), y = y2, X = X)

# fit
fit <- model2$sample(
  data = stan_data4,
  parallel_chains = 4,
  seed = 1
)

# diagnostics ------------------------------------------------------------------

# summary of betas
fit$summary("beta")

# additional automated diagnostics
fit$cmdstan_diagnose()

# extract parameters
all2 <- as_draws_df(fit$draws())
coeff2_vcr <- as_draws_df(fit$draws("beta"))
pred2_vcr <- as_draws_df(fit$draws("pred"))
coeff2_vcr <- coeff2_vcr %>% select(-.chain, -.iteration, -.draw)
pred2_vcr <- pred2_vcr %>% select(-.chain, -.iteration, -.draw)

# plot betas -------------------------------------------------------------------
colnames(coeff2_vcr) <- c('police_funding', 'high_school','college')

# to long format
coeff2_vcr <- coeff2_vcr %>% gather(Beta, Value)

#################################### GAMMA - PLOTS ############################################################

coeff2_cr$Type <- 'Crime Rate'
coeff2_vcr$Type <- 'Violent Crime Rate'

coeff2 <- rbind(coeff2_cr, coeff2_vcr)


# PLOTTING COEFFICIENTS
ggplot(data = coeff2, aes(x = Value, y = Beta)) +
  stat_eye(fill = "seagreen", alpha = 0.5) +
  facet_grid(Type ~ .) +
  labs(x = "Beta Value", y = "Variable") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "darksalmon", linewidth = 1) +
  theme(strip.text = element_text(size = 14))


#################################### PLOTTING PREDICTIONS ############################################################

# histogram
histogram_cr <- ggplot(data, aes(x = crime_rate)) +
  geom_histogram(aes(y = ..density..),bins = 14, alpha = 0.5 , color = 'black', fill = 'bisque', linewidth = 0.8) 


# transpose predictions
gamma_cr <- as.data.frame(t(as.matrix(pred2_cr)))

gamma_plot <- histogram_cr +
  geom_line(data = gamma_cr, aes(x = V2000), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)

# randomly select columns
gamma_cr <- gamma_cr[, sample(ncol(gamma_cr), 50)]

# extract column names 
column_names <- colnames(gamma_cr)

# loop to add density lines for each column
for (col_name in column_names) {
  gamma_plot <- gamma_plot +
    geom_line(data = gamma_cr, aes(x = .data[[col_name]]), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)
}

gamma_plot


# transpose predictions
gamma_vcr <- as.data.frame(t(as.matrix(pred1_vcr)))


# histogram
histogram_vcr <- ggplot(data, aes(x = violent_crime_rate)) +
  geom_histogram(aes(y = ..density..),bins = 22, alpha = 0.5 , color = 'black', fill = 'bisque', linewidth = 0.8) 

gamma_plot_vcr <- histogram_vcr + 
  geom_line(data = gamma_vcr, aes(x = V1), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)

# extract column names 
col_names <- colnames(gamma_vcr)

# loop to add density lines for each column
for (col_name in col_names) {
  gamma_plot_vcr <- gamma_plot_vcr +
    geom_line(data = gamma_vcr, aes(x = .data[[col_name]]), stat = 'density', color = 'darksalmon', alpha = 0.1, linewidth = 2)
}

grid.arrange(gamma_plot, gamma_plot_vcr, ncol(1))







