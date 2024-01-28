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
library(dplyr)
set.seed(89)

setwd("path to working directory")

# prepare the data
data <- read.csv("vo2max.csv", sep = ",")

labels <- c('male_treadmill', 'male_cycle', 'female_treadmill', 'female_cycle')

data <- data %>%
  mutate(subject = case_when(
    sex == "m" & device == "treadmill" ~ 1,
    sex == "m" & device == "cycle" ~ 2,
    sex == "f" & device == "treadmill" ~ 3,
    sex == "f" & device == "cycle" ~ 4,
    TRUE ~ NA_integer_
))


# model
model <- cmdstan_model("models/hierarchical_linear1.stan")

stan_data <- list(
  n = nrow(data),
  m = max(data$subject),
  x = data$age,
  y = data$vo2max,
  s = data$subject
)

fit <- model$sample(
  data = stan_data,
  parallel_chains = 4,
  seed = 1,
  adapt_delta = 0.97,
  max_treedepth = 20
)

# diagnostics
mcmc_trace(fit$draws("beta"))
fit$summary()

# additional automated diagnostics
fit$cmdstan_diagnose()

# samples
df <- as_draws_df(fit$draws(c("alpha", "beta", "sigma", "mu_a", "sigma_a", "mu_b", "sigma_b")))
df <- df %>% select(-.draw, -.chain, -.iteration)

# get predictions
pred <- as_draws_df(fit$draws("pred"))
pred <- pred %>% select(-.chain, -.iteration, -.draw)

pred_t <- as.data.frame(t(as.matrix(pred)))
pred_t$subject <- data$subject
pred_t$age <- data$age

################

# create a data frame with MCSE values
coeffs <- data.frame(
  b1 = df$`beta[1]`,
  b2 = df$`beta[2]`,
  b3 = df$`beta[3]`,
  b4 = df$`beta[4]`
)
colnames(coeffs) <- labels

mcse(df$`alpha[1]`)
mcse(df$`beta[1]`)
mcse(df$`alpha[2]`)
mcse(df$`beta[2]`)
mcse(df$`alpha[3]`)
mcse(df$`beta[3]`)
mcse(df$`alpha[4]`)
mcse(df$`beta[4]`)


coeffs_lf <- coeffs %>% gather(
  Variable,
  Value,
  c('male_treadmill', 'male_cycle', 'female_treadmill', 'female_cycle')
)


############################################# COMPARE SUBJECT LEVEL MEANS ###########################################


# separate predictions for each subject

one <- pred_t %>% filter(subject == 1)
one <- as.data.frame(t(as.matrix(one)))
one <- head(one, n = nrow(one) - 2)
one_lf <- one %>% gather(
  variable,
  mu
)
two <- pred_t %>% filter(subject == 2)
two <- as.data.frame(t(as.matrix(two)))
two <- head(two, n = nrow(two) - 2)
two_lf <- two %>% gather(
  variable,
  mu
)

three <- pred_t %>% filter(subject == 3)
three <- as.data.frame(t(as.matrix(three)))
three <- head(three, n = nrow(three) - 2)
three_lf <- three %>% gather(
  variable,
  mu
)

four <- pred_t %>% filter(subject == 4)
four <- as.data.frame(t(as.matrix(four)))
four <- head(four, n = nrow(four) - 2)
four_lf <- four %>% gather(
  variable,
  mu
)


df_subject <- data.frame(
  Mean = numeric(),
  Q5 = numeric(),
  Q95 = numeric(),
  Model = character(),
  fert = character()
)

# sample means
df_mu_sample <- data %>%
  group_by(subject) %>%
  summarise(mean_weight = mean(vo2max))
new_row <- data.frame(
  subject = 0,
  mean_weight = mean(data$vo2max)
)

df_mu_sample <- rbind(new_row, df_mu_sample)
df_subject <- rbind(df_subject, data.frame(
  Mean = df_mu_sample$mean_weight,
  HDI5 = df_mu_sample$mean_weight,
  HDI95 = df_mu_sample$mean_weight,
  Model = "Sample",
  fert = c('top level','male treadmill', 'male cycle', 'female treadmill', 'female cycle')
))

pred_lf <- pred %>% gather(Variable, Value)
hierarchical_mean <- mean(pred_lf$Value)
hierarchical_90_hdi <- hdi(pred_lf$Value, credMass = 0.9)
hdi_1 <- hdi(one_lf$mu, credMass = 0.9)
hdi_2 <- hdi(two_lf$mu, credMass = 0.9)
hdi_3 <- hdi(three_lf$mu, credMass = 0.9)
hdi_4 <- hdi(four_lf$mu, credMass = 0.9)
hdi5 <- c(hierarchical_90_hdi[1],hdi_1[1], hdi_2[1], hdi_3[1], hdi_4[1])
hdi95 <- c(hierarchical_90_hdi[2],hdi_1[2], hdi_2[2], hdi_3[2], hdi_4[2])
h_means <- c(hierarchical_mean, mean(one_lf$mu), mean(two_lf$mu), mean(three_lf$mu), mean(four_lf$mu))

df_subject <- rbind(df_subject, data.frame(
  Mean = h_means,
  HDI5 = hdi5,
  HDI95 = hdi95,
  Model = "Hierarchical",
  fert = c('top level', 'male treadmill', 'male cycle', 'female treadmill', 'female cycle')
))



# plot
# set model factors so the colors are the same
df_subject$Model <- factor(df_subject$Model,
                           levels = c("Hierarchical", "Sample")
)

# Specify the order of the levels
fert_order <- c('top level', 'male treadmill', 'male cycle', 'female treadmill', 'female cycle')

# Convert fert to factor with the specified order
df_subject$fert <- factor(df_subject$fert, levels = fert_order)

# plot
gg <- ggplot(
  data = df_subject,
  aes(
    x = Model,
    y = Mean,
    ymin = HDI5,
    ymax = HDI95,
    colour = Model
  )
) +
  geom_hline(yintercept = mean(data$vo2max), color = "grey75") +
  geom_point() +
  geom_errorbar(width = 0.3, linewidth = 1) +
  scale_color_brewer(palette = "Set1") +
  #ylim(0, 6) +
  facet_wrap(. ~ fert, ncol=5) +
  theme(plot.title = element_text(size = 18), axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), axis.title.x = element_text(size = 14),  
        axis.title.y = element_text(size = 14),  
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 14),
        strip.text = element_text(size = 14))
gg

ggsave("gg.png", gg, width = 10, height = 5)


 
# generating predictions
y_1 <- data.frame()
for (i in 4:11) {
  for (j in 1:nrow(df)){
    #y_i <- rnorm(1, mean = df$`alpha[1]`[j] + df$`beta[1]`[j] * i, sd = df$'sigma[1]'[j])
    y_i <- df$`alpha[1]`[j] + df$`beta[1]`[j] * i
    y_1[i-3,j] <- y_i
  }  
}


y_2 <- data.frame()
for (i in 4:11) {
  for (j in 1:nrow(df)){
    #y_i <- rnorm(1, mean = df$`alpha[2]`[j] + df$`beta[2]`[j] * i, sd = df$'sigma[2]'[j])
    y_i <- df$`alpha[2]`[j] + df$`beta[2]`[j] * i
    y_2[i-3,j] <- y_i
  }  
}

y_3 <- data.frame()
for (i in 4:11) {
  for (j in 1:nrow(df)){
    #y_i <- rnorm(1, mean = df$`alpha[3]`[j] + df$`beta[3]`[j] * i, sd = df$'sigma[3]'[j])
    y_i <- df$`alpha[3]`[j] + df$`beta[3]`[j] * i
    y_3[i-3,j] <- y_i
  }  
}

y_4 <- data.frame()
for (i in 4:11) {
  for (j in 1:nrow(df)){
    #y_i <- rnorm(1, mean = df$`alpha[4]`[j] + df$`beta[4]`[j] * i, sd = df$'sigma[4]'[j])
    y_i <- df$`alpha[4]`[j] + df$`beta[4]`[j] * i
    y_4[i-3,j] <- y_i
  }  
}

y1 <- as.data.frame(t(as.matrix(y_1)))
colnames(y1) <- c('4','5','6','7','8','9','10','11')

y1_lf <- y1 %>% gather(
  Variable,
  Value,
  '4','5','6','7','8','9','10','11'
)

y2 <- as.data.frame(t(as.matrix(y_2)))
colnames(y2) <- c('4','5','6','7','8','9','10','11')

y2_lf <- y2 %>% gather(
  Variable,
  Value,
  '4','5','6','7','8','9','10','11'
)

y3 <- as.data.frame(t(as.matrix(y_3)))
colnames(y3) <- c('4','5','6','7','8','9','10','11')
y3_lf <- y3 %>% gather(
  Variable,
  Value,
  '4','5','6','7','8','9','10','11'
)

y4 <- as.data.frame(t(as.matrix(y_4)))
colnames(y4) <- c('4','5','6','7','8','9','10','11')
y4_lf <- y4 %>% gather(
  Variable,
  Value,
  '4','5','6','7','8','9','10','11'
)

# probability for each age being the highest
prob_41 = mcse((y1$`4`>y2$`4`)&(y1$`4`>y3$`4`)&(y1$`4`>y4$`4`))
prob_41 = as.numeric(prob_41$est)
prob_42 = mcse((y2$`4`>y1$`4`)&(y2$`4`>y3$`4`)&(y2$`4`>y4$`4`))
prob_42 = as.numeric(prob_42$est)
prob_43 = mcse((y3$`4`>y2$`4`)&(y3$`4`>y1$`4`)&(y3$`4`>y4$`4`))
prob_43 = as.numeric(prob_43$est)
prob_44 = mcse((y4$`4`>y2$`4`)&(y4$`4`>y3$`4`)&(y4$`4`>y1$`4`))
prob_44 = as.numeric(prob_44$est)

prob_51 = mcse((y1$`5`>y2$`5`)&(y1$`5`>y3$`5`)&(y1$`5`>y4$`5`))
prob_51 = as.numeric(prob_51$est)
prob_52 = mcse((y2$`5`>y1$`5`)&(y2$`5`>y3$`5`)&(y2$`5`>y4$`5`))
prob_52 = as.numeric(prob_52$est)
prob_53 = mcse((y3$`5`>y2$`5`)&(y3$`5`>y1$`5`)&(y3$`5`>y4$`5`))
prob_53 = as.numeric(prob_53$est)
prob_54 = mcse((y4$`5`>y2$`5`)&(y4$`5`>y3$`5`)&(y4$`5`>y1$`5`))
prob_54 = as.numeric(prob_54$est)

prob_61 = mcse((y1$`6`>y2$`6`)&(y1$`6`>y3$`6`)&(y1$`6`>y4$`6`))
prob_61 = as.numeric(prob_61$est)
prob_62 = mcse((y2$`6`>y1$`6`)&(y2$`6`>y3$`6`)&(y2$`6`>y4$`6`))
prob_62 = as.numeric(prob_62$est)
prob_63 = mcse((y3$`6`>y2$`6`)&(y3$`6`>y1$`6`)&(y3$`6`>y4$`6`))
prob_63 = as.numeric(prob_63$est)
prob_64 = mcse((y4$`6`>y2$`6`)&(y4$`6`>y3$`6`)&(y4$`6`>y1$`6`))
prob_64 = as.numeric(prob_64$est)

prob_71 = mcse((y1$`7`>y2$`7`)&(y1$`7`>y3$`7`)&(y1$`7`>y4$`7`))
prob_71 = as.numeric(prob_71$est)
prob_72 = mcse((y2$`7`>y1$`7`)&(y2$`7`>y3$`7`)&(y2$`7`>y4$`7`))
prob_72 = as.numeric(prob_72$est)
prob_73 = mcse((y3$`7`>y2$`7`)&(y3$`7`>y1$`7`)&(y3$`7`>y4$`7`))
prob_73 = as.numeric(prob_73$est)
prob_74 = mcse((y4$`7`>y2$`7`)&(y4$`7`>y3$`7`)&(y4$`7`>y1$`7`))
prob_74 = as.numeric(prob_74$est)

prob_81 = mcse((y1$`8`>y2$`8`)&(y1$`8`>y3$`8`)&(y1$`8`>y4$`8`))
prob_81 = as.numeric(prob_81$est)
prob_82 = mcse((y2$`8`>y1$`8`)&(y2$`8`>y3$`8`)&(y2$`8`>y4$`8`))
prob_82 = as.numeric(prob_82$est)
prob_83 = mcse((y3$`8`>y2$`8`)&(y3$`8`>y1$`8`)&(y3$`8`>y4$`8`))
prob_83 = as.numeric(prob_83$est)
prob_84 = mcse((y4$`8`>y2$`8`)&(y4$`8`>y3$`8`)&(y4$`8`>y1$`8`))
prob_84 = as.numeric(prob_84$est)

prob_91 = mcse((y1$`9`>y2$`9`)&(y1$`9`>y3$`9`)&(y1$`9`>y4$`9`))
prob_91 = as.numeric(prob_91$est)
prob_92 = mcse((y2$`9`>y1$`9`)&(y2$`9`>y3$`9`)&(y2$`9`>y4$`9`))
prob_92 = as.numeric(prob_92$est)
prob_93 = mcse((y3$`9`>y2$`9`)&(y3$`9`>y1$`9`)&(y3$`9`>y4$`9`))
prob_93 = as.numeric(prob_93$est)
prob_94 = mcse((y4$`9`>y2$`9`)&(y4$`9`>y3$`9`)&(y4$`9`>y1$`9`))
prob_94 = as.numeric(prob_94$est)

prob_101 = mcse((y1$`10`>y2$`10`)&(y1$`10`>y3$`10`)&(y1$`10`>y4$`10`))
prob_101 = as.numeric(prob_101$est)
prob_102 = mcse((y2$`10`>y1$`10`)&(y2$`10`>y3$`10`)&(y2$`10`>y4$`10`))
prob_102 = as.numeric(prob_102$est)
prob_103 = mcse((y3$`10`>y2$`10`)&(y3$`10`>y1$`10`)&(y3$`10`>y4$`10`))
prob_103 = as.numeric(prob_103$est)
prob_104 = mcse((y4$`10`>y2$`10`)&(y4$`10`>y3$`10`)&(y4$`10`>y1$`10`))
prob_104 = as.numeric(prob_104$est)

prob_111 = mcse((y1$`11`>y2$`11`)&(y1$`11`>y3$`11`)&(y1$`11`>y4$`11`))
prob_111 = as.numeric(prob_111$est)
prob_112 = mcse((y2$`11`>y1$`11`)&(y2$`11`>y3$`11`)&(y2$`11`>y4$`11`))
prob_112 = as.numeric(prob_112$est)
prob_113 = mcse((y3$`11`>y2$`11`)&(y3$`11`>y1$`11`)&(y3$`11`>y4$`11`))
prob_113 = as.numeric(prob_113$est)
prob_114 = mcse((y4$`11`>y2$`11`)&(y4$`11`>y3$`11`)&(y4$`11`>y1$`11`))
prob_114 = as.numeric(prob_114$est)


prob_54 = 0


# create a dataset
age <- c(rep("age 4" , 4), rep('age 5',4), rep('age 6',4), rep('age 7',4), rep('age 8',4), rep('age 9',4), rep('age 10',4), rep('age 11',4) )
group <- rep(c("male treadmill", "male cycle", 'female treadmill', 'female cycle') , 8)
probability <- c(prob_41,prob_42,prob_43,prob_44,prob_51,prob_52,prob_53,prob_54,prob_61,prob_62,prob_63,prob_64,prob_71,prob_72,prob_73,prob_74,prob_81,prob_82,prob_83,prob_84,prob_91,prob_92,prob_93,prob_94,prob_101,prob_102,prob_103,prob_104,prob_111,prob_112,prob_113,prob_114)
probs <- data.frame(age,group,probability)
age_order <- c("age 4", "age 5", "age 6", "age 7", "age 8", "age 9", "age 10", "age 11")
probs$age <- factor(probs$age, levels = age_order)

library(viridis)
p1 <- ggplot(probs, aes(fill = group, y = probability, x = age)) +
  geom_bar(position = "fill", stat = "identity") +
  #scale_x_discrete(labels = age_order) +
  scale_fill_manual(values = c("chartreuse3", "red2", "gold", "steelblue3"))


# probability for each age being the lowest
prob_41_mean = mcse((y1$`4` < y2$`4`) & (y1$`4` < y3$`4`) & (y1$`4` < y4$`4`))
prob_41_mean = as.numeric(prob_41_mean$est)
prob_42_mean = mcse((y2$`4` < y1$`4`) & (y2$`4` < y3$`4`) & (y2$`4` < y4$`4`))
prob_42_mean = as.numeric(prob_42_mean$est)
prob_43_mean = mcse((y3$`4` < y2$`4`) & (y3$`4` < y1$`4`) & (y3$`4` < y4$`4`))
prob_43_mean = as.numeric(prob_43_mean$est)
prob_44_mean = mcse((y4$`4` < y2$`4`) & (y4$`4` < y3$`4`) & (y4$`4` < y1$`4`))
prob_44_mean = as.numeric(prob_44_mean$est)

prob_51_mean = mcse((y1$`5` < y2$`5`) & (y1$`5` < y3$`5`) & (y1$`5` < y4$`5`))
prob_51_mean = as.numeric(prob_51_mean$est)
prob_52_mean = mcse((y2$`5` < y1$`5`) & (y2$`5` < y3$`5`) & (y2$`5` < y4$`5`))
prob_52_mean = as.numeric(prob_52_mean$est)
prob_53_mean = mcse((y3$`5` < y2$`5`) & (y3$`5` < y1$`5`) & (y3$`5` < y4$`5`))
prob_53_mean = as.numeric(prob_53_mean$est)
prob_54_mean = mcse((y4$`5` < y2$`5`) & (y4$`5` < y3$`5`) & (y4$`5` < y1$`5`))
prob_54_mean = as.numeric(prob_54_mean$est)

prob_61_mean = mcse((y1$`6` < y2$`6`) & (y1$`6` < y3$`6`) & (y1$`6` < y4$`6`))
prob_61_mean = as.numeric(prob_61_mean$est)
prob_62_mean = mcse((y2$`6` < y1$`6`) & (y2$`6` < y3$`6`) & (y2$`6` < y4$`6`))
prob_62_mean = as.numeric(prob_62_mean$est)
prob_63_mean = mcse((y3$`6` < y2$`6`) & (y3$`6` < y1$`6`) & (y3$`6` < y4$`6`))
prob_63_mean = as.numeric(prob_63_mean$est)
prob_64_mean = mcse((y4$`6` < y2$`6`) & (y4$`6` < y3$`6`) & (y4$`6` < y1$`6`))
prob_64_mean = as.numeric(prob_64_mean$est)

prob_71_mean = mcse((y1$`7` < y2$`7`) & (y1$`7` < y3$`7`) & (y1$`7` < y4$`7`))
prob_71_mean = as.numeric(prob_71_mean$est)
prob_72_mean = mcse((y2$`7` < y1$`7`) & (y2$`7` < y3$`7`) & (y2$`7` < y4$`7`))
prob_72_mean = as.numeric(prob_72_mean$est)
prob_73_mean = mcse((y3$`7` < y2$`7`) & (y3$`7` < y1$`7`) & (y3$`7` < y4$`7`))
prob_73_mean = as.numeric(prob_73_mean$est)
prob_74_mean = mcse((y4$`7` < y2$`7`) & (y4$`7` < y3$`7`) & (y4$`7` < y1$`7`))
prob_74_mean = as.numeric(prob_74_mean$est)

prob_81_mean = mcse((y1$`8` < y2$`8`) & (y1$`8` < y3$`8`) & (y1$`8` < y4$`8`))
prob_81_mean = as.numeric(prob_81_mean$est)
prob_82_mean = mcse((y2$`8` < y1$`8`) & (y2$`8` < y3$`8`) & (y2$`8` < y4$`8`))
prob_82_mean = as.numeric(prob_82_mean$est)
prob_83_mean = mcse((y3$`8` < y2$`8`) & (y3$`8` < y1$`8`) & (y3$`8` < y4$`8`))
prob_83_mean = as.numeric(prob_83_mean$est)
prob_84_mean = mcse((y4$`8` < y2$`8`) & (y4$`8` < y3$`8`) & (y4$`8` < y1$`8`))
prob_84_mean = as.numeric(prob_84_mean$est)

prob_91_mean = mcse((y1$`9` < y2$`9`) & (y1$`9` < y3$`9`) & (y1$`9` < y4$`9`))
prob_91_mean = as.numeric(prob_91_mean$est)
prob_92_mean = mcse((y2$`9` < y1$`9`) & (y2$`9` < y3$`9`) & (y2$`9` < y4$`9`))
prob_92_mean = as.numeric(prob_92_mean$est)
prob_93_mean = mcse((y3$`9` < y2$`9`) & (y3$`9` < y1$`9`) & (y3$`9` < y4$`9`))
prob_93_mean = as.numeric(prob_93_mean$est)
prob_94_mean = mcse((y4$`9` < y2$`9`) & (y4$`9` < y3$`9`) & (y4$`9` < y1$`9`))
prob_94_mean = as.numeric(prob_94_mean$est)

prob_101_mean = mcse((y1$`10` < y2$`10`) & (y1$`10` < y3$`10`) & (y1$`10` < y4$`10`))
prob_101_mean = as.numeric(prob_101_mean$est)
prob_102_mean = mcse((y2$`10` < y1$`10`) & (y2$`10` < y3$`10`) & (y2$`10` < y4$`10`))
prob_102_mean = as.numeric(prob_102_mean$est)
prob_103_mean = mcse((y3$`10` < y2$`10`) & (y3$`10` < y1$`10`) & (y3$`10` < y4$`10`))
prob_103_mean = as.numeric(prob_103_mean$est)
prob_104_mean = mcse((y4$`10` < y2$`10`) & (y4$`10` < y3$`10`) & (y4$`10` < y1$`10`))
prob_104_mean = as.numeric(prob_104_mean$est)

prob_111_mean = mcse((y1$`11` < y2$`11`) & (y1$`11` < y3$`11`) & (y1$`11` < y4$`11`))
prob_111_mean = as.numeric(prob_111_mean$est)
prob_112_mean = mcse((y2$`11` < y1$`11`) & (y2$`11` < y3$`11`) & (y2$`11` < y4$`11`))
prob_112_mean = as.numeric(prob_112_mean$est)
prob_113_mean = mcse((y3$`11` < y2$`11`) & (y3$`11` < y1$`11`) & (y3$`11` < y4$`11`))
prob_113_mean = as.numeric(prob_113_mean$est)
prob_114_mean = mcse((y4$`11` < y2$`11`) & (y4$`11` < y3$`11`) & (y4$`11` < y1$`11`))
prob_114_mean = as.numeric(prob_114_mean$est)


# create a dataset
age <- c(rep("age 4" , 4), rep('age 5',4), rep('age 6',4), rep('age 7',4), rep('age 8',4), rep('age 9',4), rep('age 10',4), rep('age 11',4) )
group <- rep(c("male treadmill", "male cycle", 'female treadmill', 'female cycle') , 8)
probability <- c(
  prob_41_mean, prob_42_mean, prob_43_mean, prob_44_mean,
  prob_51_mean, prob_52_mean, prob_53_mean, prob_54_mean,
  prob_61_mean, prob_62_mean, prob_63_mean, prob_64_mean,
  prob_71_mean, prob_72_mean, prob_73_mean, prob_74_mean,
  prob_81_mean, prob_82_mean, prob_83_mean, prob_84_mean,
  prob_91_mean, prob_92_mean, prob_93_mean, prob_94_mean,
  prob_101_mean, prob_102_mean, prob_103_mean, prob_104_mean,
  prob_111_mean, prob_112_mean, prob_113_mean, prob_114_mean
)
probs2 <- data.frame(age,group,probability)
age_order <- c("age 4", "age 5", "age 6", "age 7", "age 8", "age 9", "age 10", "age 11")
probs2$age <- factor(probs2$age, levels = age_order)
probs$group <- factor(probs$group, levels = c("male treadmill", "male cycle", 'female treadmill', 'female cycle'))
probs2$group <- factor(probs2$group, levels = rev(levels(factor(probs2$group))))

p1 <- ggplot(probs, aes(fill = group, y = probability, x = age)) +
  geom_bar(position = "fill", stat = "identity") +
  #scale_x_discrete(labels = age_order) +
  scale_fill_manual(values = c("steelblue3", "red2", "gold","chartreuse3"))
  #theme(
   # legend.title = element_text(size = 12),  # Set the size of the legend title
  #  legend.text = element_text(size = 11)    # Set the size of the legend text
  #)
p2 <- ggplot(probs2, aes(fill = group, y = probability, x = age)) +
  geom_bar(position = "fill", stat = "identity") +
  scale_x_discrete(labels = age_order) +
  scale_fill_manual(values = c("steelblue3", "red2", "gold","chartreuse3"))
  #theme(
   # legend.title = element_text(size = 12),  # Set the size of the legend title
  #  legend.text = element_text(size = 11)    # Set the size of the legend text
  #)
p1 <- p1 + ylab("Highest VO2 max probability")
p2 <- p2 + ylab("Lowest VO2 max probability")
p1 <- p1 + xlab("Age")
p2 <- p2 + xlab("Age")

library(patchwork)

# Stacking plots vertically
stacked_plots <- p1/p2 + plot_layout(guides = 'collect')

# Display the stacked plots
stacked_plots



