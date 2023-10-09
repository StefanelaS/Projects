
library(ggplot2)
install.packages("caret")
library(caret)
install.packages("pagedown")
library(pagedown)
library(rpart)


# Function to generate a toy dataset
toy_data <- function(n, seed=NULL){
  set.seed(seed)
  x <- matrix(rnorm(8 * n), ncol = 8)
  z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
  y <- runif(n) > 1 / (1 + exp(-z))
  return (data.frame(x = x, y = y))
}

# Function to calculate log-loss
log_loss <- function(y, p){
  -(y * log(p) + (1 - y) * log(1 - p))
}  

df_dgp <- toy_data(100000, 0)


  
# Function that fits logistic regression model
fit_glm <- function(x, y) {
  model <- glm(y ~ ., data = x, family = binomial())
  return(model)
}  

# Function that fits decision tree
fit_dt <- function(x,y){
  model <- rpart(y ~ ., data=x, method="class", control=rpart.control(minsplit=1, minbucket=1, maxdepth = Inf))
  return(model)
}
   
# Function that predicts with logistic regression model
predict_glm<- function(model, data) {
  x_test <- data[, 1:8]
  y_pred <- predict(model, newdata = x_test, type = "response")
  return(y_pred)
}

# Function that computes risk
compute_risk <- function(data, probs) {
  risk <- mean(log_loss(matrix(data$y), matrix(probs)))
  return(risk)
}

# Function that computes standard error 
compute_standard_error <- function(estimates){
  n <- length(estimates)
  mean_estimates <- sum(estimates) / n
  variance <- sum((estimates - mean_estimates) ** 2) / n
  #variance <- sum((estimates - mean_estimates) ** 2) / (n-1)
  return (sqrt(variance / n))
}


df1 <- toy_data(50, 0) 
model_h <- fit_glm(df1[,1:8], df1$y)
probs_h <- predict_glm(model_h, df_dgp)
true_risk <- compute_risk(df_dgp, probs_h)
print(paste('True risk proxy:', true_risk))

# Estimating risk on toy data sets
estimated_risks <- c()
standard_errors <- c()
ci <- 0
for (i in 1:1000){
  set.seed(i)
  df <- toy_data(50, i**2)
  probs <- predict_glm(model_h, df)
  #estimated_risks[i] <- compute_risk(df, probs)
  log_losses <- log_loss(matrix(df$y), matrix(probs))
  estimated_risks[i] <- mean(log_losses)
  standard_errors[i] <- compute_standard_error(log_losses)
  # est_risk_ses[i] <- se(log_losses)
  #standard_error <- compute_standard_error(estimated_risk)
  #standard_errors <- append(standard_errors, standard_error)
  # Compute the 95% confidence interval
  ci_lower <- estimated_risks[i] - 1.96*standard_errors[i]
  ci_upper <- estimated_risks[i] + 1.96*standard_errors[i]
  # Check if the 95% confidence interval contains the true risk
  if (ci_lower <= true_risk & ci_upper >= true_risk) {
    ci <- ci + 1
  }
}

diffs <- estimated_risks - true_risk
mean_diff <- mean(diffs)
print(paste('Mean difference:', mean_diff))

# Generating a density plot of the differences
df_diffs <- data.frame(diffs=diffs)
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + xlab('est_risk - true_risk')
p

# Calculating true risk of always making 0.5-0.5 predictions
baseline_probs <- rep(0.5, 100000)
baseline_risk <- compute_risk(df_dgp, baseline_probs)
print(paste(' 0.5-0.5 baseline true risk:', baseline_risk))

# Computing median standard error
median_se <- median(standard_errors)
print(paste('Median standard error:', median_se))

ci_perc <- ci / 1000
print(paste('Percentage of 95CI that contain the true risk proxy:', ci_perc))

diffs <- c()
for (i in 1:50){
  df1 <- toy_data(50, i**2) 
  df2 <- toy_data(50, i**3) 
  df2 <- rbind(df1, df2)
  model_h1 = fit_glm(df1[,1:8], df1$y)
  model_h2 = fit_glm(df2[,1:8], df2$y)
  # Predicting on the huge dataset
  probs_h1 <- predict_glm(model_h1, df_dgp)
  probs_h2 <- predict_glm(model_h2, df_dgp)
  # Calculating the true risk on the huge dataset
  true_risk_h1 <- compute_risk(df_dgp, probs_h1)
  true_risk_h2 <- compute_risk(df_dgp, probs_h2)
  diffs[i] = true_risk_h1 - true_risk_h2
}
summary(diffs)

df0 <- toy_data(100, 0)
model_h0 <- fit_glm(df0[,1:8], df0$y)
probs_h0 <- predict_glm(model_h0, df_dgp)
true_risk_h0 <- compute_risk(df_dgp, probs_h0)

# Estimating risk on different train-test splits
estimated_risks <- c()
standard_errors <- c()
ci <- 0
for (i in 1:1000){
  set.seed(i)
  train_indices <- sample(1:100, 50)
  df_train <- df0[train_indices,]
  df_test <- df0[-train_indices,]
  model <- fit_glm(df_train[, 1:8], df_train$y)
  probs <- predict_glm(model, df_test)
  #estimated_risks[i] <- compute_risk(df, probs)
  log_losses <- log_loss(matrix(df_test$y), matrix(probs))
  estimated_risks[i] <- mean(log_losses)
  standard_errors[i] <- compute_standard_error(log_losses)
  # est_risk_ses[i] <- se(log_losses)
  #standard_error <- compute_standard_error(estimated_risk)
  #standard_errors <- append(standard_errors, standard_error)
  # Compute the 95% confidence interval
  ci_lower <- estimated_risks[i] - 1.96*standard_errors[i]
  ci_upper <- estimated_risks[i] + 1.96*standard_errors[i]
  # Check if the 95% confidence interval contains the true risk
  if (ci_lower <= true_risk & ci_upper >= true_risk) {
    ci <- ci + 1
  }
}

print(paste('True risk proxy:', true_risk_h0))

# Generating a density plot of the differences
diffs <- estimated_risks - true_risk_h0
df_diffs <- data.frame(diffs=diffs)
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + xlab('estimated_risk - true_risk')
p
mean_diff <- mean(diffs)
print(paste('Mean difference:', mean_diff))

# Computing median standard error
median_se <- median(standard_errors)
print(paste('Median standard error:', median_se))

ci_perc <- ci / 1000
print(paste('Percentage of 95CI that contain the true risk proxy:', ci_perc*100))

estimators <- c("2-fold", "4-fold", "10-fold", "10-fold-20-rep","LOOCV")

# Function that computes standard error 
kfold_cv <- function(estimator, data){
  n_fold = 2
  n <- nrow(data)
  data <- data[sample(n), ]
  fold_indices <- createMultiFolds(data, k = n_fold)
  return(fold_indices)
}

df <- toy_data(100, 0)
model <- fit_glm(df[,1:8], df$y)
probs <- predict_glm(model, df_dgp)
true_risk <- compute_risk(df_dgp, probs)


split_data <- function(num_folds, data){
  n <- nrow(data)
  fold_size <- floor(n/num_folds)
  data <- data[sample(n), ]
  folds <- c()
  for (i in 1:num_folds){
    start <- (i - 1) * fold_size + 1
    end <- start + fold_size - 1
    folds[[i]] <- data[start:end,  ]
  }
  return(folds)
}


cross_validation <- function(k){
  diffs <- c()
  standard_errors <- c()
  ci <- 0
  for (i in 1:500){
    # computing true risk proxy
    df <- toy_data(100, i)
    model <- fit_glm(df[,1:8], df$y)
    probs <- predict_glm(model, df_dgp)
    true_risk <- compute_risk(df_dgp, probs)
    # splitting the data
    folds <- split_data(k, df)
    estimated_risk <- c()
    standard_error <- c()
    for (j in 1:k){
      df_test <- folds[[j]]
      df_train <- folds[-j]
      df_train <- do.call(rbind, df_train)
      model <- fit_glm(df_train[, 1:8], df_train$y)
      probs <- predict_glm(model, df_test)
      log_losses <- log_loss(matrix(df_test$y), matrix(probs))
      estimated_risk[j] <- mean(log_losses)
      standard_error[j] <- compute_standard_error(log_losses)
    }
    mean_est_risk <- sum(estimated_risk)/k
    mean_se <- sum(standard_error)/k
    standard_errors[i] <- mean_se
    diffs[i] <- mean_est_risk - true_risk
    
    # Compute the 95% confidence interval
    ci_lower <- mean_est_risk - qnorm(0.975)*mean_se
    ci_upper <- mean_est_risk + qnorm(0.975)*mean_se
    # Check if the 95% confidence interval contains the true risk
    if (ci_lower <= true_risk & ci_upper >= true_risk) {
      ci <- ci + 1
    }
  }
  median_se <- median(standard_errors)
  ci_perc <- ci / 500
  result <- list(differences = diffs, median_se = median_se, ci_perc = ci_perc)
  return(result)
}
# Function to generate a toy dataset
toy_data <- function(n, seed=NULL){
  set.seed(seed)
  x <- matrix(rnorm(8 * n), ncol = 8)
  z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
  y <- runif(n) > 1 / (1 + exp(-z))
  return (data.frame(x = x, y = y))
}

# Function to calculate log-loss
log_loss <- function(y, p){
  -(y * log(p) + (1 - y) * log(1 - p))
}  

df_dgp <- toy_data(100000, 0)

# Function that fits decision tree
fit_dt <- function(x,y){
  model <- rpart(y ~ ., data=x, method="class", control=rpart.control(minsplit=2, minbucket=1))
  return(model)
}
cross_validation <- function(k){
  diffs <- c()
  standard_errors <- c()
  ci <- 0
  for (i in 1:500){
    # computing true risk proxy
    df <- toy_data(100000, i)
    n <- nrow(df)
    model <- fit_dt(df[,1:8], df$y)
    x_dgp <- df_dgp[, 1:8]
    cat(length(x_dgp))
    probs <- predict(model, newdata = x_dgp, type='prob')
    cat(probs)
    cat(length(probs))
    true_risk <- compute_risk(df_dgp, probs)
    # splitting the data
    folds <- split_data(k, df)
    log_losses <- c()
    for (j in 1:k){
      df_test <- folds[[j]]
      df_train <- folds[-j]
      df_train <- do.call(rbind, df_train)
      df_test <- rbind(df_test)
      model <- fit_dt(df_train[, 1:8], df_train$y)
      x_test <- df_test[, 1:8]
      probs <- predict(model, newdata = x_test, type='prob')
      log_losses <- c(log_losses,log_loss(matrix(df_test$y), matrix(probs)))
    }
    
    estimated_risk <- mean(log_losses)
    standard_error <- compute_standard_error(log_losses)
    standard_errors[i] <- standard_error
    diffs[i] <- estimated_risk - true_risk
    # Compute the 95% confidence interval
    ci_lower <- estimated_risk - qnorm(0.975)*standard_error
    ci_upper <- estimated_risk + qnorm(0.975)*standard_error
    # Check if the 95% confidence interval contains the true risk
    if (ci_lower <= true_risk & ci_upper >= true_risk) {
      ci <- ci + 1
      }
  }
  median_se <- median(standard_errors)
  ci_perc <- ci / 500
  result <- list(differences = diffs, median_se = median_se, ci_perc = ci_perc)
  return(result)
}

loocv <- cross_validation(99)
set.seed(0)
estimators <- c("2-fold", "4-fold", "10-fold", "10-fold-20-rep","LOOCV")
cat("Estimator:",estimators[2])

cv_2_fold <- cross_validation(2)
cat("Estimator:","2-fold")
diffs = cv_2_fold[[1]]
mean_diff <- mean(cv_2_fold[[1]])
cat('Mean difference:', mean_diff)
cat('Median standard error:', cv_2_fold[[2]])
cat('Percentage of 95CI that contain the true risk proxy:', cv_2_fold[[3]]*100)
se_estimates = compute_standard_error(estimated_risks)
cat(se_estimates)

df_diffs <- data.frame(diffs=diffs)
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '2-fold cross-validation', x='estimated_risk - true_risk') + ylim(0,5)
p
cv_4_fold <- cross_validation(4)

cat("ESTIMATOR:","4-fold")
mean_diff <- mean(cv_4_fold[[1]])
cat('Mean difference:', mean_diff)
cat('Median standard error:', cv_4_fold[[2]])
cat('Percentage of 95CI that contain the true risk proxy:', cv_4_fold[[3]]*100)
df_diffs <- data.frame(diffs=cv_4_fold[[1]])
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '4-fold cross-validation', x = 'estimated_risk - true_risk') + ylim(0.5)
p


cv_10_fold <- cross_validation(10)

cat("ESTIMATOR:","10-fold")
cv_10_fold <- cross_validation(10)
mean_diff <- mean(cv_10_fold[[1]])
cat('Mean difference:', round(mean_diff, 4))
cat('Median standard error:', round(cv_10_fold[[2]], 4))
cat('Percentage of 95CI that contain the true risk proxy:', cv_10_fold[[3]]*100)

df_diffs <- data.frame(diffs=cv_10_fold[[1]])
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '10-fold cross-validation', x = 'estimated_risk - true_risk') + ylim(0,6) + xlim(-1,7)
p

loocv <- cross_validation(1)
mean_diff <- mean(loocv[[1]])
cat('Mean difference:', mean_diff)
cat('Median standard error:', loocv[[2]])
cat('Percentage of 95CI that contain the true risk proxy:', loocv[[3]])

# Generating a density plot of the differences
df_diffs <- data.frame(diffs=diffs)
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + xlab('estimated_risk - true_risk')
p
mean_diff <- mean(diffs)
cat('Mean difference:', mean_diff)

# Computing median standard error
median_se <- median(standard_errors)
cat('Median standard error:', median_se)

ci_perc <- ci / 500
cat('Percentage of 95CI that contain the true risk proxy:', ci_perc*100)

# Function that performs cross-validation of k-folds
cross_validation <- function(k){
  diffs <- c()
  standard_errors <- c()
  #estimated_risk <- c()
  ci <- 0
  for (i in 1:200){
    # computing true risk proxy
    df <- toy_data(10000, i)
    n <- nrow(df)
    model <- fit_glm(df[,1:8], df$y)
    probs <- predict_glm(model, df_dgp)
    true_risk <- compute_risk(df_dgp, probs)
    # splitting the data
    folds <- split_data(k, df)
    log_losses <- c()
    for (j in 1:k){
      df_test <- folds[[j]]
      df_train <- folds[-j]
      df_train <- do.call(rbind, df_train)
      df_test <- rbind(df_test)
      model <- fit_glm(df_train[, 1:8], df_train$y)
      probs <- predict_glm(model, df_test)
      log_losses <- c(log_losses,log_loss(matrix(df_test$y), matrix(probs)))
    }
    estimated_risk <- mean(log_losses)
    standard_error <- compute_standard_error(log_losses)
    standard_errors[i] <- standard_error
    diffs[i] <- estimated_risk - true_risk
    # Compute the 95% confidence interval
    ci_lower <- estimated_risk - qnorm(0.975)*standard_error
    ci_upper <- estimated_risk + qnorm(0.975)*standard_error
    # Check if the 95% confidence interval contains the true risk
    if (ci_lower <= true_risk & ci_upper >= true_risk) {
      ci <- ci + 1
    }
  }
  median_se <- median(standard_errors)
  ci_perc <- ci / 200
  result <- list(differences = diffs, median_se = median_se, ci_perc = ci_perc)
  return(result)
}

cat("ESTIMATOR:","2-fold")
cv_2_fold <- cross_validation(2)
mean_diff <- mean(cv_2_fold[[1]])
cat('Mean difference:', round(mean_diff, 4))
cat('Median standard error:', round(cv_2_fold[[2]], 4))
cat('Percentage of 95CI that contain the true risk proxy:', cv_2_fold[[3]]*100)

cat("ESTIMATOR:","4-fold")
cv_4_fold <- cross_validation(4)
mean_diff <- mean(cv_4_fold[[1]])
cat('Mean difference:', round(mean_diff, 4))
cat('Median standard error:', round(cv_4_fold[[2]], 4))
cat('Percentage of 95CI that contain the true risk proxy:', cv_4_fold[[3]]*100)

cat("ESTIMATOR:","10-fold")
cv_10_fold <- cross_validation(10)
mean_diff <- mean(cv_10_fold[[1]])
cat('Mean difference:', round(mean_diff, 4))
cat('Median standard error:', round(cv_10_fold[[2]], 4))
cat('Percentage of 95CI that contain the true risk proxy:', cv_10_fold[[3]]*100)


# Generating density plots of the differences
df_diffs <- data.frame(diffs=cv_2_fold[[1]])
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '2-fold cross-validation', x = 'estimated_risk - true_risk') + ylim(0,6) + xlim(-1,7)
p

df_diffs <- data.frame(diffs=cv_4_fold[[1]])
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '4-fold cross-validation', x = 'estimated_risk - true_risk') + ylim(0,6) + xlim(-1,7)
p

df_diffs <- data.frame(diffs=cv_10_fold[[1]])
p <- ggplot(df_diffs, aes(x=diffs)) + geom_density() + labs(title = '10-fold cross-validation', x = 'estimated_risk - true_risk') + ylim(0,6) + xlim(-1,7)
p


