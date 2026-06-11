# Regression model selection on a course dataset
# Original: Data Analysis midterm, UNICAS 2024/25 (student 97549)
#
# Workflow: EDA -> 80/20 train/validation split -> three candidate models
# -> pick the lowest validation RMSE -> refit on all data -> predict test set.

# Packages
pkgs <- c("dplyr", "ggplot2", "car", "corrplot", "caret", "MASS")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install) > 0) install.packages(to_install)
invisible(lapply(pkgs, library, character.only = TRUE))

rm(list = ls())
set.seed(123)  # reproducible split and stepwise path

# Data (course-provided files, not redistributed in this repo)
train <- read.csv("train_ch.csv")
test  <- read.csv("test_ch.csv")

# Drop the index column if the first column is just 1..n
if (all(train[[1]] == 1:nrow(train))) {
  train <- train[, -1]
  test  <- test[, -1]
}

# EDA -------------------------------------------------------------------
print(str(train))
print(summary(train))

ggplot(train, aes(x = Y)) + geom_histogram(bins = 30)
ggplot(train, aes(y = Y)) + geom_boxplot()

# Correlation matrix; correlations with Y guide variable choice later
num_train <- train[, sapply(train, is.numeric)]
cormat <- cor(num_train, use = "pairwise.complete.obs")
corrplot(cormat, method = "color")

cor_with_y <- sort(cormat[, "Y"], decreasing = TRUE)
print(cor_with_y)

# Train / validation split (80/20) --------------------------------------
idx <- createDataPartition(train$Y, p = 0.8, list = FALSE)
tr <- train[idx, ]
va <- train[-idx, ]

rmse <- function(y, yhat) sqrt(mean((y - yhat)^2))

# Model 1: baseline with all predictors
m_all <- lm(Y ~ ., data = tr)
rmse_all  <- rmse(va$Y, predict(m_all, newdata = va))
adjr2_all <- summary(m_all)$adj.r.squared
vif_all   <- vif(m_all)  # multicollinearity check

# Model 2: stepwise AIC starting from the baseline
m_step <- stepAIC(m_all, direction = "both", trace = FALSE)
rmse_step  <- rmse(va$Y, predict(m_step, newdata = va))
adjr2_step <- summary(m_step)$adj.r.squared

# Model 3: moderate nonlinearity — squared terms for the three predictors
# most correlated with Y, then stepwise again. Squares only; a full
# interaction expansion would overfit a dataset of this size.
pred_names <- setdiff(names(cor_with_y), "Y")
top3 <- names(sort(abs(cor_with_y[pred_names]), decreasing = TRUE))[1:3]

quad_terms <- paste0("I(", top3, "^2)", collapse = " + ")
form_nl <- as.formula(paste("Y ~ . +", quad_terms))

m_nl      <- lm(form_nl, data = tr)
m_nl_step <- stepAIC(m_nl, direction = "both", trace = FALSE)

rmse_nl  <- rmse(va$Y, predict(m_nl_step, newdata = va))
adjr2_nl <- summary(m_nl_step)$adj.r.squared

# Compare on validation RMSE --------------------------------------------
rmse_table <- data.frame(
  Model = c("Baseline (all)", "Stepwise AIC", "Quadratic terms (top 3) + stepwise"),
  RMSE  = c(rmse_all, rmse_step, rmse_nl),
  AdjR2 = c(adjr2_all, adjr2_step, adjr2_nl)
)
print(rmse_table)

best_name <- rmse_table$Model[which.min(rmse_table$RMSE)]
print(best_name)

if (best_name == "Baseline (all)")  final_model <- m_all
if (best_name == "Stepwise AIC")    final_model <- m_step
if (best_name == "Quadratic terms (top 3) + stepwise") final_model <- m_nl_step

print(formula(final_model))
print(summary(final_model))
print(vif(final_model))

# Refit the chosen specification on ALL training data, then predict test
final_model_full <- lm(formula(final_model), data = train)

pred <- predict(final_model_full, newdata = test)
write.csv(data.frame(Y = pred), "output/predictions.csv", row.names = FALSE)
