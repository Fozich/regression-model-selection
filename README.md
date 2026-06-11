# Regression Model Selection in R

Coursework from my Data Analysis class (UNICAS, 2024/25): predict a numeric target `Y` from a course-provided dataset, choosing among competing linear specifications by validation error.

## Task

Given a training set with one target and several numeric predictors, build a regression model and submit predictions for an unseen test set.

## Method

1. **EDA** — distribution of `Y`, correlation matrix, correlations of each predictor with `Y`.
2. **80/20 train/validation split** (`caret::createDataPartition`, fixed seed).
3. **Three candidate models:**
   - baseline OLS with all predictors;
   - stepwise selection by AIC (`MASS::stepAIC`, both directions);
   - squared terms for the three predictors most correlated with `Y`, then stepwise. Squares only — a full interaction expansion would overfit at this sample size.
4. **Selection** — lowest RMSE on the held-out validation fold. The quadratic + stepwise specification won.
5. **Refit on the full training set**, predict the test set, export predictions.

Multicollinearity is checked with VIF (`car::vif`) on the baseline and the final model.

## Files

```
├── R/model_selection.R    # full workflow
└── output/predictions.csv # test-set predictions (100 obs)
```

The course dataset (`train_ch.csv`, `test_ch.csv`) is not redistributed here; the script expects the two files in the working directory.

## What I would do differently now

Single-split RMSE is noisy — k-fold cross-validation would make the model choice more stable. Stepwise selection also invalidates the usual interpretation of the final model's p-values, so I treat the summary output as descriptive, not inferential.
