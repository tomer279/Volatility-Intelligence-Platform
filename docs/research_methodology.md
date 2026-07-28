# Research Methodology (Draft)

## Target

- Forward 5-trading-day close-to-close realized volatility: `target_rv_cc_5d`

- Stored non-annualized

## Primary metric

- QLIKE (lower is better)

- Secondary: MSE, MAE

## Validation

- Expanding walk-forward

- Embargo ≥ 5 trading days between train and test

- Models refit each fold using training data only

## Baselines

- Historical mean

- EWMA (frozen at end of train)

- HAR-RV OLS on trailing RV features