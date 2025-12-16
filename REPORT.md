# Data Science Project

A dataset of approved credit card applications was provided with the request that a parsimonious solution be sought for optimizing active accounts and minimizing delinquent accounts.

A model selection analysis typically includes the following steps:

- Cleaning the data to prepare it for analysis
- Feature selection to identify important features to the preferred outcome
- Hyperparameter search to identify optimal combinations of important features for the preferred outcome.

## Cleaning the data

The data provided was already particularly clean, which is something that is unlikely in the real world. Typically, analysts can expect to spend 90% of their time preparing data for any task. So, it's fortunate that it was so clean.

There were two noteably cleaning steps if they might be called that:
1. A new column was created that combined the active_account and delinquent_account columns to aid in the feature selection.
2. The ssn_ssn column was removed because it was constant and would not have any effect on the model. 
   - I find it notable that this is even a column. Perhaps there are populations where people have more than one Social Security Number?
3. Imputing values that are missing in the columns.

## Feature Selection

It's necessary to run feature selection to be able to identify which features in the dataset are most likely to contribute to the preferred outcome (active and not delinquent account). 

To lower the possible compute cost, it might be prudent to pare down the fields for analysis, but since there are only about 30 fields, and the data is relatively clean, this won't be as necessary as it might be on a 1,000 field dataset.

In a general sense the feature selection process goes like this:

1. Set Up Data: When model() runs, it takes the prepared table of customers, pulls out the column that marks success, and keeps every other column as potential signals to learn from (main.py (lines 238-244)).

2. Clean Columns: It looks at each column and decides whether it holds numbers or categories, then creates two cleaning recipes—one that fills missing numbers and scales them, and one that fills missing category labels and turns them into marker columns so the computer can read them easily (main.py (lines 242-269)).

3. Train/Test Split & Feature Filter: The data is split into a training portion and a testing portion to keep the evaluation honest. The cleaning steps are fitted on the training part and applied to both splits. A slimmed-down logistic model then figures out which inputs matter most and drops the rest so the final model only focuses on the strongest signals (main.py (lines 270-308)).

4. Main Predictor: It tries to train a LightGBM model—think of many small decision rules working together—and if that package isn’t available it falls back to a Random Forest (a similar ensemble of decision rules). After training, it produces success probabilities for the held-out testing set and reports standard quality scores plus a confusion matrix so you can see how well it did (main.py (lines 309-351)).

5. Explain Results: Finally, it reconstructs the actual column names that survived the filtering step and, if the model exposes importances, prints a short leaderboard of the top drivers so you can see which inputs influence the predictions the most (main.py (lines 352-375)).

This produced the following top features by importance:
| Feature | Importance |
| --- | --- |
| days_from_registration | 11043 |
| income | 10689 |
| alt_risk_score | 9605 |
| alt_risk_score_2 | 8171 |
| asset_score | 6735 |
| stability_score | 4163 |
| dti_score | 3961 |
| Num_Bk_Accts | 2918|
| seg_id | 1882 |
| bk_accts_ssn | 1067 |

Therefore, the model suggests that for the dataset, days_from_registration, income, alt_risk_score, alt_risk_score_2, and asset_score are the most important features in influencing the preferred outcome.

## Hyperparameter Search

A grid search was used to find optimal 

## Conclusions and Caveats

