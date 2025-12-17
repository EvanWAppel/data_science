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

A grid search was used to find optimal parameter ranges for the selected features. The process goes like this:

1. It spins up another copy of the overall analysis setup so it can inspect the latest customer table, then prints a quick summary showing the lowest, highest, and typical values for a handful of key fields like income and risk scores (main.py (lines 388-399)).

2. Based on those summaries it defines reasonable lower and upper guardrails for each field, along with a “cost” that represents how hard it is to widen the allowed range for that field (main.py (lines 401-414)).

3. The method then builds lots of plausible combinations of these guardrails by sampling a few representative points (like quarter-percentiles) along each feature’s distribution, so it doesn’t have to brute-force every possible number (main.py (lines 419-427)).

4. For each combination it filters the data to whoever fits inside those bounds, measures how successful that filtered group tends to be and how many customers remain, and subtracts a small penalty if the bounds are too wide (reflecting the change costs) (main.py (lines 428-445)).

5. Finally, it remembers the combination with the best trade-off—high success rate, decent coverage, and not too permissive—and prints that “best recipe” so you can see which thresholds work well together (main.py (lines 437-448)).

This process provided a set of parameter ranges that the model suggests will optimize for the preferred outcome. 

## Conclusions and Caveats

The model suggests that by preferring the following ranges for these features, we can optimize for the preferred outcome of active and not delinquent accounts.

| Feature | Minimum | Maximum |
| --- | --- | --- |
| days_from_registration | 2692.0 | 3584.25 |
| income | 31200.0 | 40300.0 |
| alt_risk_score | 493.0 | 515.0 |
| alt_risk_score_2 | 527.0 | 559.0 |
| asset_score | 39.0 | 105.0 |

The biggest caveat to this analysis is that it does not suggest causality. Additionally, it was prepared without a great deal of modification of the various inputs. A true optimization would require a great deal of agonizing over every single input, the methods used to produce the analysis, and the very nature of the base dataset. 

This analysis is in no way "the answer", rather it is a starting point for a lengthy and potentially arduous analysis. 

## Future improvements

Owing to time-constraints, I half-implemented several features because the instructions were to generate suggestions, not to write an application for testing and comparing various machine learning models. However, the code is riddled with nascent ideas for future implementations. The following are some of those ideas.

- Full parameterization of the analyze class (the class that manages the model's creation). This would allow for any details about how the model should be built to be passed into the class at the time of instantiation. This would allow for automated sequential runs to compare different strategies, models, machine learning modes, et cetera.
- Streamlit application. I think that it would be an interesting exercise to set up the model in such a way that analysts can use a streamlit application GUI to tool around with a variety of models, parameters, and settings and be able to see the result in a variety of visualizations. 
- Database Caching. Something interesting that could be done here is to store previous runs of the modeler so that if someone unknowingly runs the same modeler parameters, it will bypass the modeler and simply display the results, significantly speeding up the process.
- Scaling. It's obvious that the provided dataset is a well-cleaned subset of what is likely a huge and ever expanding dataset. I think that designing the modeler to handle arbitrarily large quantities of data based on a variety of strategies from users would be an interesting challenge.
- Unit Testing. It might take some time, but it would be advantageous to design unit testing around a sample dataset. That way, it would make the application quite stable as features are added. 
- Optimization. It's obvious that the field of Data Science is enormous and continuously advancing. There are a million decisions to make across many different code libraries and it's hard for one human being to be able to make all those decisions, but if we can turn a machine on the pata-parameters (I will be disappointed if I am not the one who coined this term) then maybe we can get even more efficient, more accurate.
- Documentation. There's quite a lot of explanation that has to go into the decisions made during the optimization process and I think that it would be advantageous to design a searchable and detailed system for documentation, perhaps a wiki-like system. I recently learned of a documentation system called Dendron that I think might be a suitable candidate, but it would need significant effort to implement and train people in its use. Maybe a simple wiki will do at first.
- 