import os
from typing import List, Dict
from dataclasses import dataclass
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, ParameterGrid
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler 
from xgboost import XGBClassifier
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.feature_selection import SelectFromModel 
import numpy as np
from pprint import pprint
from md2docx_python.src.md2docx_python import markdown_to_word

def md_to_docx(markdown_file = "REPORT.md" ,word_file = "report.docx") -> None:
    # Convert the Markdown file to a Word document
    try:
        markdown_to_word(markdown_file, word_file)
        print(f"Successfully converted '{markdown_file}' to '{word_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")

class analyze:
    def __init__(self,
                #  min_child_samples:Optional[int],
                #  min_split_gain:Optional[float],
                #  min_child_weight:Optional[Any],
                 data_source: str = os.path.join(os.getcwd(),"Project Data.csv"), # analysis data
                 data_dictionary_source:str = os.path.join(os.getcwd(),"Project Data Dictionary.xlsx"), # data dictionary
                 output_prefix:str = "iteration", # appends to indicate output
                 target_col:str = "success", # modeling target (1=good on C1B, 0=bad on C1B)
                 exclude_col:List[str] = ['ssn_ssn'], # columns to exclude from modeling
                 top_n_features:int = 10, # how many top features to print at the end
                 test_size:float=0.30, 
                 random_state:int=42, 
                 penalty:str="l1",
                 solver:str="liblinear",
                 max_iter:int=3000,
                 class_weight:str="balanced",
                 n_estimators:int=1000,
                 learning_rate:float=0.03,
                 num_leaves:int=64,
                 subsample:float=0.8,
                 colsample_bytree:float=0.8,
                 reg_lambda:float=1.0,
                 n_jobs:int=-1,
                 run_feature_importance_model:bool=True

                 ):
        self.data_source:str = data_source
        self.data_dictionary_source:str = data_dictionary_source
        self.output_prefix:str = output_prefix
        self.target_col:str = target_col
        self.exclude_col:List[str] = exclude_col
        self.top_n_features:int = top_n_features
        self.test_size = test_size
        self.random_state = random_state
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        # self.min_child_samples = min_child_samples
        # self.min_child_weight = min_child_weight
        # self.min_split_gain = min_split_gain
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree =colsample_bytree
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs


        self.data_table()
        print(f"Data shape: {self.data.shape}")
        self.data_dictionary_table()
        print(f"Data dictionary shape: {self.data_dictionary.shape}")
        self.data = self.create_target(self.data)
        if run_feature_importance_model == True:
            self.feature_selection() # Runs the model
    def data_table(self) -> None:
        self.data:pd.DataFrame = (
            pd.read_csv(self.data_source)
        )
    def data_dictionary_table(self) -> None:
        self.data_dictionary:pd.DataFrame = (
            pd.read_excel(self.data_dictionary_source)
        )
    def create_target(self,
                      data:pd.DataFrame
                      ) -> pd.DataFrame:
        data["success"] = (data['active_account']==True) & (data['delinquent_account']==False)
        payload:pd.DataFrame = data.drop(columns=['active_account','delinquent_account'])
        return payload
    def feature_selection(self):
        y:pd.Series = self.data[self.target_col] 
        X:pd.DataFrame = self.data.drop(columns=self.exclude_col+[self.target_col])

        # Separate numeric vs categorical future functionality
        cat_cols = [c for c in X.columns if X[c].dtype == object] 
        num_cols = [c for c in X.columns if c not in cat_cols]

        # OneHotEncoder: handles encoding of categoricals
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True) 
        except TypeError:
            # older sklearn
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
        # Imputes numericals
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
        ])
        # Imputes categoricals
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ])
        # For columnar data in particular
        pre = ColumnTransformer(
            transformers=[
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop"
        )
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y,
        )

        # --- Fit the preprocessor first, then do L1 selection on the preprocessed matrix --- 
        pre.fit(X_train, y_train) 
        Xtr_pre = pre.transform(X_train) 
        Xte_pre = pre.transform(X_test)

        # L1 logistic for feature selection (wrapped inside SelectFromModel; no pipeline step for plain LogisticRegression) 
        lasso = LogisticRegression(
            # Fixed parameters TODO: pipe in from class object.
            penalty="l1",
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced",
        )

        selector = SelectFromModel(
            estimator=lasso, 
            threshold="median",
            ) 
        selector.fit(
            Xtr_pre, 
            y_train,
            )

        Xtr_sel = selector.transform(Xtr_pre)
        Xte_sel = selector.transform(Xte_pre)
        try:
            print(f"Selected features (post-L1): {Xtr_sel.shape[1]}")
        except Exception as e:
            print(f"Failed to print due to {e}")

        # Final model: LightGBM (fallback to RF if LGBM not installed)
        try:
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                n_estimators=1000,
                learning_rate=0.03,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                # Helps find more splits, reduce warnings
                    # This works by the way!
                min_child_samples=5,
                min_split_gain=0.0,
                min_child_weight=1e-3,
            )
            model_name = "LightGBM"
        except Exception:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
            model_name = "RandomForest"

        clf.fit(Xtr_sel, y_train)
        proba = clf.predict_proba(Xte_sel)[:, 1]

        roc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba) 
        print(f"\n{model_name} ROC AUC: {roc:.4f} | PR AUC: {pr_auc:.4f}")

        print("\n= Report @ 0.5 =")
        pred = (proba >= 0.5).astype(int)
        print(classification_report(y_test, pred, digits=3)) 
        print("Confusion matrix @ 0.5:\n", confusion_matrix(y_test, pred))

        # Build feature names from fitted preprocessor 
        num_feature_names = num_cols[:]  # numeric pass-through names

        cat_feature_names = []
        if len(cat_cols):
            oh = pre.named_transformers_["cat"].named_steps["onehot"]
            cat_feature_names = list(oh.get_feature_names_out(cat_cols))

        all_pre_names = num_feature_names + cat_feature_names

        # Mask from selector
        sel_mask = selector.get_support()
        selected_feature_names = [n for n, keep in zip(all_pre_names, sel_mask) if keep]

        # Importances
        importances = getattr(clf, "feature_importances_", None) 
        if importances is not None and len(importances) == len(selected_feature_names):
            order = np.argsort(importances)[::-1]
            print(f"\nTop {self.top_n_features} predictors:")
            for i in order[:self.top_n_features]:
                print(f"{selected_feature_names[i]:45s}  {importances[i]:.6f}")
        else:
            print("\n Feature name mismatch")
    
    def grid_search(self):
        analysis:analyze = analyze(run_feature_importance_model=False)
        # Gonna take a look at the max/min on these features to help set constraints
        important_stats = analysis.data[
            [
                "days_from_registration",
                "income",
                "alt_risk_score",
                "alt_risk_score_2",
                "asset_score",
            ]
        ].describe()
        print(important_stats)
        constraints:Dict[str,List[float]] = {
            "days_from_registration":[1,5916],
            "income":[20_000,100_000],
            "alt_risk_score":[408,702],
            "alt_risk_score_2":[391,687],
            "asset_score":[1,9_003],
        }

        change_cost:Dict[str,float|int] = {
            "days_from_registration":1,
            "income":1,
            "alt_risk_score":1,
            "alt_risk_score_2":1,
            "asset_score":1,
        }

        #data:pd.DataFrame = self.data
        df = analysis.data.copy()  # pandas DataFrame from the analyze class

        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        grid_dict = {}
        for feature, (lo, hi) in constraints.items():
            series = df[feature].dropna()
            lowers = np.clip(np.quantile(series, quantiles[:3]), lo, hi)
            uppers = np.clip(np.quantile(series, quantiles[2:]), lo, hi)
            grid_dict[f"{feature}__min"] = lowers
            grid_dict[f"{feature}__max"] = uppers

        

        best = None
        for params in ParameterGrid(grid_dict):
            bounds = {f: (params[f"{f}__min"], params[f"{f}__max"]) for f in constraints}
            if any(lo >= hi for lo, hi in bounds.values()):
                continue
            score, rows = self.evaluate(bounds,df)
            cost = sum(change_cost[f] * (bounds[f][1] - bounds[f][0]) for f in bounds)
            objective = score - 1e-4 * cost  # tune the penalty factor
            if not best or objective > best["objective"]:
                best = {"bounds": bounds, "score": score, "rows": rows, "cost": cost, "objective": objective}

        pprint(best)
    def evaluate(self,bounds,df):
            mask = np.ones(len(df), dtype=bool)
            for col, (lo, hi) in bounds.items():
                mask &= df[col].between(lo, hi)
            subset = df.loc[mask]
            if subset.empty:
                return 0.0, 0
            return subset["success"].mean(), len(subset)