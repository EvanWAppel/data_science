import os
import sys
import polars as pl
from typing import List, Dict, Callable, Any
from dataclasses import dataclass
import variables
import logging
logger = logging.getLogger(__name__)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# Create data dictionary table object
@dataclass
class Entry:
    """ Describes an entry in the data dictionary.
    """    
    Field:str
    Description:str
    Classification:str
@dataclass
class Feature:
    """ Categorizes attributes
    """    
    target:str
    immutable:str
    actionable:str
    derived:str

class DataDictionary:
    """
        A data dictionary object to group the functionality and data of the reference material.
    """    
    def __init__(self):
        self.entries:List[Entry] = [Entry(**row_dict) for row_dict in self.fetch_dictionary().to_dicts()]
    def fetch_dictionary(self,source:str=variables.data_dictionary) -> pl.DataFrame:
        """Gets the data dictionary. Defaults to the one in the current working directory.

        Returns:
            pl.DataFrame: _description_
        """        
        return (pl.read_excel(source,has_header=True,columns=['Field','Description'])
                .select(pl.col('Field','Description'))
                .with_columns(pl.lit("blank").alias('Classification')))
    def assign_classification(self, subject:str,object:str) -> None:
        """ An assign method. It takes a field name and gives it an attribute.

        Args:
            subject (str): _description_
            object (str): _description_
        """        
        for entry in self.entries:
            if entry.Field == subject:
                entry.Classification = object
    def apply_assign(self,list_of_dicts:List[Dict[str,str]],assignment_function:Callable[[str,str],None]) -> None:
        """ Takes an assign method and applies it over a list of dictionaries. 

        Args:
            list_of_dicts (List[Dict[str,str]]): _description_
            assignment_function (Callable[[str,str],None]): _description_

        Example Usage:

        l:List[Dict[str,str]] = [
            {'field':'active_account','classification':'target'},
            {'field':'delinquent_account','classification':'target'},
            {'field':'ident_monitor_opt','classification':'actionable'},
            {'field':'Num_Bk_Accts','classification':'actionable'},
            {'field':'income','classification':'actionable'},
            {'field':'bk_accts_ssn','classification':'actionable'},
            {'field':'cells_ssn','classification':'actionable'},
            {'field':'dls_ssn','classification':'actionable'},
            {'field':'emails_ssn','classification':'actionable'},
            {'field':'hmphones_ssn','classification':'actionable'},
            {'field':'addrs_ssn','classification':'actionable'},
            {'field':'ssn_ssn','classification':'actionable'},
            {'field':'zips_ssn','classification':'actionable'},
            {'field':'empl_ssn_6Mo','classification':'actionable'},
            {'field':'pday_inq_15days','classification':'actionable'},
            {'field':'dti_score','classification':'actionable'},
            {'field':'days_from_registration','classification':'actionable'},
            {'field':'days_from_login','classification':'actionable'},
            {'field':'asset_score','classification':'actionable'},
            {'field':'alt_risk_score','classification':'actionable'},
            {'field':'seg_id','classification':'actionable'},
            {'field':'stability_score','classification':'actionable'},
            {'field':'max_bkaccts','classification':'actionable'},
            {'field':'auto_inq_1dy','classification':'actionable'},
            {'field':'auto_inq_7dy','classification':'actionable'},
            {'field':'auto_inq_72hr','classification':'actionable'},
            {'field':'cc_inq_72hr','classification':'actionable'},
            {'field':'cc_inq_10dy','classification':'actionable'},
            {'field':'pl_inq_72hr','classification':'actionable'},
            {'field':'pl_inq_90dy','classification':'actionable'},
            {'field':'alt_risk_score_2','classification':'actionable'},

        ]

        dictionary.apply_assign(l,dictionary.assign_classification)
            
        """        
        [assignment_function(i['field'],i['classification']) for i in list_of_dicts]
    
    def frame_dictionary(self) -> pl.DataFrame:
        """Prints and returns a dataframe representation of the data dictionary.

        Returns:
            pl.DataFrame: _description_
        """        
        field_l:List[str] = []
        description_l:List[str] = []
        classification_l:List[str] = []
        for i in self.entries:
             field_l.append(i.Field)
             description_l.append(i.Description)
             classification_l.append(i.Classification)
        df = pl.DataFrame(
            {
                'Field':field_l,
                'Description':description_l,
                'Classification':classification_l,
            }
        )
        #print(df)
        return df

class Dataset:
    """
        A dataset object to group the functionality and data of the reference material.
    """    
    def __init__(
            self,
            source:str=variables.project_data
            ):
        self.source:str = source
        self.data:pl.DataFrame = pl.read_csv(source)
        self.success_label()
        # self.remove_nans() # prevents a mismatch error during the modeling phase
    def preview(self,n:int=5) -> pl.DataFrame:
        """ Previews the dataset.

        Args:
            n (int, optional): Number of rows to preview. Defaults to 5.

        Returns:
            pl.DataFrame: preview dataset
        """        
        return self.data.head(n)    
    
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
            self.model() # Runs the model
        # Additional cleaning of data columns? Unnecessary? Seems the dataset is currently pretty clean.
        # Keep only rows that have a target for training - David
            # we should only have success == 1 for the training of the model
        # NOTE: STEP 2 — DICTIONARY-DRIVEN CLEANING (ONE DICTIONARY) 
            # May not need to happen here.
        # NOTE: STEP 3 — MODEL: L1 feature selection → LightGBM (or RF) → Top N features 
            # start here!
                # NOTE: we need to also do grid-search
                    # Or, rather, we need to iterate through all the options in the parameters
                        # to find the optimal condition
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
    def model(self):
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
    def success_label(self) -> None:
        """
            Defines success label based on:
            success = (active == True) & (delinquent == False)
        """
        self.data = self.data.with_columns(
            (
                (pl.col('active_account') == True) & (pl.col('delinquent_account') == False)
            ).alias('success')
        )
    def remove_nans(self) -> None:
        """ Removes NaN values from the dataset.
        """        
        self.data = self.data.drop_nans()

class model:
    def __init__(self):
        self.dataset_class = Dataset()
        self.data = self.dataset_class.data
        self.dictionary_df = DataDictionary().frame_dictionary()
    def linear_regression(self):
        # Linear Regression
        # y = success label
        # X = actionable features
        y:pl.DataFrame = (
            self.data
            .drop_nans() # prevents column mismatch due to NaNs
            .select(pl.col('success'))
        )
        X:pl.DataFrame = (
            self.data
                .drop_nans() # prevents column mismatch due to NaNs
                .select(
                    self.dictionary_df
                        .filter(pl.col('Classification') == 'actionable')
                        .select(pl.col('Field'))
                        .to_series()
                        .to_list()
                )
                .drop_nans()
        )
        # Run the model

        model = LinearRegression()
        model.fit(X, y)
        # observe results
        coefficients:pl.Series = pl.Series(model.coef_) # model.coef_ is a list in a list
        print(coefficients)
        intercept:Any = model.intercept_ # model.intercept_ is a single value in a ndarray
        print("Intercept:", intercept)
        # coefficients = magnitude of the relationship for each feature
        # intercept = sign, positive or negative effect
        # so if the all the features are negative, and the intercept is positive, then...
        print(model.predict(X))
        # 0.0 - explains nothing
        # 1.0 - perfect fit
        # ~0.6–0.8 → acceptable
        r_squared = model.score(X, y)
        print("R-squared:", r_squared)
        mse = mean_squared_error(y, model.predict(X))
        print("Mean Squared Error:", mse)
        # Train/ test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42 # Thank you, Douglas Adams
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        model.score(X_test, y_test)
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        print(model.summary())
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
        pipeline.fit(X_train, y_train)
    def logistic_regression(
            self,
            target:str = "success", # binary 0/1 # target == "y"
            num_cols:List[str] = ["income", "dti_score","alt_risk_score_2"],
            cat_cols:List[str] = ["seg_id"],
            test_size:float=0.2, 
            random_state:int=42, 
            n_estimators:int=600,
            learning_rate:float=0.05,
            max_depth:int=4,
            subsample:float=0.8,
            colsample_bytree:float=0.8,
            reg_lambda:float=1.0,
            tree_method:str="hist",   # fast CPU training
            eval_metric:str="logloss",
    ):
        # Logistic Regression is for when the target is binary
        # Example cleanup (customize as needed)
        df = (
            self.data
            .with_columns([
                pl.col(target).cast(pl.Int64),
                # optional: fill nulls
                *[pl.col(c).fill_null(pl.median(c)) for c in num_cols],
                *[pl.col(c).fill_null("UNKNOWN").cast(pl.Utf8) for c in cat_cols],
            ])
        )

        # Split in sklearn (keeps it simple)
        X_pd = df.select(num_cols + cat_cols).to_pandas()
        y_pd = df.select(target).to_pandas().iloc[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(
            X_pd, y_pd, test_size=0.2, random_state=42, stratify=y_pd
        )

        # --- sklearn: preprocess + model ---
        preprocess = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ],
            remainder="drop",
        )

        clf = Pipeline(steps=[
            ("prep", preprocess),
            ("model", LogisticRegression(
                solver="lbfgs",      # good default
                max_iter=2000,
                n_jobs=None,         # ignored by lbfgs; used by some solvers
                class_weight=None    # set to "balanced" if labels are imbalanced
            )),
        ])

        clf.fit(X_train, y_train)

        # --- Evaluate ---
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        print("ROC AUC:", roc_auc_score(y_test, proba))
        print(classification_report(y_test, pred))
    def gb_trees(
            self,
            target:str = "success", # binary 0/1 # target == "y"
            num_cols:List[str] = ["income", "dti_score","alt_risk_score_2"],
            cat_cols:List[str] = ["seg_id"],
            test_size:float=0.2, 
            random_state:int=42, 
            n_estimators:int=600,
            learning_rate:float=0.05,
            max_depth:int=4,
            subsample:float=0.8,
            colsample_bytree:float=0.8,
            reg_lambda:float=1.0,
            tree_method:str="hist",   # fast CPU training
            eval_metric:str="logloss",
            ):
        # Gradient Boosted Trees
        
        df = (
            self.data
            .with_columns([
                pl.col(target).cast(pl.Int64),
                *[pl.col(c).fill_null(pl.median(c)) for c in num_cols],
                *[pl.col(c).fill_null("UNKNOWN").cast(pl.Utf8) for c in cat_cols],
            ])
        )

        # Convert to pandas for sklearn preprocessing
        X = df.select(num_cols + cat_cols).to_pandas()
        y = df.select(target).to_pandas().iloc[:, 0]

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y,
        )

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "num", 
                    "passthrough", 
                    num_cols,
                ),
                (
                    "cat", 
                    OneHotEncoder(handle_unknown="ignore"), 
                    cat_cols,
                ),
            ]
        )

        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            tree_method=tree_method,   # fast CPU training
            eval_metric=eval_metric,
            random_state=random_state,
        )

        clf = Pipeline(steps=[
                (
                    "prep", 
                    preprocess,
                ),
                (
                    "model", 
                    model,
                ),
            ]
        )

        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        print("ROC AUC:", roc_auc_score(y_test, proba))
        print(classification_report(y_test, pred))
