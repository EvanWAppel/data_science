import os
import sys
import polars as pl
from typing import List, Dict, Callable, Any, Optional, Tuple
from dataclasses import dataclass
import variables
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
#Create data dictionary table object
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
