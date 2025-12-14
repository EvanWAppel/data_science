from typing import List, Dict
import os
data_dictionary:str = os.path.join(os.getcwd(),"Project Data Dictionary.xlsx")
project_data:str = os.path.join(os.getcwd(),"Project Data.csv")
default_classifications:List[Dict[str,str]] = [
            {'field':'active_account','classification':'target'},
            {'field':'delinquent_account','classification':'target'},
            {'field':'ident_monitor_opt','classification':'actionable'},
            {'field':'Num_Bk_Accts','classification':'immutable'},
            {'field':'income','classification':'immutable'},
            {'field':'bk_accts_ssn','classification':'immutable'},
            {'field':'cells_ssn','classification':'immutable'},
            {'field':'dls_ssn','classification':'immutable'},
            {'field':'emails_ssn','classification':'immutable'},
            {'field':'hmphones_ssn','classification':'immutable'},
            {'field':'addrs_ssn','classification':'immutable'},
            {'field':'ssn_ssn','classification':'immutable'},
            {'field':'zips_ssn','classification':'immutable'},
            {'field':'empl_ssn_6Mo','classification':'immutable'},
            {'field':'pday_inq_15days','classification':'immutable'},
            {'field':'dti_score','classification':'derived'},

            {'field':'days_from_registration','classification':'actionable'},
            {'field':'days_from_login','classification':'actionable'},
            {'field':'asset_score','classification':'derived'},
            {'field':'alt_risk_score','classification':'derived'},
            {'field':'seg_id','classification':'derived'},
            {'field':'stability_score','classification':'derived'},
            {'field':'max_bkaccts','classification':'actionable'},
            {'field':'auto_inq_1dy','classification':'immutable'},
            {'field':'auto_inq_7dy','classification':'immutable'},
            {'field':'auto_inq_72hr','classification':'immutable'},
            {'field':'cc_inq_72hr','classification':'immutable'},
            {'field':'cc_inq_10dy','classification':'immutable'},
            {'field':'pl_inq_72hr','classification':'immutable'},
            {'field':'pl_inq_90dy','classification':'immutable'},
            {'field':'alt_risk_score_2','classification':'derived'}

        ]

constraints:Dict[str,List[int|float]] = {
    "ident_monitor_opt": [0, 1],
    "income": [4000,136000],
    "days_from_registration": [42,4592],
    "days_from_login": [1,424],
    "max_bkaccts": [1,4],
}
change_cost:Dict[str,int] = {
    "ident_monitor_opt": 1,
    "income": 1,
    "days_from_registration": 1,
    "days_from_login": 1,
    "max_bkaccts": 1,
}