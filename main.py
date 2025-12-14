import os
import sys
import polars as pl
from typing import List, Dict, Callable
from dataclasses import dataclass
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
    def fetch_dictionary(self,source:str=os.path.join(os.getcwd(),"Project Data Dictionary.xlsx")) -> pl.DataFrame:
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

class dataset:
    """
        A dataset object to group the functionality and data of the reference material.
    """    
    def __init__(self,source:str):
        self.source = source
        self.data = pl.read_csv(source)
    def preview(self,n:int=5) -> pl.DataFrame:
        """ Previews the dataset.

        Args:
            n (int, optional): Number of rows to preview. Defaults to 5.

        Returns:
            pl.DataFrame: _description_
        """        
        return self.data.head(n)    