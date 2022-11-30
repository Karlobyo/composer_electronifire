
from google.cloud import bigquery

import pandas as pd

from colorama import Fore, Style

#from composer_electronifire.ml_logic.params import PROJECT, DATASET


def get_bq_data(table: str,
                dtypes: dict = None,
                verbose=True) -> pd.DataFrame:
    """
    return a big query dataset table
    format the output dataframe according to the provided data types
    """
    if verbose:
        print(Fore.MAGENTA + f"Source data from big query {table}" + Style.RESET_ALL)
    client = bigquery.Client
    try:
        rows = client.list_rows(table=table)

        df = pd.DataFrame(rows)
        df = df.set_index([df.columns[0]]).sort_index()
        # read_csv(dtypes=...) will silently fail to convert data types, if column names do no match dictionnary key provided.
        #   if isinstance(dtypes, dict):
        #      assert dict(df.dtypes) == dtypes

        #if columns is not None:
         #   df.columns = columns

    except pd.errors.EmptyDataError:

        return None  # end of data

    return df


# def save_bq_table(table: str,
#                   data: pd.DataFrame,
#                   is_first: bool):
#     """
#     save a chunk of the raw dataset to big query
#     empty the table beforehands if `is_first` is True
#     """

#     print(Fore.BLUE + f"\nSave data to big query {table}:" + Style.RESET_ALL)

#     write_mode = "WRITE_TRUNCATE"

#     job = pass
