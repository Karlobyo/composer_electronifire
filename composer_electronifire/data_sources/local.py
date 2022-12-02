
import pandas as pd

from colorama import Fore, Style

#from composer_electronifire.ml_logic.params import PROJECT, DATASET


def get_local_data(file: str,
                dtypes: dict = None,
                verbose=True) -> pd.DataFrame:
    """
    return a big query dataset table
    format the output dataframe according to the provided data types
    """
    if verbose:
        print(Fore.MAGENTA + f"Source data from local {file}" + Style.RESET_ALL)

    try:


        df = pd.read_csv(file)
        df = df.set_index([df.columns[0]]).sort_index()
        # read_csv(dtypes=...) will silently fail to convert data types, if column names do no match dictionnary key provided.
     #   if isinstance(dtypes, dict):
      #      assert dict(df.dtypes) == dtypes

        # if columns is not None:
        #     df.columns = columns

    except pd.errors.EmptyDataError:

        return None  # end of data

    return df
