import re

def clean_name(x:str)->str:
    """
    Clean the pandas dataframe name to be used as a file name
    """
    x = re.sub(r'[^\w\s]', '', x)
    if x in ["type","name","index"]:
        x = f"{x}_1"
    return x