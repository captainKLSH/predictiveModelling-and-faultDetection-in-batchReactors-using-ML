import pandas as pd
import numpy as np

def genfault(filepath):
    df=pd.read_csv(filepath,header=0)
    