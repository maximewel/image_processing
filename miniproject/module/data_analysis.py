import pandas as pd
import os


df = pd.read_pickle(os.path.join(os.path.dirname(__file__) , "../out/dataframe.pkl"))  

print(df)