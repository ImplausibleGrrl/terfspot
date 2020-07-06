import pandas as pd
import csv

one = pd.read_csv('./overall_proba.csv')

one['label'] = [0] * 860

one.to_csv('./proba_all.csv', index=False)
