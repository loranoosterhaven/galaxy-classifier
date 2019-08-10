import pandas as pd

ds = pd.read_csv('dataset.csv')

N = 100

# select 100 objects from each Er, Ec, SBb, Sb
Er = ds.loc[ds['gz2_label'] == 'Er'][0:N]
Ec = ds.loc[ds['gz2_label'] == 'Ec'][0:N]
SB = ds.loc[ds['gz2_label'].isin(['SBa2l', 'SBb2l', 'SBc2l', 'SBb2m'])][0:N]
Sb = ds.loc[ds['gz2_label'].isin(['Sb2m', 'Sb', 'Sb2l'])][0:100]

# output samples in random order
pd.concat([Er, Ec, SB, Sb]).sample(frac=1).to_csv('data.csv')