

import pandas as pd
import numpy as np
import seaborn as sns 
import os
from scipy.stats import pearsonr,spearmanr
os.chdir("E:\\Avito\\submission\\no_blend")

sub1 = pd.read_csv('lgb-nn-555-no-prob.csv')
sub2 = pd.read_csv('Tricky_trick_2155.csv')

sub1 = sub1.sort_values(by='item_id')
sub2 = sub2.sort_values(by='item_id')

print(spearmanr(sub1.deal_probability, sub2.final))
print(pearsonr(sub1.deal_probability, sub2.final))

ens = sub1.copy()

ens.deal_probability = (sub1.deal_probability*0.876) + (sub2.final*0.124)

ens.deal_probability = (sub1.deal_probability*0.846) + (sub2.final*0.154)

ens.deal_probability = (sub1.deal_probability*0.876) + (sub2.final*0.124)

print(pearsonr(ens.deal_probability, sub2.final)) 

ens[['item_id','deal_probability']].to_csv('final-v1.csv', index=False)

print('correlation between models outputs')



