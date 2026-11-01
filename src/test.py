import pandas as pd
p = pd.read_csv("results/predictions_val.csv", index_col=0, parse_dates=True)
neg = p[p.y == 1]
print((neg.price > -1).mean(), (neg.price > -5).mean())   # share of barely-negative hours
fa = p[(p.pred_XGB == 1) & (p.y == 0)]
print(fa.price.describe())                                 # what "false alarms" actually cleared at