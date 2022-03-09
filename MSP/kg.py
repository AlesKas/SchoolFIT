from numpy.core.fromnumeric import sort
from scipy import stats
import pandas as pd
import numpy as np

data1 = [25.26,
20.5,
24.62,
19.63,
26.89,
21.08,
23.15,
20.13,
23.35,
19.83,
26.46,
20.28,
23.26,
18.97,
22.93,
20.77,
25.96,
21.31,
25.7,
21.22,
24.89,
21.31,
24.15,
19.78,
23.87,
19.56,
24.13,
19.48,
24.94,
20.61,
25.65,
20.12,
24.32,
21.16,
23.08,
23.88,
25.03,
19.83,
24.12,
21.33,
25.37,
19.15,
22.36,
20.44,
24.02,
21.64,
25.59,
18.47,
23.84,
22.22,
]

data2 = [21.48,
25.08,
24.41,
26.85,
25.19,
21.78,
25.08,
23.84,
23.9,
24.07,
23.56,
23.29,
22.64,
21.06,
25.26,
20.68,
26.98,
22.92,
23.63,
23.26,
24.61,
23.97,
21.62,
20.05,
23.59,
23.76,
26.5,
23.27,
24.47,
25.05,
21.59,
25.25,
23.76,
21.94,
23.79,
23.16,
27.81,
23.49,
25.15,
25.99,
26.36,
21.48,
22.86,
21.12,
23.87,
25.46,
24.51,
22.16,
27.11,
22.96,
]

stat, p = stats.shapiro(data1)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample1 looks Gaussian (fail to reject H0)')
else:
	print('Sample1 does not look Gaussian (reject H0)')


stat, p = stats.shapiro(data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample2 looks Gaussian (fail to reject H0)')
else:
	print('Sample2 does not look Gaussian (reject H0)')


stat, p_value = stats.mannwhitneyu(data1, data1)
print('Statistics=%.2f, p=%.2f' % (stat, p_value))
# Level of significance
alpha = 0.05
# conclusion
if p_value < alpha:
    print('Reject Null Hypothesis (Significant difference between two samples)')
else:
    print('Do not Reject Null Hypothesis (No significant difference between two samples)')

# import statsmodels.api as sm
# from statsmodels.formula.api import ols

# df = pd.DataFrame({
# 	'faktor-1' : np.repeat(['rano', 'poledne', 'vecer'], 15),
# 	'faktor-2' : np.tile(np.repeat(['ticho', 'hudba', 'hluk', 'krik'], 5), 2),
# 	'data' : [6,8,11,None, None, 7,8,12,10,None,8,7,20,None,None,13,21,25,None,None, 8,13,7,None,None,5,11,7,None,None,10,17,11,13,None,
# 	14,15,None,None, 7,8,6,None,None, 6,8,16,15,None,12,17,19,None,None,13,17,15,22,18]
# })

# print(df[:10])

from statsmodels.sandbox.stats.runs import mcnemar

df = pd.read_csv('data.csv',  sep = r';')
data = np.array([
	[7,2,1,1,0],
	[2,3,5,3,1],
	[3,3,1,6,3],
	[3,2,5,9,1],
	[1,3,5,3,0],
	[0,1,1,3,1],
	[0,1,4,3,5]
])

df = pd.DataFrame(data, columns=["0-2", "3-4", "5-6", "7-8", "9-10"])
q, pVal = mcnemar(data)
print(f"{q} {pVal}")