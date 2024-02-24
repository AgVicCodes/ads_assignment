import numpy as np
import pandas as pd

x = np.array([2, 3, 4, 5])

salaries = pd.read_csv('salaries.csv', index_col = 0)

print(salaries)