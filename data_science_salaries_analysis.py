import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sal = pd.read_csv('csv/salaries.csv', index_col = 0)

def clean_dataframe(df):
    """
        Function that takes a DataFrame, 
        cleans and prepare it for 
        further analyis.
    """
    # Dropping duplicate rows
    df = df.drop_duplicates()
    
    # Removing null values
    for i in df.columns:
        if df[i].isna().all():
            df.drop_na()  
    
    return df

sal.reset_index(drop = False, inplace = True)
sal = clean_dataframe(sal)

# print(sal)
# print(sal.describe(include="number"))


# Pie Chart
plt.figure(dpi = 144)

sal_by_year = sal.groupby(["work_year"])['salary_in_usd']

# print()

plt.pie(sal_by_year)
plt.legend(loc = "upper right")
plt.show()

# # Bar Chart
# plt.figure(dpi = 144)

# plt.bar(sal.salary)
# plt.legend(loc = "upper right")
# plt.show()

# # Histogram Chart
# plt.figure(dpi = 144)

# plt.hist(sal.salary)
# plt.legend(loc = "upper right")
# plt.show()

# # Scatter Chart
# plt.figure(dpi = 144)

# plt.scatter(sal.salary)
# plt.legend(loc = "upper right")
# plt.show()