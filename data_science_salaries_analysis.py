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

# print(sal.describe())

# Pie Chart
# plt.figure(dpi = 144)

sal_curr = sal.groupby(["salary_currency"])["salary"].sum()

print(sal_curr)

# print(sal.index)

sal_year = sal.groupby(["work_year"])['salary_in_usd'].sum()

currency = np.array(["AUD", "BRL", "CAD", "CHF", "CLP", "DKK", "EUR", "GBP", "HKD", "HUF", "ILS", "INR", "JPY", "MXN", "NOK", "PHP", "PLN", "SGD", "THB", "TRY", "USD", "ZAR"])

ex_rate = np.array([0.66, 0.2, 0.74, 1.14, 0.001, 0.15, 1.08, 1.27, 0.13, 0.0028, 0.28, 0.012, 0.0066, 0.058, 0.095, 0.018, 0.25, 0.74, 0.028, 0.032, 1, 0.052])

sal["new_sal_in_usd"] = 0

sal['new_sal_in_usd'] = sal['new_sal_in_usd'].astype('float64')

for index, row in sal.iterrows():
    curr = row["salary_currency"]

    currency_index = np.where(currency == curr)[0]

    if len(currency_index) > 0:
        currency_index = currency_index[0]

        new_salary_usd = row['salary'] / ex_rate[currency_index]
        
        sal.at[index, 'new_sal_in_usd'] = new_salary_usd

print(sal["new_sal_in_usd"])

print(sal.size)


clean_dataframe(sal)

sal = sal.reset_index(drop=False, inplace = True)

print(sal)



sal_job_year = sal.groupby(["job_title", "work_year"])['salary_in_usd'].sum()

sal_year_job = sal.groupby(["work_year", "job_title"])['salary_in_usd']

fig, axs = plt.subplots(1, 1, figsize = (5, 5))

print(str(sal_year))

axs.pie(sal_year, labels = sal_year)

axs.legend()

plt.show()
    

# for i in sal.index:
#     for j in currency:
#         if sal["salary_currency"][i] == currency[j]:
#             sal["new_sal_in_usd"][i] = sal["salary"][i] / ex_rate[j]

# print(sal_year_job)

# plt.pie(sal_by_year)
# plt.legend(loc = "upper right")
# plt.show()

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