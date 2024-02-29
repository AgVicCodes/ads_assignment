import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize = (5, 3), layout = "constrained")

# Creating figures
fig, ax = plt.subplots(2, 2, figsize = (4, 3), layout = 'constrained')
fig, ax = plt.subplot_mosaic([['left', 'A'], ['left', 'B']], figsize = (4, 3), layout = 'constrained')

x = np.array(np.linspace(0, 2, 25))
y = x ** 3

plt.plot(x, y, 'b-')
plt.show()

sal = pd.read_csv("csv/salaries.csv", index_col = 0)

print(sal.size)

sal = cleanse(sal)

sal = sal.reset_index()

print(sal.size)

sal.columns


# pd.set_option('display.max_rows', None)

salary_by_job = sal.groupby(['job_title'])['salary_in_usd'].sum()
salary_by_job_and_year = sal.groupby(['job_title', 'work_year'])['salary_in_usd'].sum()
print(salary_by_job_and_year.size)
print(salary_by_job_and_year)

plt.figure(figsize=(10, 6))
plt.scatter(sal['job_title'], sal['salary_in_usd'])
plt.xticks(rotation=90)
plt.xlabel('Job Title')
plt.ylabel('Salary in USD')
plt.title('Relation between Job Titles and Salaries')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(salary_by_job_and_year.index, salary_by_job_and_year.values)
plt.xticks(rotation=90)
plt.xlabel('Job Title')
plt.ylabel('Total Salary in USD')
plt.title('Total Salary by Job Title')
plt.tight_layout()
plt.show()


# sal.to_excel("data.xlsx")

# sal_year_job = sal.groupby(["work_year", "job_title"])["salary_in_usd"].aggregate("sum").unstack()
sal_by_year_job = sal.pivot_table("salary_in_usd", index = "job_title", columns = "work_year", aggfunc = "sum")

# print(sal_year_job)
# print(sal_year_job.iloc[:, 2])


for index, row in sal_by_year_job.iterrows():
    for col in sal_by_year_job.columns:
        if pd.isna(row[col]):
            sal_by_year_job.at[index, col] = 0

pd.set_option("display.max_columns", None)

# pd.set_option('display.max_rows', None)

sal_by_year_job

# .isna().any()

# plt.pie(sal_year_job)
# plt.legend(loc = "upper right")
# plt.show()


# fig, ax = plt.subplots()

sal_by_year_job.plot.bar()

plt.show


num_groups = 5
group_size = len(sal_by_year_job) 

# Split the DataFrame into three parts
grouped_dfs = [sal_by_year_job.iloc[i * group_size: (i + 1) * group_size] for i in range(num_groups)]

# Print the resulting DataFrames
for i, group_df in enumerate(grouped_dfs):
    print(f"Group {i + 1}:")
    print(group_df)
    print("\n")

data = np.random.random((12, 12))
 
colors_list = ['#0099ff', '#33cc33']
cmap = colors.ListedColormap(colors_list)

plt.imshow(data, interpolation='nearest')

# Add colorbar
plt.colorbar()
 
plt.title("Heatmap with color bar")
plt.show()

def plot_sns_sales_scatter(df):
    """
        Function to plot 
    """
    plt.figure(figsize = (14, 6), dpi = 144)

    x = df.Temperature
    y = df.Weekly_Sales

    sns.scatterplot(data = sales, x = "Temperature", y = "Weekly_Sales", color = "#3f3f3f", edgecolor = '#a5a5a5')
    # plt.scatter(x, y, c = "#3f3f3f", s = 10)

    plt.title("EFFECT OF TEMPERATURE ON WEEKLY SALES", fontdict = {"size": 20})

    plt.xlabel("Temperature in Farenheit", fontdict = {"size": 14, "color" : "#000"})  
    plt.ylabel("Weekly Sales", fontdict = {"size": 14, "color" : "#000"})

    plt.ticklabel_format(style='plain')

    plt.show()

plot_sales_scatter(sales)
plot_sns_sales_scatter(sales)

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

for i in sal.index:
    for j in currency:
        if sal["salary_currency"][i] == currency[j]:
            sal["new_sal_in_usd"][i] = sal["salary"][i] / ex_rate[j]

print(sal_year_job)

plt.pie(sal_by_year)
plt.legend(loc = "upper right")
plt.show()

# Bar Chart
plt.figure(dpi = 144)

plt.bar(sal.salary)
plt.legend(loc = "upper right")
plt.show()

# Histogram Chart
plt.figure(dpi = 144)

plt.hist(sal.salary)
plt.legend(loc = "upper right")
plt.show()

# Scatter Chart
plt.figure(dpi = 144)

plt.scatter(sal.salary)
plt.legend(loc = "upper right")
plt.show()