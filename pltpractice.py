import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize = (5, 3), layout = "constrained")



## Creating figures
# fig, ax = plt.subplots(2, 2, figsize = (4, 3), layout = 'constrained')
# fig, ax = plt.subplot_mosaic([['left', 'A'], ['left', 'B']], figsize = (4, 3), layout = 'constrained')

# x = np.array(np.linspace(0, 2, 25))
# y = x ** 3

# plt.plot(x, y, 'b-')
# plt.show()

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