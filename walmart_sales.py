import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss

def cleanse(df):
    """
        This function takes a DataFrame, 
        cleans or prepares it and returns it 
        for further analysis
    """
    
    # Removing duplicate data
    df = df.drop_duplicates()
    
    # Dropping empty columns
    for col in df.columns:
        if df[col].isna().all():
            df.dropna()
            
    return df


def plot_temp_sales_scatter(df):
    """
        This function takes a DataFrame 
        and creates a scatter plot
        relating temperature to sales.
    """

    # Setting the figure size and resolution
    plt.figure(figsize = (14, 6), dpi = 144)

    # Setting the scatter plot variables
    x = df.Temperature
    y = df.Weekly_Sales

    # The scatter plot
    # Color code contains opacity to show volume and avoid overcrowding 
    plt.scatter(x, y, c = "#3f3f3f58", s = 10)

    # Setting the title size
    plt.title("EFFECT OF TEMPERATURE ON WEEKLY SALES", 
              fontdict = {
                            "size": 20
                         })

    # Setting the axis label
    plt.xlabel("Temperature in Farenheit", fontdict = {"size": 14})  
    plt.ylabel("Weekly Sales", fontdict = {"size": 14})

    # Setting full digits display on the axis
    plt.ticklabel_format(style = "plain")

    plt.show()



def plot_unemployment_sales_scatter(df):
    """
        This function takes a DataFrame 
        and creates a scatter plot
        relating the unemplorment rate to sales.
    """

    # Setting the figure size and resolution
    plt.figure(figsize = (14, 6), dpi = 144)

    # Setting the scatter plot variables
    x = df.Unemployment
    y = df.Weekly_Sales

    # The scatter plot
    # Color code contains opacity to show volume and avoid overcrowding 
    plt.scatter(x, y, c = "#3f3f3f50", s = 10, marker = "*")

    # Setting the title size
    plt.title("EFFECT OF UNEMPLOYMENT RATE ON WEEKLY SALES", 
              fontdict = {
                            "size": 20
                         })

    # Setting the axis label
    plt.xlabel("Unemployment Rate", fontdict = {"size": 14})
    plt.ylabel("Weekly Sales", fontdict = {"size": 14})

    # Setting full digits display on the axis
    plt.ticklabel_format(style = "plain")

    plt.show()


def plot_temp_hist(df):
    """
        This function plots a Temperature histogram
    """

    # Setting the figure size and resolution
    plt.figure(figsize = (5, 2), dpi = 300)

    # Setting the histogram plot variables
    x = df.Temperature

    # The histogram plot
    plt.hist(x, bins = 30, color = "#3d3d3d")

    # Setting the axis label
    plt.xlabel("Temperature (deg F)")
    plt.ylabel("Frequency")

    # Setting the tick size
    plt.xticks(size = 5)
    plt.yticks(size = 5)

    plt.show()


def plot_fuel_hist(df):
    """
        This function plots a Fuel Price histogram
    """

    # Setting the figure size and resolution
    plt.figure(figsize = (5, 2), dpi = 300)

    # Setting the histogram plot variables
    x = df.Fuel_Price

    # The histogram plot
    plt.hist(x, bins = 30, color = "#3d3d3d")

    # Setting the axis label
    plt.xlabel("Fuel Prices ($)")
    plt.ylabel("Frequency")

    # Setting the tick size
    plt.xticks(size = 5)
    plt.yticks(size = 5)

    plt.show()


def plot_sales_hist(df):
    """
        This function plots the weekly sales histogram
    """

    # Setting the figure size and resolution
    plt.figure(figsize = (5, 2), dpi = 300)

    # Setting the histogram plot variables
    x = df.Weekly_Sales
    
    # The histogram plot
    plt.hist(x, bins = 45, color = "#3d3d3d")

    # Setting the axis label
    plt.xlabel("Weekly Sales ($)")
    plt.ylabel("Frequency")

    # Setting full digits display on the axis
    plt.ticklabel_format(style = "plain")

    # Setting the tick size
    plt.xticks(size = 5)
    plt.yticks(size = 5)

    plt.show()


def plot_cpi_boxplot(df):
    """
        This function plots a box plot 
        of the Consumer Price Index 
    """
    # Setting the figure size and resolution
    plt.figure(figsize = (5, 2), dpi = 144)

    # The Box plot
    sns.boxplot(data = df.CPI, x = df.CPI, color = "#cdcdcd")
    plt.title("Distribution of Consumer Price Index")
    plt.xlabel("CPI")
    plt.ylabel("Frequency")
    plt.show()

# Creating the dataframe variable
sales = pd.read_csv("csv/walmart_sales.csv", index_col = 0)

# Checking the shape of the dataframe before cleaning
print(sales.shape)

# Data cleaning using the cleanse function 
sales = cleanse(sales)

sales = sales.reset_index()

# Checking the shape of the dataframe after cleaning
print(sales.shape)

# Confirming numeric dtypes 
print(sales.dtypes)

sales.describe()

sales.corr(method = "pearson", numeric_only = True)

# Plotting all graphs
plot_temp_sales_scatter(sales)
plot_unemployment_sales_scatter(sales)
plot_sales_hist(sales)
plot_fuel_hist(sales)
plot_temp_hist(sales)
plot_cpi_boxplot(sales)
