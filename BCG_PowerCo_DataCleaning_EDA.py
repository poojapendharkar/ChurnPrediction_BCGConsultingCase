# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:53:54 2023

@author: penpo
"""
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


client_data=pd.read_csv("client_data.csv")
client_data.head()

price_data=pd.read_csv("price_data.csv")
price_data.head()


profile = client_data.profile_report(title='Data Preprocessing Report')
profile.to_file("ClientDataEADReport.html")

profile = price_data.profile_report(title='Data Preprocessing Report')
profile.to_file("PriceDataEADReport.html")

#check for missing and null data values
client_data.info()
price_data.info()
#No null values or missing values found

client_data.describe()

''''Data Visualisations'''

#we first start with uiveriate analysis for churn
# Counting the number of churn vs non-churn clients
churn_counts = client_data['churn'].value_counts()

# Calculating the percentages
total_clients = len(client_data)
churn_percentages = (churn_counts / total_clients) * 100

# Plotting the bar graph
churn_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Number of Churn vs Retained Clients')
plt.xlabel('Churn Status')
plt.ylabel('Number of Clients')
plt.xticks([1,0], ['Churn', 'Retained'], rotation=45)

# Adding the count and percentages above the bars
for index, value in enumerate(churn_counts):
    plt.text(index, value + 50, str(value) + '\n' + f'{churn_percentages[index]:.2f}%', ha='center')

plt.show()
#9.72% of clients churn right now

#now we perform bivariate analysis
#simple scatter plots
sns.relplot(x="cons_12m", y="forecast_cons_12m",hue='churn' , data=client_data)


##jointplot
sns.jointplot(x="cons_12m", y="forecast_cons_12m",hue='churn' , data=client_data)

#Are there any peak months for activation date which show churn numbers are high? 
# Convert the 'date_activ' column to datetime format
client_data['date_activ'] = pd.to_datetime(client_data['date_activ'])

# Extract month-year from the date
client_data['month_year'] = client_data['date_activ'].dt.to_period('M')

# Group by month-year and sum the churns (since churn is 1 for churned and 0 for non-churned)
churn_over_time = client_data.groupby('month_year')['churn'].sum()

# Plotting the time series
churn_over_time.plot(figsize=(25, 10), marker='o', linestyle='-')
plt.title('Churn Over Time')
plt.xlabel('Month-Year')
plt.ylabel('Number of Churns')
plt.grid(True)
plt.tight_layout()
plt.show()

#analysis channel sales to find pattern in churn
# Grouping by 'channel_sales' and calculating the churn rate
churn_rate_by_channel = client_data.groupby('channel_sales')['churn'].mean()

# Sorting the values for better visualization
churn_rate_by_channel = churn_rate_by_channel.sort_values(ascending=False)

# Plotting the bar graph
churn_rate_by_channel.plot(kind='bar', figsize=(15, 7), color='skyblue')
plt.title('Churn Rate by Sales Channel')
plt.xlabel('Sales Channel')
plt.ylabel('Churn Rate')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#high avg churn found for channel - foosdfpfkusacimwkcsosbicdxkicaua (The data may be encrypted due to which the channel name is not clear)


''''''

#however the price_data shows that Id is not unique identifier
#Muliple entries of prices were found to handle this we will take aggregate and std values
#this will help us avoid one to many relation when we merge datasets
pricedf  = price_data
price_data_std = pricedf.groupby('id').agg({
    'price_off_peak_var': ['mean', 'std'],
    'price_peak_var': ['mean', 'std'],
    'price_mid_peak_var': ['mean', 'std'],
    'price_off_peak_fix': ['mean', 'std'],
    'price_peak_fix': ['mean', 'std'],
    'price_mid_peak_fix': ['mean', 'std'],
}).reset_index()

price_data_std.columns = ['_'.join(col).rstrip('_') for col in price_data_std.columns.values]
price_data_std.rename(columns={'id_': 'id'}, inplace=True)

# Transform date columns to datetime type
client_data["date_activ"] = pd.to_datetime(client_data["date_activ"], format='%Y-%m-%d')
client_data["date_end"] = pd.to_datetime(client_data["date_end"], format='%Y-%m-%d')
client_data["date_modif_prod"] = pd.to_datetime(client_data["date_modif_prod"], format='%Y-%m-%d')
client_data["date_renewal"] = pd.to_datetime(client_data["date_renewal"], format='%Y-%m-%d')
price_data['price_date'] = pd.to_datetime(price_data['price_date'], format='%Y-%m-%d')

# Create mean average data
mean_year = price_data.groupby(['id']).mean().reset_index()
mean_6m = price_data[price_data['price_date'] > '2015-06-01'].groupby(['id']).mean().reset_index()
mean_3m = price_data[price_data['price_date'] > '2015-10-01'].groupby(['id']).mean().reset_index()


mean_year["mean_year_price_p1"] = mean_year["price_off_peak_var"] + mean_year["price_off_peak_fix"]
mean_year["mean_year_price_p2"] = mean_year["price_peak_var"] + mean_year["price_peak_fix"]
mean_year["mean_year_price_p3"] = mean_year["price_mid_peak_var"] + mean_year["price_mid_peak_fix"]

mean_year = mean_year.rename(
    index=str, 
    columns={
        "price_off_peak_var": "mean_year_price_offpeak_var",
        "price_peak_var": "mean_year_price_peak_var",
        "price_mid_peak_var": "mean_year_price_midpeak_var",
        "price_off_peak_fix": "mean_year_price_offpeak_fix",
        "price_peak_fix": "mean_year_price_peak_fix",
        "price_mid_peak_fix": "mean_year_price_midpeak_fix"
    }
)

mean_6m["mean_6m_price_p1"] = mean_6m["price_off_peak_var"] + mean_6m["price_off_peak_fix"]
mean_6m["mean_6m_price_p2"] = mean_6m["price_peak_var"] + mean_6m["price_peak_fix"]
mean_6m["mean_6m_price_p3"] = mean_6m["price_mid_peak_var"] + mean_6m["price_mid_peak_fix"]

mean_6m = mean_6m.rename(
    index=str, 
    columns={
        "price_off_peak_var": "mean_6m_price_offpeak_var",
        "price_peak_var": "mean_6m_price_peak_var",
        "price_mid_peak_var": "mean_6m_price_midpeak_var",
        "price_off_peak_fix": "mean_6m_price_offpeak_fix",
        "price_peak_fix": "mean_6m_price_peak_fix",
        "price_mid_peak_fix": "mean_6m_price_midpeak_fix"
    }
)


mean_3m["mean_3m_price_p1"] = mean_3m["price_off_peak_var"] + mean_3m["price_off_peak_fix"]
mean_3m["mean_3m_price_p2"] = mean_3m["price_peak_var"] + mean_3m["price_peak_fix"]
mean_3m["mean_3m_price_p3"] = mean_3m["price_mid_peak_var"] + mean_3m["price_mid_peak_fix"]

mean_3m = mean_3m.rename(
    index=str, 
    columns={
        "price_off_peak_var": "mean_3m_price_offpeak_var",
        "price_peak_var": "mean_3m_price_peak_var",
        "price_mid_peak_var": "mean_3m_price_midpeak_var",
        "price_off_peak_fix": "mean_3m_price_offpeak_fix",
        "price_peak_fix": "mean_3m_price_peak_fix",
        "price_mid_peak_fix": "mean_3m_price_midpeak_fix"
    }
)

# Merge into 1 dataframe
price_features = pd.merge(mean_year, mean_6m, on='id')
price_features = pd.merge(price_features, mean_3m, on='id')

price_allfeatures = pd.merge(price_features, price_data_std, on='id' )

#merge the datasets
merged_data = pd.merge(client_data, price_allfeatures, on='id', how='left')

merged_data.info()

#Saved the data in csv for quick analysis further on so we don't have to run this code again every time
merged_data.to_csv('MergergeCleanClientPrice.csv', index=False)


profile = merged_data.profile_report(title='Data Preprocessing Report')
profile.to_file("MergedDataEADReport.html")












