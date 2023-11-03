# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:10:22 2023

@author: penpo
"""
import numpy as np
import pandas as pd
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, classification_report
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import yeojohnson

df=pd.read_csv("MergedCleanClientPrice.csv")

df["date_activ"] = pd.to_datetime(df["date_activ"], format='%Y-%m-%d')
df["date_end"] = pd.to_datetime(df["date_end"], format='%Y-%m-%d')
df["date_modif_prod"] = pd.to_datetime(df["date_modif_prod"], format='%Y-%m-%d')
df["date_renewal"] = pd.to_datetime(df["date_renewal"], format='%Y-%m-%d')

price_df = pd.read_csv('price_data.csv')
price_df["price_date"] = pd.to_datetime(price_df["price_date"], format='%Y-%m-%d')
price_df.head()

# Group off-peak prices by companies and month
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()

#evaluate the effectiveness of a feature against a label
mergeddata = pd.merge(df, diff, on='id')

#adding other features

#comparison between the past consumption and forecasted consumption to see if the consumption is expected to increase, decrease, or stay the same
mergeddata['consumption_growth_12m'] = mergeddata['forecast_cons_12m'] - mergeddata['cons_12m']

#check if  the forecasted consumption is consistent with past consumption, i.e., within a certain percentage threshold.
threshold = 0.1  # 10% threshold for consistency
mergeddata['consistency_flag'] = ((abs(mergeddata['cons_12m'] - mergeddata['forecast_cons_12m']) / mergeddata['cons_12m']) <= threshold).astype(int)
mergeddata['consistency_flag'] = mergeddata['consistency_flag'].fillna(0)

# Calculate the Monthly Average Consumption over the past 12 months
mergeddata['average_monthly_consumption_12m'] = mergeddata['cons_12m'] / 12.0

# Compare the last month's consumption to the Monthly Average Consumption
# to see if it has increased or decreased
mergeddata['change_in_monthlyconsumption'] = mergeddata['cons_last_month'] - mergeddata['average_monthly_consumption_12m']

# You might also want to express this change as a percentage
mergeddata['percentage_change_monthlyconsumption'] = (mergeddata['change_in_monthlyconsumption'] / mergeddata['average_monthly_consumption_12m']) * 100
# Replace infinite values with NaN, which can happen if average_monthly_consumption_12m is 0
mergeddata['percentage_change_monthlyconsumption'].replace(np.inf, np.nan, inplace=True)

# Fill NaN values if needed, for example with 0, or use dropna() to remove them
mergeddata['percentage_change_monthlyconsumption'] = mergeddata['percentage_change_monthlyconsumption'].fillna(0)


# Add 'Tenure' column measuring the time for which a client was active
mergeddata['Tenure'] = (mergeddata['date_end'] - mergeddata['date_activ']).dt.days
# For any missing end dates, assuming they are still active, use current date
current_date = pd.to_datetime('today')
mergeddata['Tenure'] = mergeddata['Tenure'].fillna((current_date - mergeddata['date_activ']).dt.days)

# Add columns to measure the max change in price for offpeak, peak, and mid peak
price_df['max_price_change_offpeak'] = price_df[['price_off_peak_var', 'price_off_peak_fix']].max(axis=1)
price_df['max_price_change_peak'] = price_df[['price_peak_var', 'price_peak_fix']].max(axis=1)
price_df['max_price_change_midpeak'] = price_df[['price_mid_peak_var', 'price_mid_peak_fix']].max(axis=1)


mergeddata = pd.merge(mergeddata, price_df[['id', 'max_price_change_offpeak', 'max_price_change_peak', 'max_price_change_midpeak']], on=['id'], how='left')


# Now save the processed DataFrame
mergeddata.to_csv('ProcessedClientData.csv', index=False)

correlation_matrix = mergeddata.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)


'''-----------
Modelling 
------------'''
mergeddata=pd.read_csv('ProcessedClientData.csv')


#encoding 'channel_sales' and 'origin_up' categorical columns to numerical
label_encoder_channel = LabelEncoder()
label_encoder_origin = LabelEncoder()
mergeddata['channel_sales_encoded'] = label_encoder_channel.fit_transform(mergeddata['channel_sales'])
mergeddata['origin_up_encoded'] = label_encoder_origin.fit_transform(mergeddata['origin_up'])

#drop categorical columns 

columns_to_drop = [
    "date_activ", "date_end", "date_modif_prod", "date_renewal",
    "id", "channel_sales", "origin_up","consistency_flag"
]
mergeddata = mergeddata.drop(columns=columns_to_drop)

print(mergeddata.dtypes)
mergeddata['has_gas'] = mergeddata['has_gas'].replace({'t': 1, 'f': 0})
mergeddata.info()
mergeddata.dropna(inplace=True)


'''Now we check for skeweness'''
skewness = mergeddata.skew()
print("Original Skewness:\n", skewness)

# List of columns to transform
skewed_columns = [
    'cons_12m',
    'cons_gas_12m',
    'cons_last_month',
    'forecast_cons_12m',
    'forecast_cons_year',
    'forecast_discount_energy',
    'forecast_meter_rent_12m',
    'forecast_price_energy_off_peak',
    'forecast_price_energy_peak',
    'forecast_price_pow_off_peak'
]

# Apply Yeo-Johnson transformation to each skewed column
for column in skewed_columns:
    # The Yeo-Johnson transformation can handle both positive and negative values
    mergeddata[column], _ = yeojohnson(mergeddata[column])

# Check the skewness after transformation
new_skewness = mergeddata[skewed_columns].skew()
print(new_skewness)


#check the correlation matrix
corr_matrix = mergeddata.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.show()

threshold = 0.9

# Find pairs of features that are highly correlated
highly_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname1 = corr_matrix.columns[i]  # Name of the first column
            colname2 = corr_matrix.columns[j]  # Name of the second column
            highly_corr_pairs.append((colname1, colname2))

# Print out the pairs of highly correlated features
for pair in highly_corr_pairs:
    print(f"Highly correlated pair: {pair[0]} and {pair[1]}")

#we have 70 columns we remove a few columns before building a model
#num_years_antig = antiquity of the client (in number of years) needs to be removed
#mean price change columns are highly correlated with other price change columns however we wish to test price sensitivity we will keep those values

mergeddata = mergeddata.drop(columns='num_years_antig')

#build the model
X = mergeddata.drop(columns='churn')  
y = mergeddata['churn']

#70-30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model to the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the model with selected features: {accuracy:.2%}")
#Accuracy is 100% right now that means the model is overfitting

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

# Print each cv score (accuracy) and average them
print(f"CV scores: {cv_scores}")
print(f"CV average score: {cv_scores.mean()}")

# Get feature importances
importances = rf_classifier.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})


feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

#forecast_meter_rent_12m and change_in_monthlyconsumption are key features contributing to churn
#price related features are not impacting churn directly

#hyperparameter tuning

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GridSearchCV object
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the grid search to the data
grid_search.fit(X, y)

# Print the best parameters and the best score
print(f"Best Parameters: {grid_search.best_params_}")
#Best Parameters: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
print(f"Best CV score: {grid_search.best_score_}")
#Best CV score: 0.9080216152867198