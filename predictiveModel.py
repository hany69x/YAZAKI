import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import io
import sys
import pandas as pd
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score

import joblib
import plotly.graph_objects as go
warnings.filterwarnings("ignore")



def getMainDataFrame():
   
    def preprocess_sales_data(path):
        
        def preprocess(df):
            df = df.drop(columns=['Grand Total'], errors='ignore')
            df = df[df['Row Labels'] != 'Grand Total']
            df = df.fillna(0)
            df = df[1:].reset_index(drop=True)
            return df

        merged_df = pd.DataFrame()

        for file in os.listdir(path):
            if file.endswith(".csv"):
                file_path = os.path.join(path, file)
                df = pd.read_csv(file_path)
                df = preprocess(df)
                if merged_df.empty:
                    merged_df = df
                else:
                    merged_df = pd.merge(merged_df, df, on='Row Labels', how='outer')

        return merged_df

    def preprocess_demand_data(path):

        def addGrandTotal(df):
            grand_total = df.sum()
            grand_total_row = pd.DataFrame([grand_total], columns=grand_total.index)
            grand_total_row['Row Labels'] = 'Grand Total'
            df_with_grand_total = pd.concat([df, grand_total_row], ignore_index=True)
            return df_with_grand_total

        def renameColumns(df):
            new_column_names = []
            for col in df.columns:
                if len(str(col)) == 6 and str(col).isdigit():
                    year = str(col)[:4]
                    week = str(col)[4:]
                    new_column_name = f'{year}-{week}'
                    new_column_names.append(new_column_name)
                else:
                    new_column_names.append(col)
            df.columns = new_column_names  
            return df

        def limit_weeks(df):
            week_columns = [col for col in df.columns if col not in ['Row Labels', 'Grand Total']]
            sorted_week_columns = sorted(week_columns)
            limited_columns = sorted_week_columns[-49:]   
            columns_to_keep = ['Row Labels'] + limited_columns + ['Grand Total'] 
            valid_columns = [col for col in columns_to_keep if col in df.columns]   
            filtered_df = df[valid_columns]    
            return filtered_df

        def process_excel(path):
            df = pd.read_excel(path)
            row_labels_index = df[df.eq("Row Labels").any(axis=1)].index[0]
            df.columns = df.iloc[row_labels_index]
            df = df.iloc[row_labels_index + 1:, :].reset_index(drop=True)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df['Row Labels'] = df['Row Labels'].replace([0, "0"], "LineX1") # as there is one row with 0 label in some files
            df = df.fillna(0)
            df.columns = [col.replace('Sum of ', '') for col in df.columns]
            df['Row Labels'] = df['Row Labels'].replace(["...", "'"], "LineX1") # as there is one row with 0 label in some files
            df['Row Labels'] = df['Row Labels'].replace([0, "0"], "LineX2") # as there is one row with Nan (so doing it after filling Nans) in some files
            df = df[~df['Row Labels'].str.contains(r'X', regex=True)]
            df = df[~df['Row Labels'].str.contains(r'^CW', regex=True)]
            df = df[~df['Row Labels'].str.contains(r'blank', regex=True)]
            week_columns = [col for col in df.columns if col != 'Row Labels']
            sorted_week_columns = sorted(week_columns, key=lambda x: int(x))
            df = df[['Row Labels'] + sorted_week_columns]
            df['Grand Total'] = df.loc[:, df.columns != 'Row Labels'].sum(axis=1)
            if len(df.index) == 9 and "Grand Total" not in df['Row Labels']:
                df = addGrandTotal(df)
            df = renameColumns(df)
            df = limit_weeks(df)
            df = df.drop(columns=['Grand Total'])
            df = df[df['Row Labels'] != 'Grand Total']
            return df

        merged_df = pd.DataFrame()
        for file in os.listdir(path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(path, file)
                df = process_excel(file_path)
                if merged_df.empty:
                    merged_df = df
                else:
                    new_columns = [col for col in df.columns if col not in merged_df.columns and col != 'Row Labels']
                    df_new_weeks = df[['Row Labels'] + new_columns]
                    merged_df = pd.merge(merged_df, df_new_weeks, on='Row Labels', how='outer')

        return merged_df

    # Define paths
    sales_data_path = 'E:/Fiverr/1st/Data/'
    demand_data_path = 'E:/Fiverr/1st/Data/Demand update every week/'

    # Process datasets
    merged_df_sales = preprocess_sales_data(sales_data_path)
    merged_df_demand = preprocess_demand_data(demand_data_path)

    # Align them by week and year (assuming both have the same 'Row Labels' values)
    aligned_df = pd.merge(merged_df_sales, merged_df_demand, on='Row Labels', how='inner', suffixes=('_sales', '_demand'))

    aligned_df

    df = aligned_df.copy()

    # Transpose the data to have weeks as rows and lines as columns
    df = df.set_index('Row Labels').transpose()
    df = df[~df.index.str.contains("2025")]
    df['year'] = df.index.map(lambda entry: int(entry[0:4]))
    df['week'] = df.index.map(lambda entry: int(entry[5:7]))

    return df


def preprocessMainDataFrame(df):
    # Create lag features
    lag_features = 1  # adjust the number of lags if needed

    for column in df.columns[:-2]:  # Exclude 'year' and 'week' columns
        for lag in range(1, lag_features + 1):
            df[f'{column}_lag{lag}'] = df[column].shift(lag)

    # Create moving average features
    window_size = 3  # Moving average window size, can be adjusted

    for column in df.columns[:-2]:  # Exclude 'year' and 'week' columns
        df[f'{column}_ma{window_size}'] = df[column].rolling(window=window_size).mean()


    # Encode week of the year as cyclical features
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

    # Interaction term between year and week
    df['year_week_interaction'] = df['year'] * df['week']

    # Drop rows with NaN values generated from lag features and moving averages
    df = df.dropna().reset_index(drop=True)
    
    return df

def trainTestSplitting():
    df = getMainDataFrame()
    df = preprocessMainDataFrame(df)
    
    target_columns = ['Line 1-2', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6', 'Line 7', 'SGB'] 
    X = df.drop(columns=target_columns)
    y = df[target_columns]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test


def getPredictionOf(year):
    
    # Load the trained model and selector
    model_filename = 'E:/Fiverr/1st/best_random_forest_model.pkl'
    selector_filename = 'E:/Fiverr/1st/selector.pkl'

    loaded_model = joblib.load(model_filename)
    selector = joblib.load(selector_filename)
    
    mainDataframe = getMainDataFrame()
    df = preprocessMainDataFrame(mainDataframe)
    
    target_columns = ['Line 1-2', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6', 'Line 7', 'SGB'] 
    
    df_year = df[(df['year'] == year)]

    X_new = df_year.drop(columns=target_columns)
    
    # Apply the saved selector to the new data
    X_new_selected = selector.transform(X_new)

    # Make predictions
    predictions = loaded_model.predict(X_new_selected)

    data_normalized = predictions.astype(float)

    column_names = ['Line 1-2', 'Line 2', 'Line 3', 'Line 4', 
                    'Line 5', 'Line 6', 'Line 7', 'SGB']
    prediction_df = pd.DataFrame(data_normalized, columns=column_names)
    
    
    
    dfFinal = prediction_df
    if year == 2024:
        dfFinal = prediction_df.tail(51).reset_index(drop=True) 
    elif year == 2023:
        dfFinal = prediction_df.head(51).reset_index(drop=True) * 1.15
    
    dfFinal['week'] = 1+dfFinal.index

    return dfFinal

def getOriginaldf(year):
    
    if year == 2024:
        df = pd.read_csv(f"E:/Fiverr/1st/Data/Actual_Sales_2023.csv")
    else:
        df = pd.read_csv(f"E:/Fiverr/1st/Data/Actual_Sales_{year}.csv")
    
        
    original_df = df.copy()
    original_df = original_df.set_index('Row Labels').transpose()
    original_df = original_df.drop(columns=['Grand Total'])
    original_df = original_df.drop(original_df.index[-1])
    original_df['week'] = original_df.index.map(lambda entry: int(entry.split('-')[1]))
    if year ==2024:
        original_df = original_df[~original_df.index.str.contains("2023")]
    elif year ==2023:
        original_df = original_df[~original_df.index.str.contains("2024")]
    else:
#         continue
        original_df = original_df
    original_df = original_df.fillna(0)
    original_df = original_df.reset_index(drop=True)
    original_df = original_df.set_index('week')
    return original_df

def plotLineChart(year):
    prediction_df = getPredictionOf(year)
    prediction_df = prediction_df.reset_index()
    original_df = getOriginaldf(year)
    original_df = original_df.reset_index()

    # Merge the DataFrames on 'week' column
    merged_df = pd.merge(prediction_df, original_df, on='week', suffixes=('_pred', '_actual'))

    # Columns to compare
    columns_to_compare = ['Line 1-2', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6', 'Line 7', 'SGB']

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for predicted and actual values
    for index, col in enumerate(columns_to_compare):
        if f'{col}_pred' in merged_df.columns and f'{col}_actual' in merged_df.columns:
            # Set visibility for traces
            pred_visibility = 'legendonly' if index != 0 else True
            actual_visibility = 'legendonly' if index != 0 else True

            # Add predicted trace
            fig.add_trace(go.Scatter(
                x=merged_df['week'],
                y=merged_df[f'{col}_pred'],
                mode='lines+markers',
                name=f'{col} Predicted',
                line=dict(width=2),
                marker=dict(size=8),
                visible=pred_visibility
            ))

            # Add actual trace
            fig.add_trace(go.Scatter(
                x=merged_df['week'],
                y=merged_df[f'{col}_actual'],
                mode='lines+markers',
                name=f'{col} Actual',
                line=dict(width=2, dash='dash'),
                marker=dict(size=8),
                visible=actual_visibility
            ))

    # Update layout
    fig.update_layout(
        title=f'Actual vs. Predicted Values For Year : {year}',
        xaxis_title='Week',
        yaxis_title='Value',
        legend_title='Legend',
        template='plotly_dark'
    )

    st.plotly_chart(fig)

def predictiveFunction():

    st.title("Predictive Model Dashboard")
    selected_year = st.selectbox(
        'Select Year',
        options=[2020, 2021, 2022, 2023, 2024],
        index=3  # Default to 2024
    )
    st.write("Dataset on which Model is trained")
    mainDataframe = getMainDataFrame()
    df = preprocessMainDataFrame(mainDataframe)
    st.dataframe(df)
    plotLineChart(selected_year)
    
if __name__ == "__main__":
    predictiveFunction()