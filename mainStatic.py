import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import io
import sys
import pandas as pd
import altair as alt
import sys
import warnings
import numpy as np
import plotly.graph_objects as go
warnings.filterwarnings("ignore")


def yearsDashboardForYear(year):
    
    if year == 2024:
        df = pd.read_csv(f"E:/Fiverr/1st/Data/Actual_Sales_2023.csv")
        df = df.loc[:, df.columns.str.contains('2024')]
        df['Grand Total'] = df.sum(axis=1)
    elif year == 2023:
        df = pd.read_csv(f"E:/Fiverr/1st/Data/Actual_Sales_2023.csv")
        df = df.loc[:, df.columns.str.contains('2023')]
        df['Grand Total'] = df.sum(axis=1)
    else:
        df = pd.read_csv(f"E:/Fiverr/1st/Data/Actual_Sales_{year}.csv")

    return df

def getFactualAnalysis(df):
    

    if 'Row Labels' in df.columns:
        df = df.drop(columns=['Row Labels'])

    min_val = df.iloc[:-1, :-1].min().min()
    min_location = df.iloc[:-1, :-1].stack().idxmin()

    max_val = df.iloc[:-1, :-1].max().max()
    max_location = df.iloc[:-1, :-1].stack().idxmax()

    mean_val = df.iloc[:-1, :-1].mean().mean()

    total_sales = df.at[9, 'Grand Total']

    most_profitable_line = df.iloc[:-1, df.columns.get_loc('Grand Total')].idxmax()

    least_profitable_line = df.iloc[:-1, df.columns.get_loc('Grand Total')].idxmin()
    
    leftCol = {'Minimum Value': min_val,
        'minLocation:': min_location,
        'Maximum Value': max_val,
        'maxLocation:': max_location}
    
    rightCol = {
        'Mean Value': mean_val,
        'Total Sales': total_sales,
        'Most Profitable Line': most_profitable_line,
        'Least Profitable Line': least_profitable_line
    }

    return leftCol, rightCol

def heatMapGrowthRate(df):
    df = df.fillna(0)
    growth_rates = df.iloc[:, 1:-1].pct_change(axis=1) * 100
    growth_rates = growth_rates.dropna(axis=1, how='all')

    new_column = ['Line 1', 'Line 1-2', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6', 'Line 7', 'SGB', 'Grand Total']
    growth_rates.insert(0, 'Row Labels', new_column)

    growth_rates_melted = growth_rates.melt(id_vars=['Row Labels'], var_name='Week', value_name='Growth Rate')

    fig = px.imshow(
        growth_rates.set_index('Row Labels').T,  
        labels=dict(x="Week", y="Production Line", color="Growth Rate (%)"),
        color_continuous_scale='RdBu',  
        aspect='auto'  
    )

    fig.update_layout(
        title='Week-on-Week Growth Rates Across Production Lines',
        xaxis_title='Week',
        yaxis_title='Production Line',
        coloraxis_colorbar=dict(title='Growth Rate (%)')
    )

    st.plotly_chart(fig)
    
def trendsBarChart(df):
    df = df.fillna(0)
    trend_series = df.iloc[:, 1:-1].mean(axis=1)

    trend_df = pd.DataFrame({
        'Production Line': ['Line 1', 'Line 1-2', 'Line 2', 'Line 3', 'Line 4', 'Line 5', 'Line 6', 'Line 7', 'SGB'],
        'Average Value': trend_series.values[:9]
    })

    fig = px.bar(
        trend_df,
        x='Production Line',  # Production lines on the x-axis
        y='Average Value',    # Average trend values on the y-axis
        text='Average Value', # Display the average value on top of each bar
        title='Average Trend Across Weeks for Each Production Line',
        labels={'Average Value': 'Average Value Across Weeks'},  # Y-axis label
        color='Average Value',  # Color bars based on their value
        color_continuous_scale='Blues'  # Color scale to show intensity
    )

    fig.update_layout(
        xaxis_title='Production Line',
        yaxis_title='Average Value',
        coloraxis_showscale=False  # Hide the color scale since it's redundant
    )

    st.plotly_chart(fig)

def mainDashboardFunc():
    alt.themes.enable("dark")

    st.title("Annual Analysis Dashboard")
    selected_year = st.selectbox(
        'Select Year',
        options=[2020, 2021, 2022, 2023, 2024],  # Added 2024
        index=4  # Default to 2024
    )

    df = yearsDashboardForYear(selected_year)

    col = st.columns((1.5, 4.5, 2), gap='medium', vertical_alignment="top")
    left, right = getFactualAnalysis(df)

    with col[0]:
        st.markdown('##### Factual Counts')
        for key, value in left.items():
            if isinstance(value, tuple):
                if value[0] == 8:
                    string = f'SGB, Week {value[1]}'
                else:
                    string = f'Line {value[0]}, Week {value[1]}'
                left[key] = string

        for key, value in left.items():
            st.write(key, value)


    with col[1]:
        trendsBarChart(df)
        heatMapGrowthRate(df)

    with col[2]:
        df = pd.DataFrame.from_dict(right, orient='index', columns=['Values'])
        df = df.reset_index()
        df.columns = ['Features', 'Values']
        # df['Values'] = pd.to_numeric(df['Values'], errors='coerce')
        index = [1, 2, 3, 4]
        df.index = index

    #     st.markdown('#### Other Findings')
    #     st.dataframe(df)
        st.markdown('#### Other Findings')

        # Configure the columns
        column_config = {
            "Features": st.column_config.TextColumn("Feature Name"),
            "Values": st.column_config.ProgressColumn(
                "Value Count",
                format="%d",
                min_value=0,
                max_value=df["Values"].max()  # Adjust to the maximum value in your 'count' column
            )
        }

        # Display the DataFrame with custom column configurations
        st.dataframe(df,
                     hide_index=True,
                     column_config=column_config
                    )

        # Add an expander with additional information
        with st.expander('About', expanded=True):
            st.write('''
                - :orange[**Data Source**]: Yearly Sales Dataset 
                - :orange[**Value Count**]: The Findings in dataset
                ''')
            
if __name__ == "__main__":
    mainDashboardFunc()