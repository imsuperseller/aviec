import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st
import subprocess
import sys
import io

# Ensure required packages are installed
def install_missing_packages():
    required_packages = ['matplotlib', 'seaborn', 'scikit-learn', 'pandas', 'streamlit', 'pdfplumber']
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            st.write(f"Installing missing package: {package}")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', package])
                try:
                    __import__(package)
                    st.write(f"Successfully installed {package}")
                except ImportError:
                    st.error(f"Failed to import {package} even after installation. Please check your environment.")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to install {package}. Error: {e}")

install_missing_packages()

import pdfplumber  # Import after ensuring the package is installed

# Streamlit setup
st.title("Real Estate Analysis Tool")
st.write("This tool provides insights into property listings data, offering visualizations and trends for better decision-making.")

# Upload CSV or PDF file
uploaded_file = st.file_uploader("Upload your CSV or PDF file", type=["csv", "pdf"])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.pdf'):
        # Extract data from PDF using pdfplumber
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            pages = pdf.pages
            text = "\n".join(page.extract_text() for page in pages if page.extract_text())
            st.write("PDF content has been extracted successfully.")
            # Placeholder for further processing, converting text to a DataFrame, if possible
            st.write("Currently, only CSV files are fully supported for data analysis.")
            st.stop()

    # Replace NaN with 'Not Applicable' for readability in specific columns
    df.fillna({'Sold_Price': 'Not Applicable', 'Sale_List_Ratio': 'Not Applicable', 'Sold_Date': 'Not Applicable'}, inplace=True)

    # Summary Analysis
    st.subheader("Summary Statistics")
    summary_stats = df.groupby('Status').agg(
        Total_Listings=('MLS_ID', 'count'),
        Avg_List_Price=('List_Price', lambda x: pd.to_numeric(x, errors='coerce').mean()),
        Avg_Sold_Price=('Sold_Price', lambda x: pd.to_numeric(x, errors='coerce').mean()),
        Avg_Sale_List_Ratio=('Sale_List_Ratio', lambda x: pd.to_numeric(x.str.replace('%', '', regex=True), errors='coerce').mean()),
        Median_List_Price=('List_Price', lambda x: pd.to_numeric(x, errors='coerce').median()),
        Median_Sold_Price=('Sold_Price', lambda x: pd.to_numeric(x, errors='coerce').median())
    ).reset_index()

    # Replace NaN with readable values in summary stats after calculations
    summary_stats.fillna({'Avg_Sold_Price': 'Not Applicable', 'Avg_Sale_List_Ratio': 'Not Applicable'}, inplace=True)
    st.write(summary_stats)

    # Calculate Average DOM for each Status if the column 'CDOM' exists
    if 'CDOM' in df.columns:
        summary_stats['Avg_DOM'] = df.groupby('Status')['CDOM'].mean().reset_index(drop=True)
    else:
        summary_stats['Avg_DOM'] = 'Not Applicable'

    # Count the number of properties with pools by Status if the 'Pool' column exists
    if 'Pool' in df.columns:
        summary_stats['Properties_with_Pool'] = df[df['Pool'] == 'Yes'].groupby('Status')['MLS_ID'].count().reindex(summary_stats['Status']).fillna(0).reset_index(drop=True)
    else:
        summary_stats['Properties_with_Pool'] = 0

    # Save the results to a CSV file for easier sharing
    summary_stats.to_csv('updated_property_statistics.csv', index=False)

    # Display a few rows of the final DataFrame for verification
    st.subheader("Extracted Listings Data")
    st.write(df.head())

    # Basic Visualization
    sns.set(style="whitegrid")

    # Plot Total Listings by Status
    st.subheader("Total Property Listings by Status")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x='Status', y='Total_Listings', data=summary_stats, palette='viridis', hue='Status', dodge=False, ax=ax)
    ax.set_xlabel('Status')
    ax.set_ylabel('Total Listings')
    ax.set_title('Total Property Listings by Status')
    for index, value in enumerate(summary_stats['Total_Listings']):
        ax.text(index, value + 0.1, str(value), ha='center', fontsize=10, color='black')
    st.pyplot(fig)

    # List Price Distribution
    st.subheader("Distribution of List Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['List_Price'].dropna(), bins=20, edgecolor='black', color='skyblue')
    ax.set_xlabel('List Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of List Prices')
    ax.grid(True)
    st.pyplot(fig)

    # Sold Price Distribution
    st.subheader("Distribution of Sold Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Sold_Price'][df['Sold_Price'] != 'Not Applicable'].dropna().astype(float), bins=20, edgecolor='black', color='lightcoral')
    ax.set_xlabel('Sold Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Sold Prices')
    ax.grid(True)
    st.pyplot(fig)

    # Scatter Plot: SqFt vs List Price
    st.subheader("Property Size vs List Price")
    filtered_df_lp = df.dropna(subset=['SqFt', 'List_Price'])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(filtered_df_lp['SqFt'], pd.to_numeric(filtered_df_lp['List_Price'], errors='coerce'), alpha=0.5, color='dodgerblue')
    ax.set_xlabel('SqFt')
    ax.set_ylabel('List Price ($)')
    ax.set_title('Property Size vs List Price')
    ax.grid(True)
    x = filtered_df_lp['SqFt'].values.reshape(-1, 1)
    y = pd.to_numeric(filtered_df_lp['List_Price'], errors='coerce').values.reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    ax.plot(filtered_df_lp['SqFt'], model.predict(x), color='red', linewidth=2)
    st.pyplot(fig)

    # Scatter Plot: SqFt vs Sold Price
    st.subheader("Property Size vs Sold Price")
    filtered_df_sp = df[(df['Sold_Price'] != 'Not Applicable') & df['SqFt'].notna()]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(filtered_df_sp['SqFt'], pd.to_numeric(filtered_df_sp['Sold_Price'], errors='coerce'), alpha=0.5, color='orange')
    ax.set_xlabel('SqFt')
    ax.set_ylabel('Sold Price ($)')
    ax.set_title('Property Size vs Sold Price')
    ax.grid(True)
    x_sold = filtered_df_sp['SqFt'].values.reshape(-1, 1)
    y_sold = pd.to_numeric(filtered_df_sp['Sold_Price'], errors='coerce').values.reshape(-1, 1)
    model_sold = LinearRegression().fit(x_sold, y_sold)
    ax.plot(filtered_df_sp['SqFt'], model_sold.predict(x_sold), color='red', linewidth=2)
    st.pyplot(fig)

    # Average Days on Market (DOM) by Status (if available)
    if 'Avg_DOM' in summary_stats.columns and not isinstance(summary_stats['Avg_DOM'].iloc[0], str):
        st.subheader("Average Days on Market by Status")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='Status', y='Avg_DOM', data=summary_stats, palette='coolwarm', hue='Status', dodge=False, ax=ax)
        ax.set_xlabel('Status')
        ax.set_ylabel('Average Days on Market (DOM)')
        ax.set_title('Average Days on Market by Status')
        for index, value in enumerate(summary_stats['Avg_DOM']):
            ax.text(index, value + 0.5, str(round(value, 1)), ha='center', fontsize=10, color='black')
        st.pyplot(fig)
