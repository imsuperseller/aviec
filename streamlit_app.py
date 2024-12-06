import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import io
import pdfplumber

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
            all_data = []
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    all_data.extend(table)
            
            if all_data:
                # Convert extracted table data to DataFrame, assuming first row as header
                df = pd.DataFrame(all_data[1:], columns=all_data[0])
                # Attempt to coerce numeric columns for analysis
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                st.write("PDF table has been extracted and converted to a DataFrame successfully.")
            else:
                st.write("No tabular data found in the uploaded PDF.")
                st.stop()

    # Replace NaN with 'Not Applicable' for readability in specific columns
    df.fillna('Not Applicable', inplace=True)

    # Display available columns for user reference
    st.subheader("Columns in Dataset")
    st.write(df.columns.tolist())

    # Check for required columns and provide detailed feedback
    required_columns = ['Status', 'MLS_ID', 'List_Price', 'Sold_Price', 'SqFt']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"The dataset is missing the following required columns: {', '.join(missing_columns)}. Please check your data and try again.")
        st.stop()

    # Summary Analysis
    st.subheader("Summary Statistics")
    summary_stats = df.groupby('Status').agg(
        Total_Listings=('MLS_ID', 'count'),
        Avg_List_Price=('List_Price', lambda x: pd.to_numeric(x, errors='coerce').mean() if not x.empty else 'Not Applicable'),
        Avg_Sold_Price=('Sold_Price', lambda x: pd.to_numeric(x, errors='coerce').mean() if not x.empty else 'Not Applicable'),
        Avg_Sale_List_Ratio=('Sale_List_Ratio', lambda x: pd.to_numeric(x.str.replace('%', '', regex=True), errors='coerce').mean() if not x.empty else 'Not Applicable'),
        Median_List_Price=('List_Price', lambda x: pd.to_numeric(x, errors='coerce').median() if not x.empty else 'Not Applicable'),
        Median_Sold_Price=('Sold_Price', lambda x: pd.to_numeric(x, errors='coerce').median() if not x.empty else 'Not Applicable')
    ).reset_index()

    # Replace NaN with readable values in summary stats after calculations
    summary_stats.fillna('Not Applicable', inplace=True)
    st.write(summary_stats)

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
    ax.hist(pd.to_numeric(df['List_Price'], errors='coerce').dropna(), bins=20, edgecolor='black', color='skyblue')
    ax.set_xlabel('List Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of List Prices')
    ax.grid(True)
    st.pyplot(fig)
