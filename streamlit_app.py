import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import io
import pdfplumber
import subprocess
import sys
import os
import time

# Create a virtual environment
venv_path = "./env"
required_packages = ['matplotlib', 'seaborn', 'scikit-learn', 'pandas', 'streamlit', 'pdfplumber']

# Create virtual environment if it doesn't exist
if not os.path.exists(venv_path):
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    subprocess.check_call([os.path.join(venv_path, "bin", "pip"), "install", "--upgrade", "pip"])
    subprocess.check_call([os.path.join(venv_path, "bin", "pip"), "install"] + required_packages)

# Update the system path to use the virtual environment
activate_script = os.path.join(venv_path, 'bin', 'activate')
os.environ['VIRTUAL_ENV'] = venv_path
os.environ['PATH'] = os.path.join(venv_path, 'bin') + os.pathsep + os.environ['PATH']

# Import required packages after ensuring they are installed
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import streamlit as st
    import pdfplumber
    from sklearn.linear_model import LinearRegression
    import sklearn
    st.write("scikit-learn version:", sklearn.__version__)
except ImportError as e:
    st.error(f"Failed to import a required package: {e}. Please ensure all dependencies are installed.")
    st.stop()

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

    # Summary Analysis
    st.subheader("Summary Statistics")
    if 'Status' in df.columns:
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
    else:
        st.error("The dataset does not contain a 'Status' column, which is required for summary statistics.")

    # List Price Distribution
    if 'List_Price' in df.columns:
        st.subheader("Distribution of List Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pd.to_numeric(df['List_Price'], errors='coerce').dropna(), bins=20, edgecolor='black', color='skyblue')
        ax.set_xlabel('List Price ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of List Prices')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain a 'List_Price' column for visualization.")

    # Sold Price Distribution
    if 'Sold_Price' in df.columns:
        st.subheader("Distribution of Sold Prices")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(pd.to_numeric(df['Sold_Price'][df['Sold_Price'] != 'Not Applicable'], errors='coerce').dropna(), bins=20, edgecolor='black', color='lightcoral')
        ax.set_xlabel('Sold Price ($)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Sold Prices')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain a 'Sold_Price' column for visualization.")

    # Scatter Plot: SqFt vs List Price
    if 'SqFt' in df.columns and 'List_Price' in df.columns:
        st.subheader("Property Size vs List Price")
        filtered_df_lp = df.dropna(subset=['SqFt', 'List_Price'])
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(filtered_df_lp['SqFt'], pd.to_numeric(filtered_df_lp['List_Price'], errors='coerce'), alpha=0.5, color='dodgerblue')
        ax.set_xlabel('SqFt')
        ax.set_ylabel('List Price ($)')
        ax.set_title('Property Size vs List Price')
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.error("The dataset does not contain both 'SqFt' and 'List_Price' columns for scatter plot visualization.")
