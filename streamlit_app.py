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
if not os.path.exists(venv_path):
    subprocess.check_call([sys.executable, "-m", "venv", venv_path])
    subprocess.check_call([os.path.join(venv_path, "bin", "pip"), "install", "--upgrade", "pip"])

# Use the Python and Pip from the virtual environment
python_path = os.path.join(venv_path, "bin", "python")
pip_path = os.path.join(venv_path, "bin", "pip")

# Ensure required packages are installed in virtual environment
required_packages = ['matplotlib', 'seaborn', 'scikit-learn', 'pandas', 'streamlit', 'pdfplumber']
for package in required_packages:
    for attempt in range(3):  # Retry logic for package installation
        try:
            __import__(package)
            break  # If import is successful, break out of retry loop
        except ImportError:
            if attempt < 2:  # Retry only for the first two attempts
                st.write(f"Attempting to install missing package: {package} (Attempt {attempt + 1}/3)")
                try:
                    subprocess.check_call([pip_path, "install", package])
                    __import__(package)  # Try importing again after installation
                    st.write(f"Successfully installed {package}")
                    break
                except (subprocess.CalledProcessError, ImportError) as e:
                    st.error(f"Failed to install {package} on attempt {attempt + 1}. Retrying...")
                    time.sleep(2)  # Wait before retrying
            else:
                st.error(f"Failed to install {package} after 3 attempts. Please check your environment.")
                # Print environment variables for debugging
                st.write("Environment PATH:", os.environ["PATH"])
                st.write("Python Path:", python_path)
                st.write("Pip Path:", pip_path)
                # Check if pip list shows scikit-learn
                try:
                    installed_packages = subprocess.check_output([pip_path, "list"]).decode("utf-8")
                    st.write("Installed Packages:\n", installed_packages)
                except subprocess.CalledProcessError as e:
                    st.error(f"Failed to list installed packages. Error: {e}")
                st.stop()

# Import after ensuring the package is installed
try:
    from sklearn.linear_model import LinearRegression
    import sklearn
    st.write("scikit-learn version:", sklearn.__version__)
except ImportError:
    st.error("Failed to import scikit-learn after multiple installation attempts. Exiting the program.")
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
    ax.hist(pd.to_numeric(df['List_Price'], errors='coerce').dropna(), bins=20, edgecolor='black', color='skyblue')
    ax.set_xlabel('List Price ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of List Prices')
    ax.grid(True)
    st.pyplot(fig)

    # Sold Price Distribution
    st.subheader("Distribution of Sold Prices")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pd.to_numeric(df['Sold_Price'][df['Sold_Price'] != 'Not Applicable'], errors='coerce').dropna(), bins=20, edgecolor='black', color='lightcoral')
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
    x = filtered_df_lp['SqFt'].values
