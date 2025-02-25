# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from fpdf import FPDF
import io

# App title
st.title("Customer Churn Prediction Dashboard")
st.sidebar.title("Options")

# Initialize variables
model = None
df = None

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview", df.head())

# Handle missing values
if df is not None and st.sidebar.checkbox("Handle Missing Values"):
    if df.isnull().values.any():
        for col in df.select_dtypes(include=[np.number]).columns:  # Only apply mean to numerical columns
            df[col].fillna(df[col].mean(), inplace=True)
        st.write("Missing values handled (numerical columns filled with mean).")
    else:
        st.write("No missing values found.")

# Data preprocessing for categorical columns
if df is not None:
    # Drop irrelevant columns (e.g., customerID)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)
        st.write("Dropped 'customerID' column.")

    # Identify and encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
        st.write("Categorical columns encoded:", list(categorical_columns))

# Data summary
if df is not None and st.sidebar.checkbox("Show Data Summary"):
    st.write("### Data Summary")
    st.write(df.describe())

# Correlation heatmap
if df is not None and st.sidebar.checkbox("Show Correlation Heatmap"):
    st.write("### Correlation Heatmap")
    # Filter numerical columns
    numerical_df = df.select_dtypes(include=['number'])
    if not numerical_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.write("No numerical columns available for correlation heatmap.")

# Churn distribution
if df is not None and st.sidebar.checkbox("Show Churn Distribution"):
    st.write("### Churn Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    df['Churn'].value_counts().plot(kind='bar', color=['blue', 'orange'], ax=ax)
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Count")
    st.pyplot(fig)

# Train model
if df is not None and st.sidebar.checkbox("Train Model"):
    st.write("### Model Training")
    
    # Model selection
    model_type = st.sidebar.selectbox("Select Model", ("Random Forest", "Decision Tree"))

    # Preprocessing: Separate features and target
    if "Churn" in df.columns:
        X = df.drop(columns=["Churn"])
        y = df["Churn"]

        # Ensure target column is numeric
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0})

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model initialization
        if model_type == "Random Forest":
            model = RandomForestClassifier(random_state=42)
        else:
            model = DecisionTreeClassifier(random_state=42)

        # Train model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Metrics
        st.write(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
        st.write(f"Precision: {precision_score(y_test, predictions):.2f}")
        st.write(f"Recall: {recall_score(y_test, predictions):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, predictions):.2f}")
    else:
        st.error("No 'Churn' column found in dataset.")

# Predict on new data

if uploaded_file and model and st.sidebar.checkbox("Make Prediction on New Data"):
    st.write("### Predict Customer Churn")

    # Input form
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    rev_per_month = st.number_input("Monthly Revenue", min_value=0.0, max_value=500.0, value=50.0)
    cashback = st.number_input("Cashback", min_value=0.0, max_value=100.0, value=10.0)
    service_score = st.slider("Service Score", 1, 10, 5)

    # Prepare input data
    input_data = pd.DataFrame([[tenure, rev_per_month, cashback, service_score]],
                              columns=['Tenure', 'rev_per_month', 'cashback', 'service_score'])

    # Ensure feature order matches training data
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)
        result = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
        st.write(f"Prediction: {result}")


# Generate PDF Report
if df is not None and model is not None and st.sidebar.checkbox("Generate PDF Report"):
    st.write("### Download Report")

    # Function to create a PDF report
    def generate_pdf_report(data_summary, model_metrics):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=True, align="C")

        # Data Summary
        pdf.cell(200, 10, txt="Data Summary:", ln=True)
        for key, value in data_summary.items():
            pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

        # Model Metrics
        pdf.cell(200, 10, txt="Model Metrics:", ln=True)
        for metric, score in model_metrics.items():
            pdf.cell(200, 10, txt=f"{metric}: {score:.2f}", ln=True)

        # Save PDF to buffer (S returns a string)
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer, 'S')  # 'S' returns PDF as a string
        pdf_buffer.seek(0)  # Reset buffer position
        return pdf_buffer
        
        

    # Prepare report content
    data_summary = {
        "Rows": df.shape[0],
        "Columns": df.shape[1]
    }
    model_metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    }

    # Generate and download PDF
    if st.button("Generate Report"):
        pdf_buffer = generate_pdf_report(data_summary, model_metrics)
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer.getvalue(),  # Pass the buffer's content as raw bytes
            file_name="churn_report.pdf",
            mime="application/pdf"
        )


# Help Section
if st.sidebar.checkbox("Help"):
    st.write("### Help")
    st.write("""
        1. Upload a CSV file with customer data.
        2. Use the sidebar options to explore, visualize, and analyze the data.
        3. Train a machine learning model to predict customer churn.
        4. Enter new customer data to get a churn prediction.
        5. Download a PDF report summarizing the analysis and predictions.
    """)
