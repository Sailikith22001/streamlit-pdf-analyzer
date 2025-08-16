import streamlit as st
import pandas as pd
import fitz  # PyMuPDF library
import json
import traceback
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import re
from collections import Counter
import io
import base64

# Import Google Cloud libraries and set a flag for availability
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, Part
    IS_GCP_AVAILABLE = True
except ImportError:
    IS_GCP_AVAILABLE = False

# --- Custom CSS for Professional Styling ---
st.markdown("""
    <style>
    /* General Styling */
    body {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0052a3;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    .stFileUploader {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 16px;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        color: #374151;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e5e7eb;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #0066cc;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
        padding: 24px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    .stAlert {
        border-radius: 8px;
        padding: 16px;
    }
    .stSpinner {
        margin: 24px auto;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    .stTextInput>label, .stSelectbox>label, .stCheckbox>label {
        font-weight: 500;
        color: #374151;
    }
    .stTextInput input, .stSelectbox select {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 10px;
    }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
    }
    .stDownloadButton>button:hover {
        background-color: #059669;
    }
    /* Header Styling */
    h1, h2, h3 {
        color: #1f2937;
        font-weight: 600;
    }
    .stDivider {
        margin: 24px 0;
        background-color: #e5e7eb;
    }
    /* Card-like sections */
    .section-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Core Application Functions ---
def initialize_vertex_ai():
    """
    Initializes the Vertex AI SDK using credentials stored in Streamlit's secrets.
    This is the secure way to authenticate with your Google Cloud project.
    """
    try:
        if "gcp" not in st.secrets or "project" not in st.secrets.gcp:
            st.error("GCP 'project' ID not found in Streamlit secrets (.streamlit/secrets.toml).")
            return False
        
        project_id = st.secrets.gcp["project"]
        location = st.secrets.gcp.get("location", "us-central1")
        vertexai.init(project=project_id, location=location)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Google Vertex AI: {e}")
        st.code(traceback.format_exc())
        return False

def get_info_from_gemini(text_blob: str) -> dict:
    """
    Sends the extracted PDF text to a stable Google Gemini model
    to get structured information as a JSON object.
    """
    try:
        model = GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
        **Task**: Analyze the text from a bank statement and extract key information.

        **Input Text**:
        ---
        {text_blob}
        ---

        **Instructions**:
        1. Carefully read the text to identify the account holder's details and the bank's details.
        2. Return this information in a single, valid JSON object.
        3. **Crucially, your entire response must only be the JSON object itself.**
        4. If a specific piece of information (e.g., "E-mail") cannot be found, use "N/A".

        **Required JSON Format**:
        {{
          "Account Holder Details": {{
            "Name": "...",
            "Address": "...",
            "Contact Nr": "...",
            "E-mail": "..."
          }},
          "Bank Details": {{
            "Bank Name": "...",
            "Bank Address": "...",
            "Account Number": "..."
          }}
        }}
        """
        response = model.generate_content([Part.from_text(prompt)])
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response_text)
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        st.code(traceback.format_exc())
        return {}

def process_pdf(uploaded_file):
    """
    Uses PyMuPDF to extract text and tabular data from a PDF.
    """
    full_text = ""
    tables_found = []
    try:
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            for page in doc:
                full_text += page.get_text("text", sort=True) + "\n"
                for table in page.find_tables():
                    df = table.to_pandas()
                    if not df.empty and len(df) > 1:
                        tables_found.append(df)
        
        if tables_found:
            transaction_df = max(tables_found, key=len)
            return full_text, transaction_df
        else:
            return full_text, pd.DataFrame()
    except Exception as e:
        st.error(f"Failed to process PDF: {e}")
        st.code(traceback.format_exc())
        return "", pd.DataFrame()

@st.cache_data
def convert_df_to_csv(_df):
    """Convert DataFrame to CSV string for download."""
    return _df.to_csv(index=False).encode('utf-8')

def get_spending_insights_from_gemini(text_blob: str, transaction_data: str = "") -> dict:
    """
    Get spending insights and categorization from Gemini AI.
    """
    try:
        model = GenerativeModel("gemini-2.5-flash-lite")
        combined_data = f"{text_blob}\n\nTransaction Data:\n{transaction_data}" if transaction_data else text_blob
        prompt = f"""
        **Task**: Analyze the bank statement and provide detailed spending insights.

        **Input Data**:
        ---
        {combined_data}
        ---

        **Instructions**:
        1. Analyze all transactions and spending patterns
        2. Categorize expenses (Food, Shopping, Bills, Entertainment, etc.)
        3. Identify largest expenses and frequent merchants
        4. Calculate spending trends and patterns
        5. Return insights in JSON format only

        **Required JSON Format**:
        {{
          "spending_summary": {{
            "total_debits": "...",
            "total_credits": "...",
            "net_balance_change": "...",
            "transaction_count": "..."
          }},
          "expense_categories": {{
            "Food & Dining": "...",
            "Shopping": "...",
            "Bills & Utilities": "...",
            "Entertainment": "...",
            "Transportation": "...",
            "Healthcare": "...",
            "Others": "..."
          }},
          "top_merchants": [
            {{"name": "...", "amount": "...", "frequency": "..."}},
            {{"name": "...", "amount": "...", "frequency": "..."}}
          ],
          "spending_insights": [
            "Insight 1...",
            "Insight 2...",
            "Insight 3..."
          ]
        }}
        """
        response = model.generate_content([Part.from_text(prompt)])
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response_text)
    except Exception as e:
        st.warning(f"Could not generate spending insights: {e}")
        return {}

def detect_anomalies_with_gemini(text_blob: str) -> dict:
    """
    Use Gemini to detect unusual transactions and potential fraud indicators.
    """
    try:
        model = GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
        **Task**: Analyze bank statement for unusual transactions and potential security concerns.

        **Input Text**:
        ---
        {text_blob}
        ---

        **Instructions**:
        1. Identify unusually large transactions
        2. Detect transactions at odd hours or locations
        3. Find duplicate or suspicious charges
        4. Flag potential fraud indicators
        5. Return analysis in JSON format only

        **Required JSON Format**:
        {{
          "anomalies_detected": {{
            "large_transactions": [
              {{"amount": "...", "merchant": "...", "date": "...", "reason": "..."}}
            ],
            "unusual_patterns": [
              "Pattern description 1",
              "Pattern description 2"
            ],
            "potential_fraud_indicators": [
              "Indicator 1",
              "Indicator 2"
            ],
            "risk_score": "Low/Medium/High",
            "recommendations": [
              "Recommendation 1",
              "Recommendation 2"
            ]
          }}
        }}
        """
        response = model.generate_content([Part.from_text(prompt)])
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response_text)
    except Exception as e:
        st.warning(f"Could not perform anomaly detection: {e}")
        return {}

def create_spending_visualizations(spending_data):
    """
    Create interactive visualizations for spending analysis.
    """
    charts = {}
    if "expense_categories" in spending_data:
        categories = spending_data["expense_categories"]
        if categories:
            clean_categories = {}
            for cat, amount in categories.items():
                try:
                    numeric_amount = float(re.sub(r'[^\d.-]', '', str(amount)))
                    if numeric_amount > 0:
                        clean_categories[cat] = numeric_amount
                except:
                    continue
            if clean_categories:
                fig_pie = px.pie(
                    values=list(clean_categories.values()),
                    names=list(clean_categories.keys()),
                    title="Spending by Category",
                    template="plotly_white"
                )
                fig_pie.update_traces(textinfo='percent+label')
                charts['category_pie'] = fig_pie
    
    if "top_merchants" in spending_data and spending_data["top_merchants"]:
        merchants_data = []
        for merchant in spending_data["top_merchants"]:
            try:
                amount = float(re.sub(r'[^\d.-]', '', str(merchant.get("amount", "0"))))
                merchants_data.append({
                    'Merchant': merchant.get("name", "Unknown"),
                    'Amount': amount
                })
            except:
                continue
        if merchants_data:
            df_merchants = pd.DataFrame(merchants_data)
            fig_bar = px.bar(
                df_merchants,
                x='Merchant',
                y='Amount',
                title="Top Merchants by Spending",
                template="plotly_white",
                color_discrete_sequence=['#0066cc']
            )
            fig_bar.update_xaxes(tickangle=45)
            charts['merchants_bar'] = fig_bar
    return charts

def generate_pdf_report(account_details, bank_details, spending_insights, anomaly_report):
    """
    Generate a comprehensive PDF report of the analysis.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue
        )
        story.append(Paragraph("Bank Statement Analysis Report", title_style))
        story.append(Spacer(1, 12))
        
        if account_details:
            story.append(Paragraph("Account Holder Details", styles['Heading2']))
            for key, value in account_details.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if bank_details:
            story.append(Paragraph("Bank Details", styles['Heading2']))
            for key, value in bank_details.items():
                story.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if spending_insights:
            story.append(Paragraph("Spending Analysis", styles['Heading2']))
            if "spending_insights" in spending_insights:
                for insight in spending_insights["spending_insights"]:
                    story.append(Paragraph(f"-  {insight}", styles['Normal']))
            story.append(Spacer(1, 12))
        
        if anomaly_report and "anomalies_detected" in anomaly_report:
            story.append(Paragraph("Security Analysis", styles['Heading2']))
            anomalies = anomaly_report["anomalies_detected"]
            if "risk_score" in anomalies:
                story.append(Paragraph(f"<b>Risk Score:</b> {anomalies['risk_score']}", styles['Normal']))
            if "recommendations" in anomalies:
                story.append(Paragraph("<b>Recommendations:</b>", styles['Normal']))
                for rec in anomalies["recommendations"]:
                    story.append(Paragraph(f"-  {rec}", styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
    except ImportError:
        st.warning("ReportLab not installed. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

def create_transaction_timeline(transaction_df):
    """
    Create an interactive timeline of transactions.
    """
    if transaction_df.empty:
        return None
    date_cols = [col for col in transaction_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    amount_cols = [col for col in transaction_df.columns if 'amount' in col.lower() or 'balance' in col.lower()]
    if not date_cols or not amount_cols:
        return None
    try:
        df_copy = transaction_df.copy()
        date_col = date_cols[0]
        amount_col = amount_cols[0]
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[date_col])
        if df_copy.empty:
            return None
        fig = px.scatter(
            df_copy,
            x=date_col,
            y=amount_col,
            title="Transaction Timeline",
            hover_data=df_copy.columns.tolist(),
            template="plotly_white"
        )
        return fig
    except Exception as e:
        st.warning(f"Could not create timeline: {e}")
        return None

def perform_compliance_check(text_blob: str) -> dict:
    """
    Check for compliance and regulatory patterns using Gemini.
    """
    try:
        model = GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
        **Task**: Analyze bank statement for compliance and regulatory concerns.

        **Input Text**:
        ---
        {text_blob}
        ---

        **Instructions**:
        1. Check for large cash transactions that might require reporting
        2. Identify international transfers
        3. Look for patterns that might indicate money laundering
        4. Check transaction frequency and amounts
        5. Return analysis in JSON format only

        **Required JSON Format**:
        {{
          "compliance_check": {{
            "large_cash_transactions": [
              {{"amount": "...", "date": "...", "type": "..."}}
            ],
            "international_transfers": [
              {{"amount": "...", "destination": "...", "date": "..."}}
            ],
            "suspicious_patterns": [
              "Pattern 1",
              "Pattern 2"
            ],
            "compliance_score": "Low/Medium/High Risk",
            "regulatory_notes": [
              "Note 1",
              "Note 2"
            ]
          }}
        }}
        """
        response = model.generate_content([Part.from_text(prompt)])
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned_response_text)
    except Exception as e:
        st.warning(f"Could not perform compliance check: {e}")
        return {}

# --- Streamlit User Interface ---
st.set_page_config(
    layout="wide",
    page_title="AI Bank Statement Analyzer Pro",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# Header Section
st.markdown("""
    <div class="section-card">
        <h1 style='text-align: center;'>🏦 AI-Powered Bank Statement Analyzer Pro</h1>
        <p style='text-align: center; color: #6b7280;'>
            Advanced analysis with spending insights, fraud detection, compliance checking, and interactive visualizations.
        </p>
        <p style='text-align: center; color: #dc2626; font-weight: 500;'>
            🔒 Please use PDFs with obfuscated sensitive information for privacy.
        </p>
    </div>
""", unsafe_allow_html=True)

# Check for Google Cloud libraries
if not IS_GCP_AVAILABLE:
    st.error("Google Cloud libraries are not installed. Please add `google-cloud-aiplatform` to your requirements.txt.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: #1f2937;'>⚙️ Analysis Settings</h2>", unsafe_allow_html=True)
    available_models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite"
    ]
    selected_model = st.selectbox(
        "Select AI Model",
        available_models,
        index=0,
        help="Gemini 2.5 Flash-Lite is the most cost-effective option."
    )
    
    st.markdown("<h2 style='color: #1f2937;'>🔍 Analysis Options</h2>", unsafe_allow_html=True)
    enable_spending_insights = st.checkbox("Spending Insights & Categorization", value=True)
    enable_fraud_detection = st.checkbox("Fraud & Anomaly Detection", value=True)
    enable_compliance_check = st.checkbox("Compliance & Regulatory Check", value=True)
    enable_visualizations = st.checkbox("Interactive Visualizations", value=True)
    enable_pdf_report = st.checkbox("Generate PDF Report", value=False)

# Main Content
with st.container():
    uploaded_file = st.file_uploader(
        "Upload your obfuscated bank statement PDF",
        type="pdf",
        help="Upload a PDF with sensitive information obscured"
    )

    if uploaded_file is not None:
        st.success(f"✅ File '{uploaded_file.name}' successfully uploaded.", icon="✅")
        
        if st.button("🚀 Analyze Statement", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.info("Step 1/6: Initializing secure connection to Google Cloud...")
            progress_bar.progress(10)
            if not initialize_vertex_ai():
                status_text.error("Initialization failed. Cannot proceed.")
                progress_bar.empty()
                st.stop()
            status_text.success("✅ Secure connection established.")
            progress_bar.progress(20)
            
            status_text.info("Step 2/6: Extracting data from PDF...")
            full_text, transaction_df = process_pdf(uploaded_file)
            if not full_text:
                status_text.error("Could not extract text from PDF. File might be corrupted or image-based.")
                progress_bar.empty()
                st.stop()
            status_text.success("✅ Text and tabular data extracted.")
            progress_bar.progress(40)
            
            status_text.info(f"Step 3/6: Extracting account details with {selected_model}...")
            try:
                model = GenerativeModel(selected_model)
                prompt = f"""
                **Task**: Analyze the text from a bank statement and extract key information.

                **Input Text**:
                ---
                {full_text}
                ---

                **Instructions**:
                1. Carefully read the text to identify the account holder's details and the bank's details.
                2. Return this information in a single, valid JSON object.
                3. **Crucially, your entire response must only be the JSON object itself.**
                4. If a specific piece of information (e.g., "E-mail") cannot be found, use "N/A".

                **Required JSON Format**:
                {{
                  "Account Holder Details": {{
                    "Name": "...",
                    "Address": "...",
                    "Contact Nr": "...",
                    "E-mail": "..."
                  }},
                  "Bank Details": {{
                    "Bank Name": "...",
                    "Bank Address": "...",
                    "Account Number": "..."
                  }}
                }}
                """
                response = model.generate_content([Part.from_text(prompt)])
                cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
                gemini_response = json.loads(cleaned_response_text)
            except Exception as e:
                status_text.error(f"Error with {selected_model}: {e}")
                gemini_response = get_info_from_gemini(full_text)
            status_text.success("✅ Account details extracted.")
            progress_bar.progress(60)
            
            spending_insights = {}
            anomaly_report = {}
            compliance_report = {}
            
            if enable_spending_insights:
                status_text.info("Step 4/6: Analyzing spending patterns...")
                transaction_data_str = transaction_df.to_string() if not transaction_df.empty else ""
                spending_insights = get_spending_insights_from_gemini(full_text, transaction_data_str)
                status_text.success("✅ Spending insights generated.")
                progress_bar.progress(70)
            
            if enable_fraud_detection:
                status_text.info("Step 5/6: Performing fraud detection...")
                anomaly_report = detect_anomalies_with_gemini(full_text)
                status_text.success("✅ Fraud detection completed.")
                progress_bar.progress(80)
            
            if enable_compliance_check:
                status_text.info("Step 6/6: Running compliance checks...")
                compliance_report = perform_compliance_check(full_text)
                status_text.success("✅ Compliance check completed.")
                progress_bar.progress(100)
            
            progress_bar.empty()
            status_text.empty()
            
            # Display Results
            st.markdown("<h2 style='color: #1f2937;'>📊 Comprehensive Analysis Results</h2>", unsafe_allow_html=True)
            st.divider()
            
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🏛️ Account Details",
                "💰 Spending Insights",
                "⚠️ Security Analysis",
                "📋 Compliance Check",
                "📈 Visualizations",
                "📄 Transaction Data"
            ])
            
            with tab1:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                if gemini_response:
                    col1, col2 = st.columns(2, gap="medium")
                    with col1:
                        st.subheader("Account Holder Details")
                        st.json(gemini_response.get("Account Holder Details", "Details not found by AI."))
                    with col2:
                        st.subheader("Bank Details")
                        st.json(gemini_response.get("Bank Details", "Details not found by AI."))
                else:
                    st.error("Could not retrieve structured details from the AI model.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab2:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                if spending_insights:
                    st.subheader("💸 Spending Summary")
                    if "spending_summary" in spending_insights:
                        summary = spending_insights["spending_summary"]
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Debits", summary.get("total_debits", "N/A"))
                        with col2:
                            st.metric("Total Credits", summary.get("total_credits", "N/A"))
                        with col3:
                            st.metric("Net Change", summary.get("net_balance_change", "N/A"))
                        with col4:
                            st.metric("Transactions", summary.get("transaction_count", "N/A"))
                    
                    st.subheader("📊 Expense Categories")
                    if "expense_categories" in spending_insights:
                        st.json(spending_insights["expense_categories"])
                    
                    st.subheader("🏪 Top Merchants")
                    if "top_merchants" in spending_insights and spending_insights["top_merchants"]:
                        merchants_df = pd.DataFrame(spending_insights["top_merchants"])
                        st.dataframe(merchants_df, use_container_width=True)
                    
                    st.subheader("💡 AI Insights")
                    if "spending_insights" in spending_insights:
                        for insight in spending_insights["spending_insights"]:
                            st.info(insight)
                else:
                    st.info("Spending insights analysis was not performed or failed.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab3:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                if anomaly_report and "anomalies_detected" in anomaly_report:
                    anomalies = anomaly_report["anomalies_detected"]
                    risk_score = anomalies.get("risk_score", "Unknown")
                    if risk_score.lower() == "high":
                        st.error(f"🚨 Risk Score: {risk_score}")
                    elif risk_score.lower() == "medium":
                        st.warning(f"⚠️ Risk Score: {risk_score}")
                    else:
                        st.success(f"✅ Risk Score: {risk_score}")
                    
                    if "large_transactions" in anomalies and anomalies["large_transactions"]:
                        st.subheader("💰 Large Transactions")
                        large_trans_df = pd.DataFrame(anomalies["large_transactions"])
                        st.dataframe(large_trans_df, use_container_width=True)
                    
                    if "unusual_patterns" in anomalies and anomalies["unusual_patterns"]:
                        st.subheader("🔍 Unusual Patterns Detected")
                        for pattern in anomalies["unusual_patterns"]:
                            st.warning(f"⚠️ {pattern}")
                    
                    if "potential_fraud_indicators" in anomalies and anomalies["potential_fraud_indicators"]:
                        st.subheader("🚨 Potential Fraud Indicators")
                        for indicator in anomalies["potential_fraud_indicators"]:
                            st.error(f"🚨 {indicator}")
                    
                    if "recommendations" in anomalies and anomalies["recommendations"]:
                        st.subheader("💡 Security Recommendations")
                        for rec in anomalies["recommendations"]:
                            st.info(f"✅ {rec}")
                else:
                    st.info("Security analysis was not performed or no anomalies detected.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab4:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                if compliance_report and "compliance_check" in compliance_report:
                    compliance = compliance_report["compliance_check"]
                    compliance_score = compliance.get("compliance_score", "Unknown")
                    if "high risk" in compliance_score.lower():
                        st.error(f"🚨 Compliance Risk: {compliance_score}")
                    elif "medium risk" in compliance_score.lower():
                        st.warning(f"⚠️ Compliance Risk: {compliance_score}")
                    else:
                        st.success(f"✅ Compliance Risk: {compliance_score}")
                    
                    if "large_cash_transactions" in compliance and compliance["large_cash_transactions"]:
                        st.subheader("💵 Large Cash Transactions")
                        cash_df = pd.DataFrame(compliance["large_cash_transactions"])
                        st.dataframe(cash_df, use_container_width=True)
                    
                    if "international_transfers" in compliance and compliance["international_transfers"]:
                        st.subheader("🌍 International Transfers")
                        intl_df = pd.DataFrame(compliance["international_transfers"])
                        st.dataframe(intl_df, use_container_width=True)
                    
                    if "suspicious_patterns" in compliance and compliance["suspicious_patterns"]:
                        st.subheader("🔍 Suspicious Patterns")
                        for pattern in compliance["suspicious_patterns"]:
                            st.warning(f"⚠️ {pattern}")
                    
                    if "regulatory_notes" in compliance and compliance["regulatory_notes"]:
                        st.subheader("📋 Regulatory Notes")
                        for note in compliance["regulatory_notes"]:
                            st.info(f"📋 {note}")
                else:
                    st.info("Compliance check was not performed or no issues found.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab5:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                if enable_visualizations and spending_insights:
                    st.subheader("📊 Interactive Visualizations")
                    charts = create_spending_visualizations(spending_insights)
                    if 'category_pie' in charts:
                        st.plotly_chart(charts['category_pie'], use_container_width=True)
                    if 'merchants_bar' in charts:
                        st.plotly_chart(charts['merchants_bar'], use_container_width=True)
                    timeline_chart = create_transaction_timeline(transaction_df)
                    if timeline_chart:
                        st.plotly_chart(timeline_chart, use_container_width=True)
                    if not charts and not timeline_chart:
                        st.info("No visualizations could be generated from the available data.")
                else:
                    st.info("Visualizations are disabled or no data available.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with tab6:
                st.markdown("<div class='section-card'>", unsafe_allow_html=True)
                st.subheader("📄 Transaction Data")
                if not transaction_df.empty:
                    st.subheader("🔍 Search & Filter Transactions")
                    search_term = st.text_input(
                        "Search transactions:",
                        placeholder="Enter merchant name, amount, or description..."
                    )
                    filtered_df = transaction_df
                    if search_term:
                        mask = transaction_df.astype(str).apply(
                            lambda x: x.str.contains(search_term, case=False, na=False)
                        ).any(axis=1)
                        filtered_df = transaction_df[mask]
                    
                    st.dataframe(filtered_df, use_container_width=True)
                    
                    col1, col2 = st.columns(2, gap="medium")
                    with col1:
                        csv_data = convert_df_to_csv(filtered_df)
                        st.download_button(
                            label="⬇️ Download Transactions as CSV",
                            data=csv_data,
                            file_name=f'transactions_{uploaded_file.name}.csv',
                            mime='text/csv',
                            use_container_width=True
                        )
                    with col2:
                        if enable_pdf_report:
                            account_details = gemini_response.get("Account Holder Details", {})
                            bank_details = gemini_response.get("Bank Details", {})
                            pdf_report = generate_pdf_report(
                                account_details, bank_details,
                                spending_insights, anomaly_report
                            )
                            if pdf_report:
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_report,
                                    file_name=f'bank_analysis_report_{uploaded_file.name}.pdf',
                                    mime='application/pdf',
                                    use_container_width=True
                                )
                else:
                    st.warning("No tabular transaction data detected. Full text was still analyzed.")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with st.expander("📊 Analysis Summary Statistics"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Text Characters Extracted", len(full_text))
                with col2:
                    st.metric("Tables Found", len(transaction_df) if not transaction_df.empty else 0)
                with col3:
                    analysis_count = sum([
                        bool(gemini_response),
                        bool(spending_insights),
                        bool(anomaly_report),
                        bool(compliance_report)
                    ])
                    st.metric("AI Analyses Performed", analysis_count)
                with col4:
                    st.metric("Model Used", selected_model.replace("gemini-", "Gemini ").title())
            
            with st.expander("📝 View Raw Extracted Text"):
                st.text_area("Raw Text Sent to AI", full_text, height=400)
    
    else:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.info("Please upload a PDF to begin the comprehensive analysis.", icon="ℹ️")
        st.subheader("🌟 Features Available")
        st.markdown("""
            - **🧠 AI-Powered Categorization**: Automatic expense categorization and spending insights
            - **🛡️ Fraud Detection**: Advanced anomaly detection and security analysis  
            - **📋 Compliance Checking**: Regulatory compliance and suspicious pattern detection
            - **📊 Interactive Visualizations**: Charts, graphs, and transaction timelines
            - **🔍 Smart Search & Filtering**: Advanced transaction search capabilities
            - **📄 Comprehensive PDF Reports**: Professional analysis reports for download
            - **⚡ Real-time Processing**: Fast, efficient analysis with multiple AI models
            - **🎯 Risk Assessment**: Automated risk scoring and recommendations
        """)
        st.markdown("</div>", unsafe_allow_html=True)

