# AI-Powered Bank Statement Analyzer

This project is a sophisticated web application built with Streamlit and powered by Google's Gemini AI. It transforms unstructured PDF bank statements into structured, actionable insights, complete with automated categorization, security analysis, and interactive visualizations.

This application was developed to meet all the core requirements of the project assessment, demonstrating proficiency in Pythonic data extraction, integration with cutting-edge AI services via GCP Vertex AI, and the development and deployment of a secure, user-friendly web dashboard.

---

## 🚀 Live Demo

You can access and interact with the live, deployed application here:

**[➡️ AI-Powered Bank Statement Analyzer](https://app-pdf-analyzer.streamlit.app/)**

---

## ✨ Key Features

-   **📄 PDF Data Extraction:** Upload any standard bank statement PDF and the system intelligently extracts both raw text and structured transaction tables.
-   **🧠 AI-Powered Structuring:** Leverages the **Google Gemini model** via **Vertex AI** to identify and parse key information like account holder details and bank information into a clean JSON format.
-   **📊 Automated Spending Insights:** Automatically categorizes every transaction (e.g., Food, Shopping, Bills) and calculates spending summaries.
-   **🛡️ Advanced Security & Compliance Analysis:** Utilizes AI to perform advanced checks, flagging anomalies, potential fraud indicators, and compliance concerns.
-   **📈 Interactive Visualizations:** Features a dynamic dashboard with Plotly charts to visualize spending by category, top merchants, and transaction timelines.
-   **📥 CSV Data Export:** Allows users to download the extracted transaction data as a CSV file for further analysis in tools like Excel or Google Sheets.
-   **☁️ Fully Deployed & Secure:** The application is deployed on Streamlit Community Cloud with all API keys and credentials managed securely via environment secrets.

---

## 🛠️ Technology Stack

-   **Frontend:** Streamlit
-   **Backend & Core Logic:** Python
-   **AI & Cloud Services:** Google Cloud Platform (GCP) - Vertex AI (Gemini 2.5 Flash Model)
-   **PDF Parsing:** PyMuPDF (`fitz`)
-   **Data Manipulation:** Pandas
-   **Data Visualization:** Plotly
-   **Deployment:** Streamlit Community Cloud

---

## ⚙️ System Architecture

The application follows a simple yet powerful data flow:

1.  **User Interface (Streamlit):** The user uploads a PDF file via the web dashboard.
2.  **Backend Processing (Python):** The Streamlit backend receives the file. The `PyMuPDF` library extracts raw text and tabular data.
3.  **AI Service Integration (Vertex AI):** The extracted text is sent via secure API calls to the Gemini model hosted on Google's Vertex AI.
4.  **AI Analysis:** Gemini processes the data based on engineered prompts to perform structuring, categorization, and analysis.
5.  **Data Presentation:** The structured JSON and analysis from the AI are sent back to the Streamlit app, which then renders the results, tables, and interactive visualizations.

---

## 🔧 Setup and Local Installation

To run this project on your local machine, please follow these steps:

**1. Clone the Repository**
```bash
git clone https://github.com/Sailikith22001/streamlit-pdf-analyzer.git
cd streamlit-pdf-analyzer
