import streamlit as st
import pandas as pd
import joblib
import openai
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

# Example initialization
data = pd.DataFrame()  # Creates an empty DataFrame

# Configure OpenAI API
load_dotenv()  # Loads variables from .env
openai.api_key = os.getenv("OPENAI_API_KEY")
try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Test message"}]
    )
    print("GPT-4 is accessible:", response)
except openai.error.OpenAIError as e:
    print("Error accessing GPT-4:", e)

# # Custom CSS Styling

##
# Model Loading with Error Handling
try:
    model = joblib.load("best_fraud_detection_model.pkl")
except Exception as e:
    model = None
    st.error(f"Error loading model: {str(e)}")

# Session State Initialization
session_state_defaults = {
    'sop_steps': [],
    'connected_tools': {},
    'audit_logs': [],
    'chat_history': [],
    'current_page': "Dashboard"
}

for key, value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

def fraud_analysis_chatbot(question, context=""):
    """Generates AI-powered fraud analysis responses"""
    prompt = f"""You are a senior financial fraud analyst AI assistant.
    Context: {context}
    Question: {question}
    
    Provide:
    1. Clear technical explanation
    2. Step-by-step investigation reasoning
    3. Recommendations for next actions
    4. Potential false positive indicators
    5. Regulatory compliance considerations
    6. Can you provide a detailed analysis of the recent fraudulent transactions?
    7. What patterns or anomalies indicate potential fraud in the dataset?
    8. How does the current fraud detection performance compare to historical data?
    9. What are the common characteristics of false positives in our fraud detection system?

    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Navigation Sidebar
with st.sidebar:
    # col1, col2 = st.columns([1, 3])
    # with col1:
    #     st.image(r"C:\Users\junior.turatsinze_on\Desktop\Hackthon_Irembo\venv\images\fixer.brender logo.png",
    #              width=1000)
    # with col2:
    #     st.markdown("### Navigation")

    pages = {
        "Dashboard": "🏠",
        "SOP Builder": "📝",
        "Tools": "🛠️",
        "Audit Logs": "📜"
    }
    
    for page_name, icon in pages.items():
        btn_class = "nav-button" + (" active-page" if st.session_state.current_page == page_name else "")
        if st.button(
            f"{icon} {page_name}",
            key=f"nav_{page_name}",
            help=f"Go to {page_name}",
            use_container_width=True
        ):
            st.session_state.current_page = page_name

# Main Content
st.title("Fix.AI- AI Agent That investigates financial fraud") 


# Page Routing
if st.session_state.current_page == "SOP Builder":
    st.header("📝 SOP Builder")
    
    # SOP Import Section
    with st.expander("Import Existing SOP"):
        uploaded_sop = st.file_uploader("Upload SOP Document", 
                                       type=["txt", "csv"],
                                       help="Supported formats: TXT (one step per line) or CSV (columns: step,tool)")
        if uploaded_sop:
            try:
                if uploaded_sop.name.endswith('.csv'):
                    sop_df = pd.read_csv(uploaded_sop)
                    for _, row in sop_df.iterrows():
                        new_step = {
                            "step": len(st.session_state.sop_steps) + 1,
                            "description": row['step'],
                            "tool": row['tool'],
                            "status": "Pending"
                        }
                        st.session_state.sop_steps.append(new_step)
                else:
                    lines = uploaded_sop.getvalue().decode().split('\n')
                    for line in lines:
                        if line.strip():
                            parts = line.split('|')
                            new_step = {
                                "step": len(st.session_state.sop_steps) + 1,
                                "description": parts[0].strip(),
                                "tool": parts[1].strip() if len(parts) > 1 else "General",
                                "status": "Pending"
                            }
                            st.session_state.sop_steps.append(new_step)
                st.success("SOP imported successfully!")
            except Exception as e:
                st.error(f"Error importing SOP: {str(e)}")

    # SOP Management
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        with st.expander("Create New SOP"):
            sop_name = st.text_input("SOP Name")
            step_description = st.text_input("Step Description")
            tool_used = st.selectbox("Tool Used", ["Fraud Ops Dashboard", "Postgres", "Jira", "Gmail"])
            
            if st.button("Add Step"):
                new_step = {
                    "step": len(st.session_state.sop_steps) + 1,
                    "description": step_description,
                    "tool": tool_used,
                    "status": "Pending"
                }
                st.session_state.sop_steps.append(new_step)
    
    with col2:
        st.markdown("### SOP Actions")
        if st.button("Clear All Steps"):
            st.session_state.sop_steps = []
        if st.button("Export SOP"):
            sop_df = pd.DataFrame(st.session_state.sop_steps)
            st.download_button(
                label="Download SOP",
                data=sop_df.to_csv(index=False).encode('utf-8'),
                file_name="sop_template.csv",
                mime="text/csv"
            )

    # Display SOP
    st.subheader("Current SOP")
    if st.session_state.sop_steps:
        for step in st.session_state.sop_steps:
            with st.container():
                cols = st.columns([0.6, 0.2, 0.2])
                cols[0].markdown(f"*Step {step['step']}*: {step['description']}")
                cols[1].markdown(f"{step['tool']}")
                cols[2].markdown(f"{step['status']}")
    else:
        st.info("No SOP steps defined yet.")

elif st.session_state.current_page == "Tools":
    st.header("🛠️ Tools Integration")
    
    with st.expander("Connect a Tool"):
        tool_name = st.selectbox("Select Tool", ["Fraud Ops Dashboard", "Postgres", "Jira", "Gmail"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Connect"):
            st.session_state.connected_tools[tool_name] = {
                "status": "Connected",
                "last_used": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.success(f"Successfully connected to {tool_name}!")

    st.subheader("Connected Tools")
    for tool, details in st.session_state.connected_tools.items():
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown(f"*{tool}*")
            st.caption(f"Last used: {details['last_used']}")
        with col2:
            st.markdown(f"{details['status']}")

elif st.session_state.current_page == "Audit Logs":
    st.header("📜 Audit Logs")
    
    if st.session_state.audit_logs:
        for log in st.session_state.audit_logs:
            with st.expander(f"{log['timestamp']} - {log['action']}"):
                st.markdown(f"*Tool Used*: {log['tool']}")
                st.markdown(f"*Duration*: {log['duration']}s")
                st.markdown("*Details*:")
                st.json(log)
    else:
        st.info("No audit logs available yet.")

else:  # Dashboard
    st.header("🔍 Fraud Investigation Dashboard")
    
    uploaded_file = st.file_uploader("Upload Transaction Dataset", type="csv")
    
    if uploaded_file and model:
        try:
            data = pd.read_csv(uploaded_file)
            required_features = [
                'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
            ]
            
            missing_features = [f for f in required_features if f not in data.columns]
            
            if not missing_features:
                if 'Amount' in data.columns:
                    data['Amount'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()
                
                st.subheader("Dataset Preview")
                st.write(data.head())
                
                if st.button("Run Full Investigation"):
                    start_time = time.time()
                    
                    # SOP Execution Simulation
                    for step in st.session_state.sop_steps:
                        log_entry = {
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "action": step['description'],
                            "tool": step['tool'],
                            "duration": round(time.time() - start_time, 2),
                            "status": "Completed"
                        }
                        st.session_state.audit_logs.append(log_entry)
                    
                    # Model Prediction
                    predictions = model.predict(data[required_features])
                    prediction_proba = model.predict_proba(data[required_features])[:, 1]
                    
                    if len(predictions)>0:
                        st.session_state.predictions = predictions
                        st.session_state.prediction_proba = prediction_proba
                    else:
                        predictions = st.session_state.predictions
                        prediction_proba = st.session_state.prediction_proba
                    
                    data['Prediction'] = predictions
                    data['Fraud Probability'] = prediction_proba
                    
                    
                    st.subheader("Fraud Detection Results")
                    st.write(data)
                    st.success(f"Investigation complete! {sum(predictions)} fraudulent transactions detected.")
                    
                    # Download Results
                    st.download_button(
                        label="Download Results",
                        data=data.to_csv(index=False).encode('utf-8'),
                        file_name="fraud_detection_results.csv",
                        mime="text/csv"
                    )
            else:
                st.error(f"Missing required features: {missing_features}")
                st.write("Required columns:")
                st.write(required_features)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    if "Predictions" not in data.columns:
        if "predictions" in st.session_state and "prediction_proba" in st.session_state:
            data['Prediction'] = st.session_state.predictions
            data['Fraud Probability'] = st.session_state.prediction_proba
    
    # Single Transaction Analysis
    st.header("Single Transaction Analysis")
    input_fields = {}
    cols = st.columns(4)
    features = ['Time', 'Amount'] + [f'V{i}' for i in range(1,29)]
    
    for i, feat in enumerate(features):
        with cols[i%4]:
            input_fields[feat] = st.number_input(feat, value=0.0)

    if st.button("Analyze Single Transaction") and model:
        try:
            input_data = pd.DataFrame([input_fields])
            prediction = model.predict(input_data)
            proba = model.predict_proba(input_data)[:,1]
            
            st.subheader("Analysis Result")
            if prediction[0] == 1:
                st.error("Fraudulent Transaction Detected!")
            else:
                st.success("Transaction is Legitimate.")
            
            st.write(f"Fraud Probability: {proba[0]:.2f}")
            
            # Audit Log
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "action": "Single Transaction Analysis",
                "tool": "Fraud Detection Model",
                "duration": 0.5,
                "status": "Completed"
            }
            st.session_state.audit_logs.append(log_entry)
        
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Chat Assistant

with st.container():
    st.markdown("---")
    st.header("🧠 FixAI Assistant")
    
    current_context = ""
    if st.session_state.current_page == "Dashboard" and 'data' in locals():
        # Calculate values separately
        total_transactions = len(data)
        fraudulent_transactions = sum(data['Prediction']) if 'Prediction' in data else 0
        if 'Fraud Probability' in data:
             average_fraud_probability = f"{data[data['Prediction']>0]['Fraud Probability'].mean()*100:.6f}"
            #average_fraud_probability = f"{data['Fraud Probability'].mean():.6f}"
        else:
            average_fraud_probability = "0"

        # Construct the current_context string
        current_context = f"""
        Current Investigation Context:
        - Total transactions analyzed: {total_transactions}
        - Fraudulent transactions detected: {fraudulent_transactions}
        - Average fraud probability: {average_fraud_probability}
        """

        # Display the current_context
        st.write(current_context)


    with st.form("chat_form"):
        user_question = st.text_input("Ask about fraud patterns, results, or next steps:")
        submitted = st.form_submit_button("Ask FixAI")
        
        if submitted and user_question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            with st.spinner("Analyzing..."):
                ai_response = fraud_analysis_chatbot(user_question, current_context)
            
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

    # Chat History
    for message in st.session_state.chat_history[-5:]:
        timestamp = message['timestamp']
        if message['role'] == "user":
            st.markdown(f"""
            <div class="user-message">
                <small>{timestamp}</small>
                <p>👤 User: {message["content"]}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="assistant-message">
                <small>{timestamp}</small>
                <p>🤖 FixAI: {message["content"]}</p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Fix.AI- AI Agent That investigates financial fraud")

# To run: streamlit run app.py


