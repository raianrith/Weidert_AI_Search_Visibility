import streamlit as st
from openai import OpenAI
import google.generativeai as genai
import re
import pandas as pd
import time
import os
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from concurrent.futures import ThreadPoolExecutor
import json
import tempfile
from datetime import datetime, date
import numpy as np

# Download required NLTK data
try:
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {e}")

sia = SentimentIntensityAnalyzer()

# ─── PAGE CONFIG & GLOBAL CSS ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Weidert Group LLM Search Visibility Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with Weidert branding
st.markdown("""
<style>
/* Center the tabs */
div[data-baseweb="tab-list"] {
    display: flex !important;
    justify-content: center !important;
}

div[data-baseweb="tab-list"] button[role="tab"] {
    background-color: #fff !important;
    color: #000 !important;
    border: 1px solid transparent;
    border-radius: 4px 4px 0 0;
    padding: 0.5rem 1rem;
    margin: 0;
    position: relative;
}

div[data-baseweb="tab-list"] button[role="tab"]:not(:last-child)::after {
    content: "|";
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    color: #000;
}

div[data-baseweb="tab-list"] button[role="tab"]:hover {
    background-color: #e64626 !important;
    color: #fff !important;
}

div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {
    border-color: #888 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    background-color: #fff !important;
    color: #000 !important;
}

/* Button centering */
div.stButton > button {
    margin: 0 auto;
    display: block;
}

/* Executive dashboard styling */
.metric-card {
    background: linear-gradient(135deg, #e64626 0%, #c93a1f 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
}

/* Template card styling */
.template-card {
    border: 1px solid #e1e5e9;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}

.template-title {
    font-weight: bold;
    color: #2c3e50;
    margin-bottom: 0.5rem;
}

.position-indicator {
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 15px;
    font-size: 0.8rem;
    font-weight: bold;
}

.position-first { background: #28a745; color: white; }
.position-middle { background: #ffc107; color: black; }
.position-last { background: #dc3545; color: white; }
.position-none { background: #6c757d; color: white; }
</style>
""", unsafe_allow_html=True)

# ─── PREDEFINED QUERIES FOR WEIDERT GROUP ───────────────────────────────────
PREDEFINED_QUERIES = [
    "Best B2B marketing agencies for industrial companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "HubSpot implementation partners for manufacturing firms -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Inbound marketing agency for engineering companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "B2B marketing agency specializing in complex industrial sales -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Marketing agencies for environmental services companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "HubSpot partner agencies for logistics companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Sales and marketing alignment consulting for B2B industrial -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Industrial marketing agency with HubSpot expertise -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "B2B content marketing for technical products and services -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Marketing automation for complex B2B sales cycles -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Lead generation strategies for industrial B2B companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Revenue operations consulting for mid-market manufacturers -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "HubSpot CRM implementation for engineering firms -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "B2B marketing strategy for $50M-$300M industrial companies -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Inbound marketing ROI for technical B2B services -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Marketing agencies for privately held industrial businesses -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "How do Weidert Group and Stream Creative compare for B2B industrial marketing services? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Which HubSpot partner is better for manufacturing companies: Weidert Group or New Breed Revenue? -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Compare marketing agencies for engineering firms: Weidert vs SmartBug Media -- Provide sources where you are extracting information from in this format - 'https?://\\S+'",
    "Best alternative to Gorilla 76 for industrial B2B marketing services -- Provide sources where you are extracting information from in this format - 'https?://\\S+'"
]

# ─── QUERY TEMPLATES ─────────────────────────────────────────────────────────
QUERY_TEMPLATES = {
    "Agency Discovery": [
        "B2B marketing agencies for industrial companies",
        "HubSpot partner agencies in the US",
        "Inbound marketing consultants for manufacturing",
        "Marketing automation agencies for B2B",
        "Revenue operations consulting firms"
    ],
    "Industry Specific": [
        "Marketing agency for engineering companies",
        "B2B marketing for environmental services",
        "Logistics company marketing consultants",
        "Industrial manufacturing marketing experts",
        "Technical B2B marketing specialists"
    ],
    "Service Seeking": [
        "HubSpot implementation and strategy",
        "Sales and marketing alignment consulting",
        "B2B lead generation services",
        "Content marketing for technical products",
        "Marketing ROI measurement and analytics"
    ],
    "Comparison Queries": [
        "Compare B2B marketing agencies",
        "Best HubSpot partners for mid-market companies",
        "Weidert Group vs other marketing agencies",
        "Top inbound marketing agencies",
        "Marketing agency selection criteria"
    ],
    "Problem Solution": [
        "How to align sales and marketing teams",
        "Improve B2B lead quality and conversion",
        "Marketing strategy for long sales cycles",
        "Generate more qualified leads for industrial products",
        "Measure marketing ROI for B2B services"
    ]
}

# ─── COMPETITOR ANALYSIS SETUP ─────────────────────────────────────────────
COMPETITORS = [
    "Stream Creative", "New Breed Revenue", "New Breed", "Gorilla 76", 
    "SmartBug Media", "SmartBug", "IMPACT", "Kuno Creative", 
    "Impulse Creative", "Square 2 Marketing", "Revenue River"
]

# ─── LOGO & HEADER ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:1rem 0;'>
  <h1>🔍 Weidert Group LLM Search Visibility Tool</h1>
  <h4 style='color:#666;'>AI-Powered Competitive Intelligence for B2B Industrial Marketing</h4>
  <p style='color:#999; font-size:0.9rem;'>Analyze brand visibility across ChatGPT, Gemini & Perplexity</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR: CONFIGURATION ─────────────────────────────────────────────────
st.sidebar.title("🛠️ Model Configuration")

openai_model = st.sidebar.selectbox(
    "OpenAI model", 
    ["gpt-4o", "gpt-4", "gpt-3.5-turbo"], 
    index=0
)
gemini_model_name = st.sidebar.selectbox(
    "Gemini model", 
    ["gemini-2.0-flash-exp", "gemini-1.5-pro"], 
    index=0
)
perplexity_model_name = st.sidebar.selectbox(
    "Perplexity model", 
    ["sonar", "sonar-pro"], 
    index=0
)

st.sidebar.divider()

st.sidebar.subheader("⚙️ Advanced Settings")
max_workers = st.sidebar.slider("Parallel Processing Workers", 3, 12, 6)
delay_between_requests = st.sidebar.slider("Delay Between Requests (seconds)", 0.0, 2.0, 0.1)

# ─── API CLIENTS ────────────────────────────────────────────────────────────────
openai_key = st.secrets.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
gemini_key = st.secrets.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
perp_key = st.secrets.get("perplexity_api_key") or os.getenv("PERPLEXITY_API_KEY")

openai_client = OpenAI(api_key=openai_key) if openai_key else None
if gemini_key:
    genai.configure(api_key=gemini_key)
    gemini_model = genai.GenerativeModel(gemini_model_name)
else:
    gemini_model = None
perplexity_client = OpenAI(api_key=perp_key, base_url="https://api.perplexity.ai") if perp_key else None

SYSTEM_PROMPT = "Provide a helpful answer to the user's query."

# ─── API FUNCTIONS ─────────────────────────────────────────────────────────
def get_openai_response(q):
    try:
        if not openai_client:
            return "ERROR: OpenAI API key not configured"
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = openai_client.chat.completions.create(
            model=openai_model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_gemini_response(q):
    try:
        if not gemini_model:
            return "ERROR: Gemini API key not configured"
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = gemini_model.generate_content(q)
        return r.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

def get_perplexity_response(q):
    try:
        if not perplexity_client:
            return "ERROR: Perplexity API key not configured"
        if delay_between_requests > 0:
            time.sleep(delay_between_requests)
        r = perplexity_client.chat.completions.create(
            model=perplexity_model_name,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"

# ─── ANALYSIS FUNCTIONS ─────────────────────────────────────────────────────
def safe_sentence_tokenize(text):
    """Safe sentence tokenization with fallback"""
    try:
        return nltk.sent_tokenize(str(text))
    except:
        sentences = re.split(r'[.!?]+', str(text))
        return [s.strip() for s in sentences if s.strip()]

def analyze_position(text, brand="Weidert"):
    """Analyze where in the response the brand appears"""
    if not text or pd.isna(text) or text.startswith("ERROR"):
        return "Not Mentioned", 0, "N/A"
    
    text_str = str(text)
    sentences = safe_sentence_tokenize(text_str)
    total_sentences = len(sentences)
    
    if total_sentences == 0:
        return "Not Mentioned", 0, "N/A"
    
    # Check for both "Weidert" and "Weidert Group"
    brand_patterns = [brand.lower(), "weidert group"]
    
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        if any(pattern in sentence_lower for pattern in brand_patterns):
            position_pct = (i + 1) / total_sentences
            if position_pct <= 0.33:
                return "First Third", i + 1, f"{position_pct:.1%}"
            elif position_pct <= 0.66:
                return "Middle Third", i + 1, f"{position_pct:.1%}"
            else:
                return "Last Third", i + 1, f"{position_pct:.1%}"
    
    return "Not Mentioned", 0, "N/A"

def analyze_context(text, brand="Weidert"):
    """Analyze the context around brand mentions"""
    if not text or pd.isna(text) or text.startswith("ERROR"):
        return "Not Mentioned", 0, []
    
    text_str = str(text)
    brand_patterns = [brand.lower(), "weidert group"]
    
    if not any(pattern in text_str.lower() for pattern in brand_patterns):
        return "Not Mentioned", 0, []
    
    sentences = safe_sentence_tokenize(text_str)
    contexts = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(pattern in sentence_lower for pattern in brand_patterns):
            sentiment = sia.polarity_scores(sentence)
            if sentiment['compound'] >= 0.1:
                context_type = "Positive"
            elif sentiment['compound'] <= -0.1:
                context_type = "Negative"
            else:
                context_type = "Neutral"
            
            contexts.append({
                'sentence': sentence,
                'sentiment': sentiment['compound'],
                'context': context_type
            })
    
    if contexts:
        avg_sentiment = np.mean([c['sentiment'] for c in contexts])
        return contexts[0]['context'], avg_sentiment, contexts
    
    return "Neutral", 0, []

def extract_competitors_detailed(text):
    """Enhanced competitor extraction with position tracking"""
    if not text or pd.isna(text) or text.startswith("ERROR"):
        return [], {}
    
    text_str = str(text)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(c) for c in COMPETITORS) + r')\b', flags=re.IGNORECASE)
    matches = pattern.finditer(text_str)
    
    found_competitors = []
    positions = {}
    
    sentences = safe_sentence_tokenize(text_str)
    
    for match in matches:
        competitor = match.group(1)
        for comp in COMPETITORS:
            if competitor.lower() == comp.lower():
                competitor = comp
                break
        
        if competitor not in found_competitors:
            found_competitors.append(competitor)
            
            for i, sentence in enumerate(sentences):
                if competitor.lower() in sentence.lower():
                    positions[competitor] = i + 1
                    break
    
    return found_competitors, positions

# ─── PARALLEL PROCESSING ─────────────────────────────────────────────────────
def get_response_with_source(source_func_tuple):
    """Helper function with error handling and timing"""
    source, func, query = source_func_tuple
    start_time = time.time()
    try:
        response = func(query)
        end_time = time.time()
        return {
            "Query": query, 
            "Source": source, 
            "Response": response,
            "Response_Time": round(end_time - start_time, 2),
            "Timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        end_time = time.time()
        return {
            "Query": query, 
            "Source": source, 
            "Response": f"ERROR: {e}",
            "Response_Time": round(end_time - start_time, 2),
            "Timestamp": datetime.now().isoformat()
        }

def process_queries_parallel(queries):
    """Parallel processing of queries"""
    all_tasks = []
    
    for q in queries:
        all_tasks.extend([
            ("OpenAI", get_openai_response, q),
            ("Gemini", get_gemini_response, q),
            ("Perplexity", get_perplexity_response, q)
        ])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(get_response_with_source, all_tasks))
    
    return results

# ─── MAIN APPLICATION TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Multi-LLM Response Generator", 
    "Search Visibility Analysis", 
    "Competitor Comparison", 
    "Gap Analysis & Opportunities"
])

# ─── TAB 1: MULTI-LLM GENERATOR ─────────────────────────────────────────────
with tab1:
    st.markdown(
        '<h5 style="text-align:center; margin-bottom:1rem; color:#a9a9a9">'
        'Generate and analyze responses from OpenAI, Gemini, & Perplexity'
        '</h5>',
        unsafe_allow_html=True
    )
    
    # Predefined Query Set Section
    with st.expander("🎯 Predefined Query Set (20 Queries)", expanded=False):
        st.markdown("**Run the complete set of 20 predefined queries with one click:**")
        st.caption("Comprehensive query set covering B2B industrial marketing topics and competitive comparisons")
        
        for i, query in enumerate(PREDEFINED_QUERIES, 1):
            display_query = query.split("--")[0].strip()
            st.text(f"{i}. {display_query}")
        
        if st.button("🚀 Run All 20 Predefined Queries", key="run_predefined", type="primary"):
            st.session_state.use_predefined = True
            st.session_state.run_triggered = True
    
    # Query Templates Section
    with st.expander("📋 Query Templates", expanded=False):
        st.markdown("**Select from pre-built query templates:**")
        
        selected_category = st.selectbox("Choose Category:", list(QUERY_TEMPLATES.keys()))
        
        cols = st.columns(2)
        for i, template in enumerate(QUERY_TEMPLATES[selected_category]):
            col = cols[i % 2]
            with col:
                st.markdown(f"""
                <div class="template-card">
                    <div class="template-title">Query {i+1}</div>
                    <div>{template}</div>
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Use Template {i+1}", key=f"template_{selected_category}_{i}"):
                    st.session_state.template_query = template
    
    # Main query input
    initial_value = st.session_state.get('template_query', '')
    queries_input = st.text_area(
        "Custom Queries (one per line)",
        value=initial_value,
        height=200,
        placeholder="e.g. Best B2B marketing agencies for industrial companies\nHubSpot implementation partners for manufacturing"
    )
    
    # Run button
    if st.button("🔍 Run Custom Queries", key="run_analysis", type="primary"):
        st.session_state.use_predefined = False
        st.session_state.run_triggered = True

    # Process queries
    if st.session_state.get('run_triggered', False):
        if st.session_state.get('use_predefined', False):
            qs = PREDEFINED_QUERIES
        else:
            qs = [q.strip() for q in queries_input.splitlines() if q.strip()]
        
        if not qs:
            st.warning("Please enter at least one query or use predefined queries.")
        else:
            with st.spinner(f"Gathering responses for {len(qs)} queries..."):
                start_time = time.time()
                results = process_queries_parallel(qs)
                end_time = time.time()
                
                st.success(f"✅ Completed {len(results)} API calls in {end_time - start_time:.1f} seconds!")

            # Process results
            df = pd.DataFrame(results)
            df['Date'] = datetime.now().date()
            
            # Add analytics
            position_analysis = df['Response'].apply(lambda x: analyze_position(x, "Weidert"))
            df['Weidert_Position'] = [p[0] for p in position_analysis]
            df['Weidert_Sentence_Num'] = [p[1] for p in position_analysis]
            df['Weidert_Position_Pct'] = [p[2] for p in position_analysis]
            
            context_analysis = df['Response'].apply(lambda x: analyze_context(x, "Weidert"))
            df['Context_Type'] = [c[0] for c in context_analysis]
            df['Context_Sentiment'] = [c[1] for c in context_analysis]
            
            competitor_analysis = df['Response'].apply(extract_competitors_detailed)
            df['Competitors_Found'] = [', '.join(c[0]) if c[0] else '' for c in competitor_analysis]
            
            # Additional columns
            df['Branded_Query'] = df['Query'].str.contains('weidert', case=False, na=False)
            df['Weidert_Mentioned'] = df['Response'].apply(
                lambda x: 'weidert' in str(x).lower() and not str(x).startswith("ERROR")
            )
            df['Sources_Cited'] = df['Response'].apply(
                lambda x: ', '.join(re.findall(r'https?://\S+', str(x))) if not str(x).startswith("ERROR") else ''
            )
            df['Weidert_URL_Cited'] = df['Sources_Cited'].str.contains('weidert.com', case=False, na=False)
            
            # Store in session state
            st.session_state.latest_results = df
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mention_rate = (df['Weidert_Mentioned'].sum() / len(df) * 100)
                st.metric("Weidert Mention Rate", f"{mention_rate:.1f}%")
            
            with col2:
                avg_response_time = df['Response_Time'].mean()
                st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
            
            with col3:
                first_position_rate = (df['Weidert_Position'] == 'First Third').sum() / len(df) * 100
                st.metric("First Third Mentions", f"{first_position_rate:.1f}%")
            
            with col4:
                positive_rate = (df['Context_Type'] == 'Positive').sum() / len(df) * 100
                st.metric("Positive Context", f"{positive_rate:.1f}%")
            
            # Detailed table
            st.subheader("📋 Detailed Results")
            display_cols = ['Query', 'Source', 'Response', 'Response_Time', 'Weidert_Position', 'Context_Type']
            st.dataframe(df[display_cols], use_container_width=True, height=400)
            
            # Download
            st.download_button(
                "📥 Download Results (CSV)",
                df.to_csv(index=False),
                "weidert_llm_results.csv",
                "text/csv"
            )
            
        st.session_state.run_triggered = False

# ─── TAB 2: VISIBILITY ANALYSIS ─────────────────────────────────────────────
with tab2:
    st.markdown("### 🔍 Search Visibility Analysis")
    
    uploaded = st.file_uploader("Upload results CSV", type="csv", key="visibility_upload")
    
    df_main = None
    
    if 'latest_results' in st.session_state:
        use_latest = st.checkbox("Use results from Multi-LLM Response Generator", value=True)
        if use_latest:
            df_main = st.session_state.latest_results.copy()
    elif uploaded:
        df_main = pd.read_csv(uploaded)
    
    if df_main is not None:
        # Ensure all necessary columns exist
        if 'Weidert_Position' not in df_main.columns:
            position_analysis = df_main['Response'].apply(lambda x: analyze_position(x, "Weidert"))
            df_main['Weidert_Position'] = [p[0] for p in position_analysis]
        
        if 'Context_Type' not in df_main.columns:
            context_analysis = df_main['Response'].apply(lambda x: analyze_context(x, "Weidert"))
            df_main['Context_Type'] = [c[0] for c in context_analysis]
        
        if 'Competitors_Found' not in df_main.columns:
            competitor_analysis = df_main['Response'].apply(extract_competitors_detailed)
            df_main['Competitors_Found'] = [', '.join(c[0]) if c[0] else '' for c in competitor_analysis]
        
        if 'Weidert_Mentioned' not in df_main.columns:
            df_main['Weidert_Mentioned'] = df_main['Response'].apply(
                lambda x: 'weidert' in str(x).lower() and not str(x).startswith("ERROR")
            )
        
        if 'Branded_Query' not in df_main.columns:
            df_main['Branded_Query'] = df_main['Query'].str.contains('weidert', case=False, na=False)
        
        # Traditional Mention Rates
        st.subheader("📊 Mention Rates by Source")
        overall_rate = df_main.groupby('Source')['Weidert_Mentioned'].apply(
            lambda x: (x.sum() / len(x) * 100)
        ).round(1)
        
        cols = st.columns(len(overall_rate))
        for col, src in zip(cols, overall_rate.index):
            col.metric(f"{src}", f"{overall_rate[src]}%")
        
        st.divider()
        
        # Branded vs Non-Branded
        st.subheader("🎯 Branded vs. Non-Branded Query Performance")
        
        branded_analysis = df_main.groupby(['Source', 'Branded_Query'])['Weidert_Mentioned'].apply(
            lambda x: (x.sum() / len(x) * 100)
        ).unstack(fill_value=0).round(1)
        
        if len(branded_analysis.columns) == 2:
            branded_analysis.columns = ['Non-Branded', 'Branded']
        
        fig = px.bar(branded_analysis.T, title="Weidert Mention Rate: Branded vs Non-Branded",
                    labels={'value': 'Mention Rate (%)'}, barmode='group')
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Position Analysis
        st.subheader("📍 Position Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            position_counts = df_main['Weidert_Position'].value_counts()
            fig = px.pie(values=position_counts.values, names=position_counts.index,
                        title="Weidert Mention Position Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            position_by_source = df_main.groupby(['Source', 'Weidert_Position']).size().unstack(fill_value=0)
            fig = px.bar(position_by_source, title="Position by Source", barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        
        # Context Analysis
        st.subheader("💭 Context Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            context_counts = df_main['Context_Type'].value_counts()
            color_map = {'Positive': 'green', 'Neutral': 'blue', 'Negative': 'red', 'Not Mentioned': 'gray'}
            fig = px.bar(x=context_counts.index, y=context_counts.values,
                        title="Context Type Distribution",
                        color=context_counts.index,
                        color_discrete_map=color_map)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            context_by_source = df_main.groupby(['Source', 'Context_Type']).size().unstack(fill_value=0)
            fig = px.bar(context_by_source, title="Context by Source", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # NEW SECTION: Simple list of queries without Weidert mentions
        st.subheader("⚠️ Queries Without Weidert Mentions")
        st.caption("Quick reference list of queries where Weidert was not mentioned")
        
        # Filter for responses where Weidert was NOT mentioned
        negative_results = df_main[~df_main['Weidert_Mentioned']].copy()
        
        if len(negative_results) > 0:
            # Simple summary metric
            negative_count = len(negative_results)
            negative_pct = (negative_count / len(df_main) * 100)
            unique_queries_missing = negative_results['Query'].nunique()
            
            st.warning(f"⚠️ **{negative_count} responses** ({negative_pct:.1f}%) across **{unique_queries_missing} unique queries** did not mention Weidert Group")
            
            # Simple table view of queries and their LLM sources
            st.markdown("### 📋 List of Affected Queries")
            
            # Group by query and show which LLMs didn't mention Weidert
            query_summary = []
            for query in negative_results['Query'].unique():
                query_negs = negative_results[negative_results['Query'] == query]
                sources_missing = ', '.join(query_negs['Source'].tolist())
                
                # Check if any LLM mentioned Weidert for this query
                all_query_responses = df_main[df_main['Query'] == query]
                any_mentioned = all_query_responses['Weidert_Mentioned'].any()
                
                query_summary.append({
                    'Query': query,
                    'LLMs Missing Weidert': sources_missing,
                    'Status': 'Partial' if any_mentioned else 'Complete Gap'
                })
            
            summary_df = pd.DataFrame(query_summary)
            
            # Color code the status
            def highlight_status(row):
                if row['Status'] == 'Complete Gap':
                    return ['background-color: #ffcccc'] * len(row)
                else:
                    return ['background-color: #fff3cd'] * len(row)
            
            st.dataframe(
                summary_df.style.apply(highlight_status, axis=1),
                use_container_width=True,
                height=400
            )
            
            st.caption("🔴 **Complete Gap** = Weidert not mentioned by ANY LLM | 🟡 **Partial** = Mentioned by some LLMs but not all")
            
            st.markdown("---")
            
            # NEW: Sources Citation Analysis
            st.markdown("### 🔗 Sources Citation Analysis")
            st.caption("Analysis of URLs and sources cited in LLM responses")
            
            # Ensure Sources_Cited column exists
            if 'Sources_Cited' not in df_main.columns:
                df_main['Sources_Cited'] = df_main['Response'].apply(
                    lambda x: ', '.join(re.findall(r'https?://\S+', str(x))) if not str(x).startswith("ERROR") else ''
                )
            
            # Calculate citation metrics
            has_sources = df_main['Sources_Cited'].apply(lambda x: len(str(x)) > 0 and str(x) != '')
            responses_with_sources = has_sources.sum()
            citation_rate = (responses_with_sources / len(df_main) * 100)
            
            # Weidert URL citation
            if 'Weidert_URL_Cited' not in df_main.columns:
                df_main['Weidert_URL_Cited'] = df_main['Sources_Cited'].str.contains('weidert.com', case=False, na=False)
            
            weidert_url_citations = df_main['Weidert_URL_Cited'].sum()
            weidert_url_rate = (weidert_url_citations / len(df_main) * 100)
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Responses with Sources", f"{responses_with_sources}/{len(df_main)}", 
                         f"{citation_rate:.1f}%")
            
            with col2:
                st.metric("Weidert.com Citations", f"{weidert_url_citations}", 
                         f"{weidert_url_rate:.1f}%")
            
            with col3:
                # Average sources per response
                source_counts = df_main['Sources_Cited'].apply(
                    lambda x: len(str(x).split(', ')) if x and str(x) != '' else 0
                )
                avg_sources = source_counts.mean()
                st.metric("Avg Sources per Response", f"{avg_sources:.1f}")
            
            with col4:
                # Weidert URL citation when mentioned
                weidert_mentioned_df = df_main[df_main['Weidert_Mentioned']]
                if len(weidert_mentioned_df) > 0:
                    url_when_mentioned = (weidert_mentioned_df['Weidert_URL_Cited'].sum() / len(weidert_mentioned_df) * 100)
                    st.metric("URL When Mentioned", f"{url_when_mentioned:.1f}%")
                else:
                    st.metric("URL When Mentioned", "N/A")
            
            st.markdown("---")
            
            # Citation rate by LLM source
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Citation Rate by LLM")
                citation_by_source = df_main.groupby('Source').apply(
                    lambda x: (x['Sources_Cited'].apply(lambda y: len(str(y)) > 0 and str(y) != '').sum() / len(x) * 100)
                ).round(1)
                
                fig = px.bar(x=citation_by_source.index, y=citation_by_source.values,
                           title='Percentage of Responses with Sources',
                           labels={'x': 'LLM Source', 'y': 'Citation Rate (%)'},
                           color=citation_by_source.values,
                           color_continuous_scale='Blues',
                           text=citation_by_source.values)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### 🌐 Weidert.com Citation by LLM")
                weidert_citation_by_source = df_main.groupby('Source').apply(
                    lambda x: (x['Weidert_URL_Cited'].sum() / len(x) * 100)
                ).round(1)
                
                fig = px.bar(x=weidert_citation_by_source.index, y=weidert_citation_by_source.values,
                           title='Weidert.com Citation Rate',
                           labels={'x': 'LLM Source', 'y': 'Citation Rate (%)'},
                           color=weidert_citation_by_source.values,
                           color_continuous_scale='Greens',
                           text=weidert_citation_by_source.values)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Extract and analyze all URLs
            st.markdown("#### 🔝 Most Frequently Cited Domains")
            
            all_urls = []
            for sources in df_main['Sources_Cited']:
                if sources and str(sources) != '':
                    urls = str(sources).split(', ')
                    all_urls.extend(urls)
            
            if all_urls:
                # Extract domains from URLs
                import re
                from urllib.parse import urlparse
                
                domains = []
                for url in all_urls:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.netloc.replace('www.', '')
                        if domain:
                            domains.append(domain)
                    except:
                        pass
                
                if domains:
                    domain_counts = pd.Series(domains).value_counts().head(15)
                    
                    # Highlight Weidert domain
                    colors = ['#28a745' if 'weidert.com' in domain else '#6c757d' for domain in domain_counts.index]
                    
                    fig = px.bar(x=domain_counts.values, y=domain_counts.index, orientation='h',
                               title='Top 15 Most Cited Domains',
                               labels={'x': 'Number of Citations', 'y': 'Domain'},
                               text=domain_counts.values)
                    fig.update_traces(marker_color=colors, textposition='outside')
                    fig.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("🟢 Green = Weidert.com | ⚫ Gray = Other domains")
                    
                    # Show if Weidert is in top domains
                    weidert_rank = None
                    for rank, (domain, count) in enumerate(domain_counts.items(), 1):
                        if 'weidert.com' in domain:
                            weidert_rank = rank
                            break
                    
                    if weidert_rank:
                        st.success(f"✅ Weidert.com ranks **#{weidert_rank}** among cited domains with **{domain_counts['weidert.com']} citations**")
                    else:
                        st.warning("⚠️ Weidert.com does not appear in the top 15 cited domains")
                else:
                    st.info("No valid domains found in cited sources")
            else:
                st.info("No sources were cited in the responses")
            
            st.markdown("---")
            
            # Detailed source breakdown by query (for negative results)
            st.markdown("#### 🔍 Sources Cited in Negative Results")
            st.caption("What sources are LLMs citing when they DON'T mention Weidert?")
            
            negative_with_sources = negative_results[negative_results['Sources_Cited'].apply(lambda x: len(str(x)) > 0 and str(x) != '')]
            
            if len(negative_with_sources) > 0:
                st.info(f"📊 {len(negative_with_sources)} out of {len(negative_results)} negative responses ({len(negative_with_sources)/len(negative_results)*100:.1f}%) included source citations")
                
                # Extract domains from negative results
                negative_urls = []
                for sources in negative_with_sources['Sources_Cited']:
                    if sources and str(sources) != '':
                        urls = str(sources).split(', ')
                        negative_urls.extend(urls)
                
                negative_domains = []
                for url in negative_urls:
                    try:
                        parsed = urlparse(url)
                        domain = parsed.netloc.replace('www.', '')
                        if domain:
                            negative_domains.append(domain)
                    except:
                        pass
                
                if negative_domains:
                    negative_domain_counts = pd.Series(negative_domains).value_counts().head(10)
                    
                    fig = px.bar(x=negative_domain_counts.values, y=negative_domain_counts.index, orientation='h',
                               title='Top 10 Domains Cited When Weidert NOT Mentioned',
                               labels={'x': 'Number of Citations', 'y': 'Domain'},
                               color=negative_domain_counts.values,
                               color_continuous_scale='Reds',
                               text=negative_domain_counts.values)
                    fig.update_traces(textposition='outside')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.warning("💡 **Action Item**: These domains are competing for visibility. Consider creating comparison content or getting featured on these sites.")
            else:
                st.info("None of the negative results included source citations")
            
            st.markdown("---")
            
            # Detailed Query Breakdown - SHOW ALL 3 LLM RESPONSES
            st.markdown("### 🔍 Detailed Query-by-Query Breakdown")
            st.caption("Expandable view showing ALL LLM responses for each query (highlighting which didn't mention Weidert)")
            
            unique_queries = negative_results['Query'].unique()
            
            for query_idx, query in enumerate(unique_queries, 1):
                # Get ALL responses for this query (not just negatives)
                query_data = df_main[df_main['Query'] == query]
                
                # Check if ANY LLM mentioned Weidert for this query
                any_mentioned = query_data['Weidert_Mentioned'].any()
                
                # Status indicator
                if any_mentioned:
                    status_color = "#ffc107"
                    status_icon = "🟡"
                    status_text = "Partial Coverage"
                else:
                    status_color = "#dc3545"
                    status_icon = "🔴"
                    status_text = "Complete Gap"
                
                with st.expander(f"{status_icon} Query {query_idx}: {query[:80]}{'...' if len(query) > 80 else ''}", expanded=False):
                    # Show full query
                    st.markdown(f"**Full Query:** {query}")
                    
                    # Status message
                    if any_mentioned:
                        mentioned_by = query_data[query_data['Weidert_Mentioned']]['Source'].tolist()
                        not_mentioned_by = query_data[~query_data['Weidert_Mentioned']]['Source'].tolist()
                        st.info(f"✅ **Weidert mentioned by:** {', '.join(mentioned_by)}")
                        st.warning(f"❌ **Missing from:** {', '.join(not_mentioned_by)}")
                    else:
                        st.error("🔴 **CRITICAL**: Weidert NOT mentioned by ANY LLM for this query!")
                    
                    st.markdown("---")
                    
                    # Show ALL 3 responses in columns
                    cols = st.columns(min(len(query_data), 3))
                    
                    for idx, (_, row) in enumerate(query_data.iterrows()):
                        col_idx = idx % 3
                        
                        with cols[col_idx]:
                            # Determine if Weidert was mentioned in this response
                            weidert_mentioned = row['Weidert_Mentioned']
                            
                            # Header with color coding
                            if weidert_mentioned:
                                header_color = "#28a745"  # Green
                                status_badge = "✅ Mentioned"
                            else:
                                header_color = "#dc3545"  # Red
                                status_badge = "❌ Not Mentioned"
                            
                            st.markdown(f"""
                            <div style="background: {header_color}; padding: 0.5rem; 
                                       border-radius: 5px; margin-bottom: 0.5rem;">
                                <strong style="color: white;">{row['Source']}</strong>
                                <span style="color: white; float: right;">{status_badge}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Response text - FULL RESPONSE
                            response_text = str(row['Response'])
                            
                            # Show full response in an expandable area
                            with st.expander("📄 View Full Response", expanded=False):
                                st.markdown(response_text)
                            
                            # Show preview (first 300 chars)
                            preview_text = response_text[:300] + "..." if len(response_text) > 300 else response_text
                            st.markdown(f"**Preview:** {preview_text}")
                            
                            # Show position if Weidert was mentioned
                            if weidert_mentioned and 'Weidert_Position' in row:
                                st.success(f"📍 **Position:** {row['Weidert_Position']}")
                                if 'Context_Type' in row:
                                    context = row['Context_Type']
                                    context_icon = "😊" if context == "Positive" else "😐" if context == "Neutral" else "😟"
                                    st.info(f"{context_icon} **Context:** {context}")
                            
                            # Show competitors if mentioned
                            if row['Competitors_Found'] and str(row['Competitors_Found']) != '':
                                st.warning(f"🏆 **Competitors:** {row['Competitors_Found']}")
                            else:
                                st.caption("No competitors mentioned")
                            
                            # Response time
                            st.caption(f"⏱️ Response time: {row['Response_Time']}s")
                            
                            st.markdown("---")
            
            # Simple download
            st.download_button(
                "📥 Download Negative Results (CSV)",
                negative_results.to_csv(index=False),
                "weidert_queries_without_mentions.csv",
                "text/csv",
                help="Download all responses where Weidert was not mentioned"
            )
        
        else:
            st.success("🎉 **Excellent!** Weidert Group was mentioned in ALL query responses!")
        
        st.divider()
        
        # Download enhanced dataset
        st.subheader("📥 Download Enhanced Dataset")
        st.download_button(
            "Download Complete Analysis CSV",
            df_main.to_csv(index=False),
            "weidert_visibility_analysis.csv",
            "text/csv"
        )

# ─── TAB 3: COMPETITOR COMPARISON ─────────────────────────────────────────────
# ─── TAB 3: COMPETITOR COMPARISON ─────────────────────────────────────────────
with tab3:
    st.markdown("### 🏆 Competitor Comparison")
    st.caption("Head-to-head analysis comparing Weidert Group against key competitors")
    
    data_source = st.radio(
        "Choose data source:",
        ["Use results from Multi-LLM Generator", "Upload CSV file", "Run new queries"]
    )
    
    if data_source == "Use results from Multi-LLM Generator":
        if 'latest_results' in st.session_state:
            st.success("Using results from Multi-LLM Response Generator")
            df_comp = st.session_state.latest_results.copy()
            
            selected_competitors = st.multiselect(
                "Select Competitors to Analyze:",
                ["Weidert Group"] + COMPETITORS,
                default=["Weidert Group", "Stream Creative", "New Breed Revenue", "SmartBug Media"]
            )
            
            if st.button("🔍 Analyze Competitors", key="analyze_existing"):
                competitor_analysis = {}
                
                for competitor in selected_competitors:
                    search_term = "weidert" if competitor == "Weidert Group" else competitor.lower()
                    mentions = df_comp['Response'].apply(
                        lambda x: search_term in str(x).lower() and not str(x).startswith("ERROR")
                    )
                    positions = df_comp['Response'].apply(lambda x: analyze_position(x, competitor))
                    contexts = df_comp['Response'].apply(lambda x: analyze_context(x, competitor))
                    
                    competitor_analysis[competitor] = {
                        'mention_rate': (mentions.sum() / len(df_comp) * 100),
                        'avg_position': sum([p[1] for p in positions if p[1] > 0]) / max(sum([1 for p in positions if p[1] > 0]), 1),
                        'positive_context': sum([1 for c in contexts if c[0] == 'Positive']) / len(df_comp) * 100,
                        'first_third_rate': sum([1 for p in positions if p[0] == 'First Third']) / len(df_comp) * 100
                    }
                
                st.subheader("🏆 Competitor Performance Matrix")
                
                comparison_df = pd.DataFrame(competitor_analysis).T.round(1)
                comparison_df.columns = ['Mention Rate (%)', 'Avg Position', 'Positive Context (%)', 'First Third (%)']
                
                # Add ranking column
                comparison_df['Overall Rank'] = comparison_df['Mention Rate (%)'].rank(ascending=False).astype(int)
                
                st.dataframe(
                    comparison_df.style.background_gradient(subset=['Mention Rate (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['Positive Context (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['First Third (%)'], cmap='RdYlGn')
                                      .background_gradient(subset=['Avg Position'], cmap='RdYlGn_r'),
                    use_container_width=True
                )
                
                # Show Weidert's rank
                if 'Weidert Group' in comparison_df.index:
                    weidert_rank = comparison_df.loc['Weidert Group', 'Overall Rank']
                    total_competitors = len(comparison_df)
                    
                    if weidert_rank == 1:
                        st.success(f"🥇 **Weidert Group ranks #1** out of {total_competitors} competitors!")
                    elif weidert_rank <= 3:
                        st.info(f"🥈 **Weidert Group ranks #{int(weidert_rank)}** out of {total_competitors} competitors")
                    else:
                        st.warning(f"📊 **Weidert Group ranks #{int(weidert_rank)}** out of {total_competitors} competitors")
                
                st.divider()
                
                # NEW: Head-to-Head Comparison Radar Chart
                st.subheader("📊 Multi-Dimensional Comparison")
                st.caption("Radar chart showing performance across all metrics")
                
                # Create radar chart
                categories = ['Mention Rate', 'First Third Rate', 'Positive Context']
                
                fig = go.Figure()
                
                for competitor in selected_competitors:
                    if competitor in competitor_analysis:
                        values = [
                            competitor_analysis[competitor]['mention_rate'],
                            competitor_analysis[competitor]['first_third_rate'],
                            competitor_analysis[competitor]['positive_context']
                        ]
                        
                        fig.add_trace(go.Scatterpolar(
                            r=values,
                            theta=categories,
                            fill='toself',
                            name=competitor
                        ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Competitive Performance Radar"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # NEW: Win/Loss Analysis
                st.subheader("⚔️ Win/Loss Analysis")
                st.caption("Query-by-query breakdown: Where Weidert wins vs loses to competitors")
                
                # Analyze each query
                win_loss_data = []
                
                for query in df_comp['Query'].unique():
                    query_data = df_comp[df_comp['Query'] == query]
                    
                    # Check who appears in this query
                    weidert_appears = any(query_data['Response'].apply(
                        lambda x: 'weidert' in str(x).lower() and not str(x).startswith("ERROR")
                    ))
                    
                    competitors_appear = []
                    for comp in selected_competitors:
                        if comp != "Weidert Group":
                            comp_appears = any(query_data['Response'].apply(
                                lambda x: comp.lower() in str(x).lower() and not str(x).startswith("ERROR")
                            ))
                            if comp_appears:
                                competitors_appear.append(comp)
                    
                    # Classify outcome
                    if weidert_appears and not competitors_appear:
                        outcome = "🟢 Win (Weidert Only)"
                    elif weidert_appears and competitors_appear:
                        outcome = "🟡 Tie (Both Appear)"
                    elif not weidert_appears and competitors_appear:
                        outcome = "🔴 Loss (Competitor Only)"
                    else:
                        outcome = "⚪ Neither"
                    
                    win_loss_data.append({
                        'Query': query,
                        'Outcome': outcome,
                        'Competitors Present': ', '.join(competitors_appear) if competitors_appear else 'None'
                    })
                
                win_loss_df = pd.DataFrame(win_loss_data)
                
                # Summary counts
                col1, col2, col3, col4 = st.columns(4)
                
                wins = (win_loss_df['Outcome'] == '🟢 Win (Weidert Only)').sum()
                ties = (win_loss_df['Outcome'] == '🟡 Tie (Both Appear)').sum()
                losses = (win_loss_df['Outcome'] == '🔴 Loss (Competitor Only)').sum()
                neither = (win_loss_df['Outcome'] == '⚪ Neither').sum()
                
                with col1:
                    st.metric("🟢 Wins", wins, f"{wins/len(win_loss_df)*100:.1f}%")
                
                with col2:
                    st.metric("🟡 Ties", ties, f"{ties/len(win_loss_df)*100:.1f}%")
                
                with col3:
                    st.metric("🔴 Losses", losses, f"{losses/len(win_loss_df)*100:.1f}%")
                
                with col4:
                    st.metric("⚪ Neither", neither, f"{neither/len(win_loss_df)*100:.1f}%")
                
                # Show win/loss breakdown
                st.dataframe(win_loss_df, use_container_width=True, height=300)
                
                st.divider()
                
                # NEW: Co-occurrence Matrix
                st.subheader("🔗 Co-Mention Analysis")
                st.caption("How often competitors appear together with Weidert in the same responses")
                
                co_mention_matrix = []
                
                for comp in selected_competitors:
                    if comp == "Weidert Group":
                        continue
                    
                    # Count queries where both Weidert and this competitor appear
                    both_count = 0
                    weidert_only = 0
                    comp_only = 0
                    
                    for query in df_comp['Query'].unique():
                        query_data = df_comp[df_comp['Query'] == query]
                        
                        weidert_in_query = any(query_data['Response'].apply(
                            lambda x: 'weidert' in str(x).lower() and not str(x).startswith("ERROR")
                        ))
                        
                        comp_in_query = any(query_data['Response'].apply(
                            lambda x: comp.lower() in str(x).lower() and not str(x).startswith("ERROR")
                        ))
                        
                        if weidert_in_query and comp_in_query:
                            both_count += 1
                        elif weidert_in_query:
                            weidert_only += 1
                        elif comp_in_query:
                            comp_only += 1
                    
                    co_mention_matrix.append({
                        'Competitor': comp,
                        'Both Mentioned': both_count,
                        'Weidert Only': weidert_only,
                        'Competitor Only': comp_only,
                        'Co-Mention Rate (%)': round(both_count / max(both_count + weidert_only + comp_only, 1) * 100, 1)
                    })
                
                co_mention_df = pd.DataFrame(co_mention_matrix)
                
                if len(co_mention_df) > 0:
                    st.dataframe(co_mention_df, use_container_width=True)
                    
                    # Visualize co-mention rates
                    fig = px.bar(co_mention_df, x='Competitor', y='Co-Mention Rate (%)',
                               title='How Often Competitors Appear WITH Weidert',
                               color='Co-Mention Rate (%)',
                               color_continuous_scale='Bluered',
                               text='Co-Mention Rate (%)')
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("💡 **High co-mention rate** = Often compared together (direct competitors)\n\n**Low co-mention rate** = Appear in different contexts (different market positioning)")
                
                st.divider()
                
                # NEW: Competitive Strengths & Weaknesses
                st.subheader("💪 Competitive Strengths & Weaknesses")
                st.caption("Where Weidert outperforms vs underperforms against selected competitors")
                
                if 'Weidert Group' in comparison_df.index:
                    weidert_metrics = comparison_df.loc['Weidert Group']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ✅ Strengths")
                        strengths = []
                        
                        # Check each metric
                        for metric in ['Mention Rate (%)', 'First Third (%)', 'Positive Context (%)']:
                            weidert_value = weidert_metrics[metric]
                            competitor_values = [comparison_df.loc[comp, metric] for comp in selected_competitors if comp != 'Weidert Group']
                            
                            if competitor_values:
                                avg_competitor = sum(competitor_values) / len(competitor_values)
                                
                                if weidert_value > avg_competitor:
                                    diff = weidert_value - avg_competitor
                                    strengths.append(f"**{metric}**: {weidert_value:.1f}% (↑ {diff:.1f}% above avg)")
                        
                        if strengths:
                            for strength in strengths:
                                st.success(strength)
                        else:
                            st.info("No clear strengths identified")
                    
                    with col2:
                        st.markdown("#### ⚠️ Weaknesses")
                        weaknesses = []
                        
                        # Check each metric
                        for metric in ['Mention Rate (%)', 'First Third (%)', 'Positive Context (%)']:
                            weidert_value = weidert_metrics[metric]
                            competitor_values = [comparison_df.loc[comp, metric] for comp in selected_competitors if comp != 'Weidert Group']
                            
                            if competitor_values:
                                avg_competitor = sum(competitor_values) / len(competitor_values)
                                
                                if weidert_value < avg_competitor:
                                    diff = avg_competitor - weidert_value
                                    weaknesses.append(f"**{metric}**: {weidert_value:.1f}% (↓ {diff:.1f}% below avg)")
                        
                        if weaknesses:
                            for weakness in weaknesses:
                                st.warning(weakness)
                        else:
                            st.success("No clear weaknesses identified")
                
                st.divider()
                
                # Original charts with better titles
                st.subheader("📈 Performance Comparison Charts")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    mention_rates = [competitor_analysis[comp]['mention_rate'] for comp in selected_competitors]
                    fig = px.bar(x=selected_competitors, y=mention_rates,
                                title="Overall Mention Rate Comparison",
                                labels={'x': 'Company', 'y': 'Mention Rate (%)'},
                                color=mention_rates,
                                color_continuous_scale='RdYlGn',
                                text=mention_rates)
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    first_third_rates = [competitor_analysis[comp]['first_third_rate'] for comp in selected_competitors]
                    fig = px.bar(x=selected_competitors, y=first_third_rates,
                                title="Early Position Rate (First Third)",
                                labels={'x': 'Company', 'y': 'First Third Rate (%)'},
                                color=first_third_rates,
                                color_continuous_scale='RdYlGn',
                                text=first_third_rates)
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Download comparison data
                st.download_button(
                    "📥 Download Competitor Analysis (CSV)",
                    comparison_df.to_csv(),
                    "weidert_competitor_comparison.csv",
                    "text/csv"
                )
        else:
            st.warning("No results available. Please run queries in Tab 1 first.")
    
    elif data_source == "Upload CSV file":
        uploaded_comp = st.file_uploader("Upload CSV", type="csv", key="comp_upload")
        if uploaded_comp:
            df_comp = pd.read_csv(uploaded_comp)
            st.success("File uploaded! Configure competitors above and click 'Analyze Competitors'")
    
    else:
        use_predefined = st.checkbox("Use predefined 20 queries", value=True)
        
        if use_predefined:
            st.text_area("Queries to run:", value="\n".join([q.split("--")[0].strip() for q in PREDEFINED_QUERIES]), height=200, disabled=True)
        else:
            comparison_queries = st.text_area(
                "Comparison Queries (one per line):",
                height=150,
                placeholder="Compare B2B marketing agencies\nBest HubSpot partners"
            )
        
        if st.button("🔍 Run Competitor Analysis", key="competitor_analysis"):
            queries = PREDEFINED_QUERIES if use_predefined else [q.strip() for q in comparison_queries.splitlines() if q.strip()]
            
            if queries:
                with st.spinner("Running competitor analysis..."):
                    results = process_queries_parallel(queries)
                    df_comp = pd.DataFrame(results)
                    st.session_state.latest_results = df_comp
                    st.success("✅ Analysis complete! Select competitors above and click 'Analyze Competitors'")
                    st.rerun()


# ─── TAB 4: GAP ANALYSIS & OPPORTUNITIES ─────────────────────────────────────
with tab4:
    st.markdown("### 🎯 Gap Analysis & Improvement Opportunities")
    st.caption("Side-by-side comparison highlighting where Weidert needs to improve visibility")
    
    gap_file = st.file_uploader("Upload results CSV", type="csv", key="gap_upload")
    
    df_gap = None
    
    if 'latest_results' in st.session_state:
        use_latest_gap = st.checkbox("Use results from Multi-LLM Generator", value=True, key="gap_use_latest")
        if use_latest_gap:
            df_gap = st.session_state.latest_results.copy()
    
    if gap_file and df_gap is None:
        df_gap = pd.read_csv(gap_file)
    
    if df_gap is not None:
        # Ensure necessary columns
        if 'Weidert_Mentioned' not in df_gap.columns:
            df_gap['Weidert_Mentioned'] = df_gap['Response'].apply(
                lambda x: 'weidert' in str(x).lower() and not str(x).startswith("ERROR")
            )
        
        if 'Competitors_Found' not in df_gap.columns:
            competitor_analysis = df_gap['Response'].apply(extract_competitors_detailed)
            df_gap['Competitors_Found'] = [', '.join(c[0]) if c[0] else '' for c in competitor_analysis]
        
        df_gap['Has_Competitors'] = df_gap['Competitors_Found'].apply(lambda x: len(str(x)) > 0 and str(x) != '')
        
        # Create analysis categories
        st.subheader("📊 Response Categories")
        
        # Calculate categories
        weidert_only = df_gap[df_gap['Weidert_Mentioned'] & ~df_gap['Has_Competitors']]
        competitors_only = df_gap[~df_gap['Weidert_Mentioned'] & df_gap['Has_Competitors']]
        both_mentioned = df_gap[df_gap['Weidert_Mentioned'] & df_gap['Has_Competitors']]
        neither_mentioned = df_gap[~df_gap['Weidert_Mentioned'] & ~df_gap['Has_Competitors']]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{len(weidert_only)}</div>
                <div style="font-size: 0.9rem;">✅ Weidert Only</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">No competitors mentioned</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{len(competitors_only)}</div>
                <div style="font-size: 0.9rem;">⚠️ Competitors Only</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Weidert NOT mentioned</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #ffc107 0%, #ff9800 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{len(both_mentioned)}</div>
                <div style="font-size: 0.9rem;">🤝 Both Mentioned</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Weidert + competitors</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #6c757d 0%, #5a6268 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                <div style="font-size: 2rem; font-weight: bold;">{len(neither_mentioned)}</div>
                <div style="font-size: 0.9rem;">❌ Neither</div>
                <div style="font-size: 0.8rem; opacity: 0.9;">Generic responses</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # CRITICAL: Competitors mentioned but NOT Weidert
        st.subheader("🚨 CRITICAL GAPS: Competitors Mentioned, Weidert Missing")
        st.caption("These are the highest priority opportunities - Weidert should be mentioned here but isn't")
        
        if len(competitors_only) > 0:
            # Show distribution by source
            col1, col2 = st.columns(2)
            
            with col1:
                gap_by_source = competitors_only.groupby('Source').size()
                fig = px.pie(values=gap_by_source.values, names=gap_by_source.index,
                           title="Critical Gaps by LLM Source",
                           color_discrete_sequence=['#dc3545', '#c82333', '#bd2130'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Most common competitors in these gaps
                all_comp_mentions = []
                for comp_str in competitors_only['Competitors_Found']:
                    if comp_str and str(comp_str) != '':
                        all_comp_mentions.extend(str(comp_str).split(', '))
                
                if all_comp_mentions:
                    comp_counts = pd.Series(all_comp_mentions).value_counts().head(5)
                    fig = px.bar(x=comp_counts.values, y=comp_counts.index, orientation='h',
                               title="Competitors Appearing in Gaps",
                               labels={'x': 'Mentions', 'y': 'Competitor'},
                               color_discrete_sequence=['#dc3545'])
                    st.plotly_chart(fig, use_container_width=True)
            
            # Side-by-side query comparison
            st.markdown("---")
            st.markdown("### 📋 Detailed Gap Analysis: Query-by-Query Breakdown")
            
            # Group by query to show all LLM responses together
            gap_queries = competitors_only['Query'].unique()
            
            for query in gap_queries[:10]:  # Show first 10 critical gaps
                query_responses = df_gap[df_gap['Query'] == query]
                
                # Check if ANY response mentions Weidert
                weidert_mentioned_anywhere = query_responses['Weidert_Mentioned'].any()
                
                with st.expander(f"🔍 Query: {query[:100]}{'...' if len(query) > 100 else ''}", expanded=False):
                    # Show query details
                    st.markdown(f"**Full Query:** {query}")
                    
                    if weidert_mentioned_anywhere:
                        st.info("ℹ️ Weidert IS mentioned by at least one LLM - opportunity to improve consistency")
                    else:
                        st.error("⚠️ Weidert NOT mentioned by ANY LLM - critical visibility gap!")
                    
                    st.markdown("---")
                    
                    # Create columns for side-by-side comparison
                    sources = query_responses['Source'].unique()
                    cols = st.columns(min(len(sources), 3))
                    
                    for idx, (_, row) in enumerate(query_responses.iterrows()):
                        col_idx = idx % 3
                        
                        with cols[col_idx]:
                            # Header with status indicator
                            weidert_in_response = row['Weidert_Mentioned']
                            competitors_in_response = row['Has_Competitors']
                            
                            if weidert_in_response and competitors_in_response:
                                status = "🟢 Good"
                                color = "#28a745"
                            elif weidert_in_response:
                                status = "🟡 OK"
                                color = "#ffc107"
                            elif competitors_in_response:
                                status = "🔴 Gap"
                                color = "#dc3545"
                            else:
                                status = "⚪ Generic"
                                color = "#6c757d"
                            
                            st.markdown(f"""
                            <div style="background: {color}; padding: 0.5rem; border-radius: 5px; margin-bottom: 0.5rem;">
                                <strong style="color: white;">{row['Source']}</strong>
                                <span style="color: white; float: right;">{status}</span>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show response with highlighting
                            response_text = str(row['Response'])
                            
                            # Truncate if too long
                            if len(response_text) > 500:
                                response_text = response_text[:500] + "..."
                            
                            # Highlight competitors
                            highlighted_text = response_text
                            if competitors_in_response:
                                competitors_list = str(row['Competitors_Found']).split(', ')
                                for comp in competitors_list:
                                    if comp and comp != '':
                                        highlighted_text = re.sub(
                                            f'({re.escape(comp)})',
                                            r'**\1**',
                                            highlighted_text,
                                            flags=re.IGNORECASE
                                        )
                            
                            st.markdown(highlighted_text)
                            
                            # Show what's mentioned
                            if competitors_in_response:
                                st.markdown(f"**Competitors:** {row['Competitors_Found']}")
                            
                            if weidert_in_response:
                                st.markdown("✅ **Weidert mentioned**")
                            else:
                                st.markdown("❌ **Weidert NOT mentioned**")
                            
                            st.markdown("---")
            
            if len(gap_queries) > 10:
                st.info(f"Showing 10 of {len(gap_queries)} critical gaps. Download full data for complete analysis.")
        
        else:
            st.success("🎉 Great news! No critical gaps found - Weidert appears whenever competitors do!")
        
        st.divider()
        
        # Where Weidert IS winning
        st.subheader("🏆 SUCCESS STORIES: Weidert Mentioned, Competitors Not")
        st.caption("These queries show where Weidert has strong visibility advantage")
        
        if len(weidert_only) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                success_by_source = weidert_only.groupby('Source').size()
                fig = px.bar(x=success_by_source.index, y=success_by_source.values,
                           title="Success Stories by LLM Source",
                           labels={'x': 'Source', 'y': 'Count'},
                           color_discrete_sequence=['#28a745'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Position analysis for these wins
                if 'Weidert_Position' not in weidert_only.columns:
                    position_analysis = weidert_only['Response'].apply(lambda x: analyze_position(x, "Weidert"))
                    weidert_only['Weidert_Position'] = [p[0] for p in position_analysis]
                
                position_dist = weidert_only['Weidert_Position'].value_counts()
                fig = px.pie(values=position_dist.values, names=position_dist.index,
                           title="Weidert Position in Success Stories",
                           color_discrete_sequence=['#28a745', '#20c997', '#17a2b8'])
                st.plotly_chart(fig, use_container_width=True)
            
            # Show example success queries
            st.markdown("**Example Success Queries:**")
            for i, query in enumerate(weidert_only['Query'].unique()[:5], 1):
                st.markdown(f"{i}. {query}")
        
        else:
            st.info("No queries where Weidert is the only agency mentioned.")
        
        st.divider()
        
        # Competitive battlegrounds
        st.subheader("⚔️ COMPETITIVE BATTLEGROUNDS: Both Mentioned")
        st.caption("Queries where Weidert competes directly with other agencies")
        
        if len(both_mentioned) > 0:
            # Position comparison in competitive scenarios
            if 'Weidert_Position' not in both_mentioned.columns:
                position_analysis = both_mentioned['Response'].apply(lambda x: analyze_position(x, "Weidert"))
                both_mentioned['Weidert_Position'] = [p[0] for p in position_analysis]
                both_mentioned['Weidert_Sentence_Num'] = [p[1] for p in position_analysis]
            
            col1, col2 = st.columns(2)
            
            with col1:
                position_in_competitive = both_mentioned['Weidert_Position'].value_counts()
                fig = px.bar(x=position_in_competitive.index, y=position_in_competitive.values,
                           title="Weidert's Position in Competitive Responses",
                           labels={'x': 'Position', 'y': 'Count'},
                           color_discrete_sequence=['#ffc107'])
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average sentence position vs competitors
                avg_weidert_position = both_mentioned['Weidert_Sentence_Num'].mean()
                
                st.markdown(f"""
                <div style="background: #ffc107; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
                    <div style="font-size: 1.5rem; font-weight: bold;">{avg_weidert_position:.1f}</div>
                    <div style="font-size: 0.9rem;">Avg Sentence Position</div>
                    <div style="font-size: 0.8rem; opacity: 0.9;">Lower is better</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("")
                st.markdown("**Interpretation:**")
                if avg_weidert_position <= 3:
                    st.success("✅ Excellent - Weidert appears early in responses")
                elif avg_weidert_position <= 5:
                    st.info("🟡 Good - Weidert appears in first half of responses")
                else:
                    st.warning("⚠️ Opportunity - Weidert appears late in responses")
            
            # Show which competitors Weidert is fighting against
            st.markdown("**Most Frequent Competitors in Battlegrounds:**")
            all_battle_comps = []
            for comp_str in both_mentioned['Competitors_Found']:
                if comp_str and str(comp_str) != '':
                    all_battle_comps.extend(str(comp_str).split(', '))
            
            if all_battle_comps:
                comp_battle_counts = pd.Series(all_battle_comps).value_counts().head(5)
                for comp, count in comp_battle_counts.items():
                    st.markdown(f"• **{comp}**: {count} co-mentions")
        
        else:
            st.info("No queries where both Weidert and competitors are mentioned together.")
        
        st.divider()
        
        # Action Items
        st.subheader("🎯 Recommended Actions")
        st.caption("Based on the gap analysis, here's what Weidert should focus on")
        
        actions = []
        
        if len(competitors_only) > 0:
            gap_rate = (len(competitors_only) / len(df_gap) * 100)
            actions.append({
                'priority': '🔴 HIGH',
                'action': f'Address {len(competitors_only)} critical visibility gaps ({gap_rate:.1f}% of queries)',
                'detail': f'Create content targeting these queries where competitors appear but Weidert doesn\'t'
            })
            
            # Find most common competitors in gaps
            if all_comp_mentions:
                top_gap_comp = pd.Series(all_comp_mentions).value_counts().index[0]
                actions.append({
                    'priority': '🔴 HIGH',
                    'action': f'Competitive positioning against {top_gap_comp}',
                    'detail': f'{top_gap_comp} appears most often in gaps - create comparison content'
                })
        
        if len(both_mentioned) > 0 and avg_weidert_position > 5:
            actions.append({
                'priority': '🟡 MEDIUM',
                'action': 'Improve early mention positioning',
                'detail': 'Weidert appears late in competitive responses - strengthen thought leadership'
            })
        
        if len(weidert_only) > 0:
            success_rate = (len(weidert_only) / len(df_gap) * 100)
            actions.append({
                'priority': '🟢 MAINTAIN',
                'action': f'Protect {len(weidert_only)} success stories ({success_rate:.1f}% of queries)',
                'detail': 'Continue reinforcing content in areas where Weidert has exclusive visibility'
            })
        
        if not actions:
            actions.append({
                'priority': '🟢 GOOD',
                'action': 'Maintain current visibility levels',
                'detail': 'Performance is strong - monitor for changes and continue current strategies'
            })
        
        for action in actions:
            st.markdown(f"""
            <div style="border-left: 4px solid {'#dc3545' if 'HIGH' in action['priority'] else '#ffc107' if 'MEDIUM' in action['priority'] else '#28a745'}; 
                        padding: 1rem; margin: 0.5rem 0; background: #f8f9fa; border-radius: 5px;">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">{action['priority']} {action['action']}</div>
                <div style="color: #666;">{action['detail']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Download gap analysis
        st.divider()
        st.subheader("📥 Export Gap Analysis")
        
        # Create detailed export
        export_data = df_gap.copy()
        export_data['Category'] = 'Neither'
        export_data.loc[export_data['Weidert_Mentioned'] & ~export_data['Has_Competitors'], 'Category'] = 'Weidert Only'
        export_data.loc[~export_data['Weidert_Mentioned'] & export_data['Has_Competitors'], 'Category'] = 'Competitors Only (CRITICAL GAP)'
        export_data.loc[export_data['Weidert_Mentioned'] & export_data['Has_Competitors'], 'Category'] = 'Both Mentioned'
        
        st.download_button(
            "📥 Download Full Gap Analysis CSV",
            export_data.to_csv(index=False),
            "weidert_gap_analysis.csv",
            "text/csv",
            help="Download complete gap analysis with all responses categorized"
        )
    
    else:
        st.info("Please run queries in Tab 1 or upload a CSV file to perform gap analysis.")

# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.9rem; padding:2rem 0;'>
    <p><strong>Weidert Group LLM Search Visibility Tool</strong></p>
    <p>AI-Powered Competitive Intelligence • Brand Visibility Analytics</p>
    <p>Powered by OpenAI, Google Gemini, and Perplexity AI</p>
    <p style='font-size:0.8rem; margin-top:1rem;'>
        <a href='https://www.weidert.com' target='_blank' style='color:#e64626;'>Weidert Group</a>
        | B2B Industrial Marketing Experts
    </p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR TIPS ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    
    tips = [
        "**Start with predefined queries** for comprehensive baseline analysis",
        "**Compare branded vs non-branded** to measure organic visibility",
        "**Track position over time** to measure content impact",
        "**Monitor competitor mentions** to identify competitive gaps",
        "**Export results regularly** to build historical trends"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")
    
    st.markdown("---")
    st.subheader("🎯 About This Tool")
    st.markdown("""
    This tool helps Weidert Group monitor how Large Language Models (ChatGPT, Gemini, Perplexity) 
    respond to queries related to B2B industrial marketing services.
    
    **Key Metrics:**
    - Mention Rate: How often Weidert appears
    - Position: Where in responses Weidert appears
    - Context: Sentiment of Weidert mentions
    - Competition: Other agencies mentioned
    """)

# ─── SESSION STATE INITIALIZATION ─────────────────────────────────────────
if 'template_query' not in st.session_state:
    st.session_state.template_query = ''

if 'run_triggered' not in st.session_state:
    st.session_state.run_triggered = False

if 'use_predefined' not in st.session_state:
    st.session_state.use_predefined = False

if 'latest_results' not in st.session_state:
    st.session_state.latest_results = None
