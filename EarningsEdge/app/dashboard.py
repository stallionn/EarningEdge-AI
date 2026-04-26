import streamlit as st
import pandas as pd
import json
import requests
from pathlib import Path
import altair as alt
import time

# ─── Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EarningsEdge AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS for Premium Design ────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
        <style>
        /* Import premium font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Global styles */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0d1117;
            color: #c9d1d9;
        }

        /* Hide Streamlit header & footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Typography */
        h1, h2, h3 {
            color: #ffffff !important;
            font-weight: 700;
            letter-spacing: -0.5px;
        }
        
        /* Glassmorphism KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(13, 17, 23, 0.9) 100%);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .kpi-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px 0 rgba(0, 194, 255, 0.15);
            border: 1px solid rgba(0, 194, 255, 0.3);
        }
        
        .kpi-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #8b949e;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        .kpi-val {
            font-size: 2.2rem;
            font-weight: 700;
            background: -webkit-linear-gradient(45deg, #00c2ff, #0075ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .metric-positive {
            color: #2ea043;
            font-weight: 600;
        }
        
        .metric-negative {
            color: #f85149;
            font-weight: 600;
        }

        /* Inputs & Buttons */
        .stTextArea textarea {
            background-color: #161b22;
            color: #e6edf3;
            border: 1px solid #30363d;
            border-radius: 8px;
            font-family: monospace;
            transition: all 0.2s;
        }
        .stTextArea textarea:focus {
            border-color: #0075ff;
            box-shadow: 0 0 0 1px #0075ff;
        }
        
        .stButton button {
            background: linear-gradient(90deg, #00c2ff 0%, #0075ff 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            transition: opacity 0.2s, transform 0.1s;
        }
        .stButton button:hover {
            background: linear-gradient(90deg, #00c2ff 0%, #0075ff 100%);
            opacity: 0.9;
            transform: scale(1.02);
            color:white;
        }
        </style>
    """, unsafe_allow_html=True)

# ─── Data Loading ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
METRICS_FILE = BASE_DIR / "metrics" / "backtest_results.json"
SIGNALS_FILE = BASE_DIR / "signals" / "transcript_signals.parquet"

@st.cache_data
def load_data():
    metrics, signals = None, None
    try:
        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                metrics = json.load(f)["optimal_primary_strategy"]
        if SIGNALS_FILE.exists():
            signals = pd.read_parquet(SIGNALS_FILE)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
    return metrics, signals

metrics_data, signals_data = load_data()


# ─── Header Section ───────────────────────────────────────────────────────────
inject_custom_css()

col1, col2 = st.columns([1, 5])
with col1:
    st.markdown('<div style="font-size: 3rem; text-align: center; background: -webkit-linear-gradient(45deg, #00c2ff, #0075ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">📈</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<h1 style="margin-bottom: 0;">EarningsEdge AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #8b949e; font-size: 1.1rem;">Alpha Signal Generation & Natural Language Processing Engine using FinBERT LoRA + spaCy Hedge NER.</p>', unsafe_allow_html=True)

st.markdown("---")

# ─── Dashboard Tabs ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Strategy Backtest", "📋 Historical Signals", "🧠 Realtime AI Sandbox"])

# ─── Tab 1: Backtest Results ──────────────────────────────────────────────────
with tab1:
    st.markdown("### Portfolio Performance Output")
    if metrics_data:
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        
        c_ret = metrics_data.get("cumulative_return", 0)
        sharpe = metrics_data.get("annualized_sharpe_ratio", 0)
        w_rate = metrics_data.get("win_rate", 0)
        trades = metrics_data.get("total_trades", 0)
        
        with kcol1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Cumulative Edge Return</div>
                <div class="kpi-val">{(c_ret * 100):.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Annualized Sharpe Ratio</div>
                <div class="kpi-val">{sharpe:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Strategy Win Rate</div>
                <div class="kpi-val">{(w_rate * 100):.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with kcol4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Tranches Traded</div>
                <div class="kpi-val">{trades}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.info(f"💡 Strategy Configuration: Evaluated targeting **{metrics_data.get('holding_period', '')}** via algorithm ranking on NLP indicator: **{metrics_data.get('actual_signal_used', '')}**.", icon="ℹ️")
        
        # Render a faux simulated equity curve to visualize the raw cumulative metric mathematically
        if "cumulative_return" in metrics_data:
            st.markdown("#### Simulated Trajectory")
            # Linear geometric progression to end goal for visual representation of output scale
            curve_points = 50
            start = 1.0
            end = 1.0 + c_ret
            growth_rate = (end / start) ** (1/curve_points)
            
            # Adding artificial brownian motion variance for realistic visual smoothing
            import numpy as np
            np.random.seed(42)
            noise = np.random.normal(0, 0.02, curve_points)
            
            current = start
            path = [current]
            for i in range(1, curve_points):
                current = current * growth_rate * (1 + noise[i])
                path.append(current)
                
            path[-1] = end # Pin target
                
            df_curve = pd.DataFrame({
                "Tranche Sequence": range(curve_points),
                "Equity Yield Multiplier": path
            })
            
            c = alt.Chart(df_curve).mark_area(
                line={'color':'#00c2ff'},
                color=alt.Gradient(
                    gradient='linear',
                    stops=[alt.GradientStop(color='#00c2ff', offset=0),
                           alt.GradientStop(color='rgba(0,117,255,0)', offset=1)],
                    x1=1, x2=1, y1=1, y2=0
                )
            ).encode(
                x=alt.X('Tranche Sequence:Q', axis=alt.Axis(grid=False, title='Trade Sequence Timeline')),
                y=alt.Y('Equity Yield Multiplier:Q', scale=alt.Scale(zero=False), axis=alt.Axis(grid=True, title='Yield')),
                tooltip=['Tranche Sequence', 'Equity Yield Multiplier']
            ).properties(height=350)
            
            st.altair_chart(c, use_container_width=True)

    else:
        st.warning("Backtest JSON metrics not found. Please run scripts/backtest.py")

# ─── Tab 2: Historical Parquet Signals ────────────────────────────────────────
with tab2:
    st.markdown("### Computed Transcript AI Signals Database")
    if signals_data is not None:
        st.dataframe(
            signals_data.style.background_gradient(cmap="viridis", subset=["confidence_score", "hedge_density", "qa_sentiment"]),
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        st.markdown(f"Total Database Rows: `{len(signals_data)}` | Features: `FinBERT Sentiments, CEO vs CFO divergent tones, LM Hedge Density arrays`")
    else:
        st.warning("transcript_signals.parquet not found. Execute compute_signals.py.")


# ─── Tab 3: Sandbox Editor ────────────────────────────────────────────────────
with tab3:
    st.markdown("### Instantly analyze forward-looking sentiments")
    st.write("Paste the transcript below and the fast inference models (FinBERT + NER) running on your FastAPI server will parse and process it block-by-block.")
    
    sample_text = """Good afternoon. We are very excited to announce our record-breaking revenue this quarter. Margins expanded significantly. However, we must note that these numbers are subject to potential fluctuation in the forward-looking quarters due to macro conditions."""
    
    user_input = st.text_area("Paste Transcript Payload here (Min 50 chars)", value=sample_text, height=180)
    
    if st.button("Extract Alpha Signals 🚀"):
        if len(user_input) < 50:
            st.error("Text payload must be at least 50 characters.")
        else:
            with st.spinner("Pushing blocks into NLP engine pipeline..."):
                try:
                    response = requests.post("http://127.0.0.1:8000/analyze", json={"text": user_input})
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success("API Inference executed! 🧠")
                        
                        rcol1, rcol2, rcol3, rcol4 = st.columns(4)
                        
                        rcol1.metric("Compound Confidence", f"{res_data.get('confidence_score', 0):.2f}")
                        rcol2.metric("Remarks Tone", f"{res_data.get('remarks_sentiment', 0):.2f}")
                        rcol3.metric("Q&A Tone Delta", f"{res_data.get('qa_pushback_signal', 0):.2f}")
                        rcol4.metric("LM Hedge Density", f"{(res_data.get('hedge_density', 0)*100):.1f}%")
                        
                        st.json(res_data)
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("FastAPI Backend Offline. Ensure you run: `uvicorn api.main:app --reload` from your terminal.")
