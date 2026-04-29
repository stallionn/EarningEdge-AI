import streamlit as st
import pandas as pd
import json
import requests
from pathlib import Path
import altair as alt
import time
import sys
import io

# Optional imports for PDF & Audio extraction
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    from pdf_extractor import extract_pdf_text
except ImportError:
    pass

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
    st.markdown("### Mix Text, PDF, and Real-time Audio Alpha Generation")
    st.write("Upload a PDF transcript and/or an Audio file. Audio will be streamed second-by-second to identify hesitation, pauses, and reluctance.")
    
    # New Combined Analysis Section
    st.markdown("---")
    st.markdown("### 🎯 Combined Audio + Transcript Analysis (AI-Powered)")
    st.write("Upload both audio file and transcript for comprehensive analysis with AI buy/sell recommendation.")
    
    combined_audio = st.file_uploader("Upload Audio File", type=["wav", "mp3"], key="combined_audio")
    combined_transcript = st.text_area("Paste Full Transcript Here", height=150, key="combined_transcript")
    
    if st.button("Run Combined Analysis with AI 🤖", key="combined_btn"):
        if combined_audio and len(combined_transcript) > 50:
            with st.spinner("Processing audio and transcript with AI analysis..."):
                try:
                    files = {"file": (combined_audio.name, combined_audio.getvalue(), combined_audio.type)}
                    data = {"transcript_text": combined_transcript}
                    
                    response = requests.post("http://127.0.0.1:8000/analyze/combined", files=files, data=data)
                    
                    if response.status_code == 200:
                        res_data = response.json()
                        st.success("Combined Analysis Complete! 🎯")
                        
                        # Main combined score
                        c_col1, c_col2, c_col3 = st.columns(3)
                        c_col1.metric("Combined Confidence", f"{(res_data.get('combined_confidence', 0)*100):.2f}%")
                        c_col2.metric("Audio Quality Score", f"{(res_data.get('audio_confidence', 0)*100):.2f}%")
                        c_col3.metric("Transcript Specificity", f"{(res_data.get('transcript_specificity', 0)*100):.2f}%")
                        
                        # AI Analysis
                        st.markdown("### 🤖 AI Analysis & Recommendation")
                        st.info(res_data.get('ai_analysis', 'AI analysis not available'))
                        
                        # Audio Metrics
                        with st.expander("🔊 Audio Quality Details"):
                            a_col1, a_col2, a_col3 = st.columns(3)
                            a_col1.metric("SNR", f"{res_data.get('audio_metrics', {}).get('snr_db', 0):.2f} dB")
                            a_col2.metric("Quality", res_data.get('audio_metrics', {}).get('quality_flag', 'N/A'))
                            a_col3.metric("Voice Activity", f"{(res_data.get('audio_metrics', {}).get('voice_activity_ratio', 0)*100):.1f}%")
                            st.metric("Hesitation Pauses (>500ms)", res_data.get('audio_metrics', {}).get('hesitation_pauses_500ms', 0))
                            st.metric("Long Pauses (>1000ms)", res_data.get('audio_metrics', {}).get('hesitation_pauses_1000ms', 0))
                        
                        # Transcript Metrics
                        with st.expander("📋 Transcript Detail Analysis"):
                            t_col1, t_col2, t_col3 = st.columns(3)
                            t_col1.metric("Word Count", res_data.get('transcript_metrics', {}).get('word_count', 0))
                            t_col2.metric("Financial Numbers", res_data.get('transcript_metrics', {}).get('financial_numbers', 0))
                            t_col3.metric("Dollar Amounts", res_data.get('transcript_metrics', {}).get('dollar_amounts', 0))
                            st.metric("Percentages Mentioned", res_data.get('transcript_metrics', {}).get('percentages', 0))
                            st.metric("Financial Keywords", res_data.get('transcript_metrics', {}).get('financial_keywords', 0))
                        
                        # NLP Signals
                        with st.expander("🧠 NLP Sentiment Signals"):
                            n_col1, n_col2, n_col3 = st.columns(3)
                            n_col1.metric("Remarks Sentiment", f"{res_data.get('nlp_signals', {}).get('remarks_sentiment', 0):.2f}")
                            n_col2.metric("Q&A Sentiment", f"{res_data.get('nlp_signals', {}).get('qa_sentiment', 0):.2f}")
                            n_col3.metric("Hedge Density", f"{(res_data.get('nlp_signals', {}).get('hedge_density', 0)*100):.1f}%")
                            st.metric("CEO/CFO Divergence", f"{res_data.get('nlp_signals', {}).get('ceo_cfo_divergence', 0):.2f}")
                        
                        # Score Breakdown
                        with st.expander("📊 Score Composition"):
                            breakdown = res_data.get('score_breakdown', {})
                            st.json(breakdown)
                        
                        # Full JSON
                        with st.expander("📄 Full Response JSON"):
                            st.json(res_data)
                            
                    else:
                        st.error(f"API Error ({response.status_code}): {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("FastAPI Backend Offline. Ensure you run: `uvicorn api.main:app --reload`")
        else:
            st.error("Please upload both an audio file and provide transcript text (min 50 characters).")
    
    st.markdown("---")
    
    t3col1, t3col2 = st.columns([5, 4])
    
    with t3col1:
        st.markdown("**Transcript Processing (Text & PDF)**")
        pdf_file = st.file_uploader("Upload Transcript PDF", type=["pdf"])
        
        sample_text = """Good afternoon. We are very excited to announce our record-breaking revenue this quarter. Margins expanded significantly. However, we must note that these numbers are subject to potential fluctuation in the forward-looking quarters due to macro conditions."""
        
        extracted_text = ""
        if pdf_file:
            with st.spinner("Extracting from PDF..."):
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(pdf_file.read())
                    temp_pdf = f.name
                extracted_text = extract_pdf_text(temp_pdf)
                os.remove(temp_pdf)
                st.success("PDF Extracted successfully!")
                
        user_input = st.text_area("Paste Transcript Payload here (Min 50 chars)", value=extracted_text if extracted_text else sample_text, height=180)
        
        if st.button("Extract Text Alpha Signals 🚀"):
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
                            rcol1.metric("Compound Confidence", f"{(res_data.get('confidence_score', 0)*100):.2f}%")
                            rcol2.metric("Remarks Tone", f"{res_data.get('remarks_sentiment', 0):.2f}")
                            rcol3.metric("Q&A Tone Delta", f"{res_data.get('qa_pushback_signal', 0):.2f}")
                            rcol4.metric("LM Hedge Density", f"{(res_data.get('hedge_density', 0)*100):.1f}%")
                            
                            st.json(res_data)
                        else:
                            st.error(f"API Error ({response.status_code}): {response.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("FastAPI Backend Offline. Ensure you run: `uvicorn api.main:app --reload` from your terminal.")

    with t3col2:
        st.markdown("**Real-Time Audio Pipeline (Hesitation, Pauses, SNR)**")
        audio_file = st.file_uploader("Upload Call Audio for Live Tracking", type=["wav", "mp3"])
        
        if audio_file:
            st.audio(audio_file)
            if st.button("Start Live Audio Analysis 🎙️"):
                import tempfile
                import os
                import librosa
                
                with st.spinner("Loading audio matrix..."):
                    # Process stream chunks locally and hit endpoint
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                        f.write(audio_file.getvalue())
                        temp_audio = f.name
                    
                    try:
                        # Load whole audio to simulate real-time streaming
                        waveform, sr = librosa.load(temp_audio, sr=22050)
                        duration = len(waveform) / sr
                        
                        st.markdown(f"**Audio Loaded:** `{duration:.1f} seconds` @ `22.05kHz`")
                        
                        chart_placeholder = st.empty()
                        metrics_placeholder = st.empty()
                        
                        live_stats = { "time": [], "merged_conf": [], "snr": [], "hesitations": [] }
                        
                        # Use 3-second chunks for realistic pause detection live
                        chunk_size = sr * 3 # 3 seconds
                        total_chunks = int(len(waveform) / chunk_size)
                        
                        prog_bar = st.progress(0.0)
                        
                        for i in range(total_chunks):
                            prog_bar.progress((i + 1) / total_chunks, text=f"Processing Block {i+1} / {total_chunks}")
                            
                            chunk_wav = waveform[i*chunk_size : (i+1)*chunk_size]
                            
                            # Export chunk securely to memory buffer (prevents Windows file lock issues)
                            import soundfile as sf
                            import io
                            
                            buf = io.BytesIO()
                            sf.write(buf, chunk_wav, sr, format='WAV')
                            buf.seek(0)
                                
                            files = {"file": ("chunk.wav", buf, "audio/wav")}
                            data = {"transcript_text": ""} # Empty to force pure live acoustic variability
                            
                            try:
                                res = requests.post("http://127.0.0.1:8000/analyze/audio", files=files, data=data)
                                if res.status_code == 200:
                                        chunk_data = res.json()
                                        
                                        live_stats["time"].append((i+1)*3)
                                        live_stats["merged_conf"].append(chunk_data.get("merged_confidence", 0.0))
                                        live_stats["snr"].append(chunk_data.get("snr_db", 0.0))
                                        live_stats["hesitations"].append(chunk_data.get("hesitation_pauses_count", 0))
                                        
                                        # Render real-time charts using native Streamlit line chart to prevent flickering
                                        live_df = pd.DataFrame(live_stats).set_index("time")
                                        
                                        # Scale SNR down so it visually fits on the same 0-1 scale as Confidence for the chart
                                        render_df = pd.DataFrame({
                                            "Overall Confidence": live_df["merged_conf"],
                                            "Acoustic Clarity (SNR scaled)": live_df["snr"] / 30.0  
                                        })
                                        
                                        chart_placeholder.line_chart(render_df, height=250, use_container_width=True)
                                        
                                        # Aggregate the strict total number of hesitations logged physically by the backend
                                        total_hesitation_penalties = sum(live_stats["hesitations"])
                                        
                                        m_col1, m_col2, m_col3 = metrics_placeholder.columns(3)
                                        m_col1.metric("Live SNR", f"{chunk_data.get('snr_db', 0):.1f} dB")
                                        m_col2.metric("Total Hesitations Detected", f"{total_hesitation_penalties}")
                                        m_col3.metric("Live Overall Confidence", f"{(chunk_data.get('merged_confidence', 0)*100):.1f}%")
                                        
                            except Exception as e:
                                st.error(f"Live buffer disconnect at block {i+1}: {e}")
                                break
                                
                            time.sleep(1.0) # simulate real-time processing duration
                    except Exception as e:
                        st.error(f"Audio processing error: {e}")
                    finally:
                        if os.path.exists(temp_audio):
                            os.remove(temp_audio)
