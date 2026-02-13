"""
Crypto Screenshot Analyzer
Professional trading analysis from chart and orderbook screenshots
"""

import streamlit as st
from PIL import Image
import io
from datetime import datetime

# Import core modules
from core.ocr_engine import OCREngine
from core.indicator_parser import IndicatorParser
from core.orderbook_parser import OrderbookParser
from core.technical_engine import TechnicalEngine
from core.orderbook_engine import OrderbookEngine
from core.scoring_engine import ScoringEngine
from core.llm_engine import LLMEngine
from core.pdf_exporter import PDFExporter

# Page configuration
st.set_page_config(
    page_title="Crypto Screenshot Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Main app styling */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2d3436;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .score-card {
        background: linear-gradient(135deg, #0f3460 0%, #1a1a2e 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        border: 2px solid #e94560;
        box-shadow: 0 8px 16px rgba(233, 69, 96, 0.2);
    }
    
    .score-value {
        font-size: 72px;
        font-weight: bold;
        color: #e94560;
        text-shadow: 0 0 20px rgba(233, 69, 96, 0.5);
    }
    
    .score-label {
        font-size: 18px;
        color: #a0a0a0;
        margin-top: 10px;
    }
    
    .signal-buy {
        color: #00ff88;
        font-weight: bold;
    }
    
    .signal-sell {
        color: #ff4444;
        font-weight: bold;
    }
    
    .signal-neutral {
        color: #ffaa00;
        font-weight: bold;
    }
    
    .grade-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 16px;
    }
    
    .grade-a { background-color: #00ff88; color: #000; }
    .grade-b { background-color: #88ff00; color: #000; }
    .grade-c { background-color: #ffaa00; color: #000; }
    .grade-d { background-color: #ff8800; color: #000; }
    .grade-f { background-color: #ff4444; color: #fff; }
    
    /* Section headers */
    .section-header {
        font-size: 24px;
        font-weight: bold;
        color: #e94560;
        margin: 20px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #2d3436;
    }
    
    /* Metric styling */
    .metric-label {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #fff;
    }
    
    /* Progress bar custom colors */
    .stProgress > div > div > div > div {
        background-color: #e94560;
    }
    
    /* Error/Warning styling */
    .error-box {
        background-color: rgba(255, 68, 68, 0.1);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background-color: rgba(255, 170, 0, 0.1);
        border: 1px solid #ffaa00;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .success-box {
        background-color: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* AI Report styling */
    .ai-report {
        background-color: #1a1a2e;
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        border-left: 4px solid #e94560;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'technical_summary' not in st.session_state:
        st.session_state.technical_summary = None
    if 'orderbook_summary' not in st.session_state:
        st.session_state.orderbook_summary = None
    if 'final_score' not in st.session_state:
        st.session_state.final_score = None
    if 'ai_report' not in st.session_state:
        st.session_state.ai_report = None
    if 'chart_image' not in st.session_state:
        st.session_state.chart_image = None
    if 'orderbook_image' not in st.session_state:
        st.session_state.orderbook_image = None


def display_header():
    """Display the main header."""
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #e94560; font-size: 42px; margin-bottom: 10px;'>
            📊 Crypto Screenshot Analyzer
        </h1>
        <p style='color: #888; font-size: 18px;'>
            Professional trading analysis from chart and orderbook screenshots
        </p>
    </div>
    """, unsafe_allow_html=True)


def display_sidebar():
    """Display sidebar with settings and info."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Model selection
        st.selectbox(
            "LLM Model",
            options=["mistral", "llama2", "codellama", "mixtral"],
            key="llm_model",
            help="Select the Ollama model for AI analysis"
        )
        
        # Temperature
        st.slider(
            "AI Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1,
            key="llm_temperature",
            help="Lower = more focused, Higher = more creative"
        )
        
        st.markdown("---")
        
        # Score weights
        st.markdown("### ⚖️ Score Weights")
        tech_weight = st.slider(
            "Technical Analysis Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            key="tech_weight"
        )
        
        ob_weight = 1.0 - tech_weight
        st.info(f"Orderbook Weight: {ob_weight:.1f}")
        
        st.markdown("---")
        
        # Info
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Upload a **chart screenshot** with technical indicators
        2. Upload an **orderbook screenshot**
        3. Click **Analyze** to process
        4. Review the analysis results
        5. Generate AI report (optional)
        6. Export PDF report
        """)
        
        st.markdown("---")
        
        # Connection status
        st.markdown("### 🔌 LLM Status")
        llm = LLMEngine()
        if llm.check_connection():
            st.success("✅ Ollama Connected")
            models = llm.list_available_models()
            if models:
                st.info(f"Available: {', '.join(models[:3])}")
        else:
            st.warning("⚠️ Ollama Offline")


def display_upload_section():
    """Display the image upload section."""
    st.markdown("<div class='section-header'>📤 Upload Screenshots</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Chart Screenshot")
        chart_file = st.file_uploader(
            "Upload chart with indicators (RSI, MACD, EMA, etc.)",
            type=['png', 'jpg', 'jpeg'],
            key="chart_upload",
            help="Upload a screenshot of your trading chart"
        )
        
        if chart_file:
            image = Image.open(chart_file)
            st.image(image, caption="Chart Preview", use_container_width=True)
            st.session_state.chart_image = image
    
    with col2:
        st.markdown("#### 📊 Orderbook Screenshot")
        orderbook_file = st.file_uploader(
            "Upload orderbook screenshot",
            type=['png', 'jpg', 'jpeg'],
            key="orderbook_upload",
            help="Upload a screenshot of the order book"
        )
        
        if orderbook_file:
            image = Image.open(orderbook_file)
            st.image(image, caption="Orderbook Preview", use_container_width=True)
            st.session_state.orderbook_image = image
    
    return chart_file, orderbook_file


def analyze_images(chart_image: Image.Image=None, orderbook_image: Image.Image=None) -> tuple:
    """
    Perform analysis on uploaded images.
    
    Args:
        chart_image: PIL Image of chart
        orderbook_image: PIL Image of orderbook
        
    Returns:
        Tuple of (technical_summary, orderbook_summary, final_score)
    """
    # Initialize engines
    ocr = OCREngine()
    tech_engine = TechnicalEngine()
    ob_engine = OrderbookEngine()
    
    tech_weight = st.session_state.get('tech_weight', 0.6)
    scoring_engine = ScoringEngine(
        technical_weight=tech_weight,
        orderbook_weight=1.0 - tech_weight
    )
    
    # Process chart
    technical_summary = None
    technical_score = None

    if chart_image is not None:
        with st.spinner("🔍 Extracting chart data..."):
            try:
                chart_text = ocr.extract_chart_text(chart_image)           
                technical_score = tech_engine.analyze_from_text(chart_text)
                technical_summary = tech_engine.get_analysis_summary(technical_score)
            except Exception as e:
                st.error(f"Error processing chart: {str(e)}")
                technical_summary = {
                    'overall_score': 0,
                    'signal': 'Error',
                    'confidence_percent': 0,
                    'interpretation': f'Failed to process chart: {str(e)}',
                    'component_scores': {},
                    'valid_indicators': 0
                }
                technical_score = tech_engine.analyze_from_text("")
    else:
        technical_score = tech_engine.analyze_from_text("")
        technical_summary = tech_engine.get_analysis_summary(technical_score)
    
    # Process orderbook
    orderbook_summary = None
    orderbook_score = None

    if orderbook_image is not None:
        with st.spinner("🔍 Extracting orderbook data..."):
            try:
                orderbook_text = ocr.extract_orderbook_text(orderbook_image)
                orderbook_score = ob_engine.analyze_from_text(orderbook_text)
                orderbook_summary = ob_engine.get_analysis_summary(orderbook_score)
            except Exception as e:
                st.error(f"Error processing orderbook: {str(e)}")
                orderbook_summary = {
                    'overall_score': 0,
                    'signal': 'Error',
                    'confidence_percent': 0,
                    'interpretation': f'Failed to process orderbook: {str(e)}',
                    'metrics': {},
                    'component_scores': {}
                }
                orderbook_score = ob_engine.analyze_from_text("")
    else:
        orderbook_score = ob_engine.analyze_from_text("")
        orderbook_summary = ob_engine.get_analysis_summary(orderbook_score)
    
    # Calculate final score
    with st.spinner("📊 Calculating final score..."):
        final_score_obj = scoring_engine.calculate_final_score(technical_score, orderbook_score)
        final_score = scoring_engine.get_score_summary(final_score_obj)
        
    return technical_summary, orderbook_summary, final_score


def display_score_dashboard(final_score: dict):
    """Display the main score dashboard."""
    st.markdown("<div class='section-header'>📊 Score Dashboard</div>", unsafe_allow_html=True)
    
    # Main score card
    score_value = final_score.get('final_score', 0)
    signal = final_score.get('signal', 'N/A')
    risk_grade = final_score.get('risk_grade', 'N/A')
    confidence = final_score.get('confidence', 'N/A')
    
    # Determine colors
    if score_value >= 7:
        score_color = "#00ff88"
    elif score_value >= 5:
        score_color = "#ffaa00"
    else:
        score_color = "#ff4444"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class='score-card'>
            <div class='score-value' style='color: {score_color};'>{score_value}/10</div>
            <div class='score-label'>OVERALL SCORE</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <div class='metric-label'>SIGNAL</div>
            <div style='font-size: 20px; font-weight: bold; margin-top: 10px; color: {score_color};'>
                {signal}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <div class='metric-label'>CONFIDENCE</div>
            <div style='font-size: 20px; font-weight: bold; margin-top: 10px; color: #fff;'>
                {confidence}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Risk grade with color coding
        grade_class = "grade-c"
        if risk_grade.startswith('A'):
            grade_class = "grade-a"
        elif risk_grade.startswith('B'):
            grade_class = "grade-b"
        elif risk_grade == 'C':
            grade_class = "grade-c"
        elif risk_grade == 'D':
            grade_class = "grade-d"
        else:
            grade_class = "grade-f"
        
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <div class='metric-label'>RISK GRADE</div>
            <div style='margin-top: 10px;'>
                <span class='grade-badge {grade_class}'>{risk_grade}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card' style='text-align: center;'>
            <div class='metric-label'>SIGNALS</div>
            <div style='font-size: 14px; margin-top: 10px; color: {"#00ff88" if final_score.get("signals_aligned") else "#ffaa00"};'>
                {"✓ Aligned" if final_score.get("signals_aligned") else "⚠ Divergent"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendation
    recommendation = final_score.get('recommendation', '')
    if recommendation:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label' style='margin-bottom: 10px;'>RECOMMENDATION</div>
            <div style='color: #fff; white-space: pre-line;'>{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)


def display_technical_summary(technical_summary: dict):
    """Display technical analysis summary."""
    st.markdown("<div class='section-header'>📈 Technical Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score and signal
        score = technical_summary.get('overall_score', 0)
        signal = technical_summary.get('signal', 'N/A')
        confidence = technical_summary.get('confidence_percent', 0)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Technical Score</div>
            <div class='metric-value'>{score}/10</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Signal</div>
            <div class='metric-value'>{signal}</div>
            <div style='margin-top: 5px;'>
                <small style='color: #888;'>Confidence: {confidence}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Interpretation
        interpretation = technical_summary.get('interpretation', 'N/A')
        st.markdown(f"""
        <div class='metric-card' style='height: 100%;'>
            <div class='metric-label'>Analysis</div>
            <div style='color: #fff; margin-top: 10px;'>{interpretation}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Component scores
    st.markdown("#### Indicator Scores")
    component_scores = technical_summary.get('component_scores', {})
    
    if component_scores:
        cols = st.columns(len(component_scores))
        for i, (indicator, score) in enumerate(component_scores.items()):
            with cols[i]:
                score_val = score if score is not None else 0
                color = "#00ff88" if score_val > 0 else "#ff4444" if score_val < 0 else "#ffaa00"
                st.markdown(f"""
                <div class='metric-card' style='text-align: center;'>
                    <div class='metric-label'>{indicator}</div>
                    <div style='font-size: 24px; font-weight: bold; color: {color}; margin-top: 10px;'>
                        {score_val:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def display_orderbook_summary(orderbook_summary: dict):
    """Display orderbook analysis summary."""
    st.markdown("<div class='section-header'>📊 Orderbook Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score and signal
        score = orderbook_summary.get('overall_score', 0)
        signal = orderbook_summary.get('signal', 'N/A')
        confidence = orderbook_summary.get('confidence_percent', 0)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Orderbook Score</div>
            <div class='metric-value'>{score}/10</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Signal</div>
            <div class='metric-value'>{signal}</div>
            <div style='margin-top: 5px;'>
                <small style='color: #888;'>Confidence: {confidence}%</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Interpretation
        interpretation = orderbook_summary.get('interpretation', 'N/A')
        st.markdown(f"""
        <div class='metric-card' style='height: 100%;'>
            <div class='metric-label'>Analysis</div>
            <div style='color: #fff; margin-top: 10px;'>{interpretation}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown("#### Orderbook Metrics")
    metrics = orderbook_summary.get('metrics', {})
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bid_vol = metrics.get('total_bid_volume', 0)
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Total Bid Volume</div>
                <div style='font-size: 18px; font-weight: bold; color: #00ff88; margin-top: 10px;'>
                    {bid_vol:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            buy_walls = metrics.get('buy_walls_count', 0)
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Buy Walls</div>
                <div style='font-size: 18px; font-weight: bold; color: #00ff88; margin-top: 10px;'>
                    {buy_walls}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ask_vol = metrics.get('total_ask_volume', 0)
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Total Ask Volume</div>
                <div style='font-size: 18px; font-weight: bold; color: #ff4444; margin-top: 10px;'>
                    {ask_vol:,.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            sell_walls = metrics.get('sell_walls_count', 0)
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Sell Walls</div>
                <div style='font-size: 18px; font-weight: bold; color: #ff4444; margin-top: 10px;'>
                    {sell_walls}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            imbalance = metrics.get('imbalance_ratio', 0)
            imb_color = "#00ff88" if imbalance > 0 else "#ff4444" if imbalance < 0 else "#ffaa00"
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Imbalance Ratio</div>
                <div style='font-size: 18px; font-weight: bold; color: {imb_color}; margin-top: 10px;'>
                    {imbalance:.4f}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            spread = metrics.get('spread_percent', 0)
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;'>
                <div class='metric-label'>Spread %</div>
                <div style='font-size: 18px; font-weight: bold; color: #fff; margin-top: 10px;'>
                    {spread:.4f}%
                </div>
            </div>
            """, unsafe_allow_html=True)


def display_ai_analysis():
    """Display AI analysis section."""
    st.markdown("<div class='section-header'>🤖 AI Analysis</div>", unsafe_allow_html=True)
    
    if st.session_state.ai_report:
        st.markdown(f"""
        <div class='ai-report'>
            {st.session_state.ai_report.replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("🤖 Click 'Generate AI Report' to get an AI-powered trading analysis")
        
        with col2:
            if st.button("🧠 Generate AI Report", type="primary", use_container_width=True):
                generate_ai_report()


def generate_ai_report():
    """Generate AI report using LLM."""
    llm = LLMEngine(
        model=st.session_state.get('llm_model', 'mistral'),
        temperature=st.session_state.get('llm_temperature', 0.3)
    )
    
    if not llm.check_connection():
        st.error("❌ Cannot connect to Ollama. Make sure it's running on localhost:11434")
        return
    
    with st.spinner("🧠 Generating AI analysis... This may take a moment."):
        response = llm.generate_report(
            st.session_state.technical_summary,
            st.session_state.orderbook_summary,
            st.session_state.final_score
        )
    
    if response.success:
        st.session_state.ai_report = response.report
        st.rerun()
    else:
        st.error(f"❌ AI Report Generation Failed: {response.error}")


def display_export_section():
    """Display PDF export section."""
    st.markdown("<div class='section-header'>📥 Export Report</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📄 Export PDF Report", type="primary", use_container_width=True):
            export_pdf_report()


def export_pdf_report():
    """Generate and download PDF report."""
    with st.spinner("📄 Generating PDF report..."):
        try:
            exporter = PDFExporter()
            pdf_bytes = exporter.export(
                st.session_state.technical_summary,
                st.session_state.orderbook_summary,
                st.session_state.final_score,
                st.session_state.ai_report
            )
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_analysis_{timestamp}.pdf"
            
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True
            )
            
            st.success("✅ PDF report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate PDF: {str(e)}")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Display header
    display_header()
    
    # Display sidebar
    display_sidebar()
    
    # Main content
    if not st.session_state.analysis_complete:
        # Upload section
        chart_file, orderbook_file = display_upload_section()
        
        st.markdown("---")
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            analyze_disabled = not (chart_file or orderbook_file)
            
            if st.button(
                "🔍 Analyze Screenshots", 
                type="primary", 
                use_container_width=True,
                disabled=analyze_disabled
            ):
                chart_image = None
                orderbook_image = None
                if chart_file:
                    chart_image = Image.open(chart_file)
                if orderbook_file:
                    orderbook_image = Image.open(orderbook_file)
                    
                technical_summary, orderbook_summary, final_score = analyze_images(
                    chart_image, orderbook_image
                )
                
                st.session_state.technical_summary = technical_summary
                st.session_state.orderbook_summary = orderbook_summary
                st.session_state.final_score = final_score
                st.session_state.analysis_complete = True
                
                st.rerun()
            
            if analyze_disabled:
                st.warning("⚠️ Please upload at least one screenshot (chart or orderbook)")
    
    else:
        # Display results
        
        # Score Dashboard
        display_score_dashboard(st.session_state.final_score)
        
        st.markdown("---")
        
        # Technical and Orderbook summaries
        col1, col2 = st.columns(2)
        
        with col1:
            display_technical_summary(st.session_state.technical_summary)
        
        with col2:
            display_orderbook_summary(st.session_state.orderbook_summary)
        
        st.markdown("---")
        
        # AI Analysis
        display_ai_analysis()
        
        st.markdown("---")
        
        # Export section
        display_export_section()
        
        st.markdown("---")
        
        # Reset button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                # Reset session state
                st.session_state.analysis_complete = False
                st.session_state.technical_summary = None
                st.session_state.orderbook_summary = None
                st.session_state.final_score = None
                st.session_state.ai_report = None
                st.session_state.chart_image = None
                st.session_state.orderbook_image = None
                st.rerun()


if __name__ == "__main__":
    main()
