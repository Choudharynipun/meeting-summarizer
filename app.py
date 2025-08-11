import streamlit as st
from openai import OpenAI # type: ignore
import io
import os
from datetime import datetime
import re
import tempfile
import json
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env


# Page configuration
st.set_page_config(
    page_title="Meeting Audio Summarizer",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .summary-card {
        border-left-color: #4CAF50;
    }
    
    .keypoints-card {
        border-left-color: #FF9800;
    }
    
    .insights-card {
        border-left-color: #9C27B0;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-around;
        margin-top: 1rem;
    }
    
    .metric-box {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        flex: 1;
        margin: 0 0.5rem;
    }
    
    .metric-number {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        color: #666;
        margin-top: 0.5rem;
    }
    
    .processing-spinner {
        text-align: center;
        padding: 2rem;
    }
    
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .section-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .bullet-point {
        margin: 0.5rem 0;
        padding-left: 1rem;
    }
    
    @media (max-width: 768px) {
        .metric-container {
            flex-direction: column;
        }
        .metric-box {
            margin: 0.5rem 0;
        }
        .app-title {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

def initialize_openai():
    """Initialize OpenAI client with API key from env file or hardcoded value"""
    
    api_key = None

    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.strip().startswith("OPENAI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1]
                    break

    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    
    # Safety check
    if not api_key or api_key.startswith("sk-your-hardcoded"):
        st.error("⚠️ OpenAI API key not found. Please set it in a .env file or hardcode it.")
        st.stop()

    OpenAI.api_key = api_key


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI Whisper"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Transcribe using OpenAI Whisper
        with open(tmp_file_path, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return transcript.text  # Access text directly
    
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None
    
def analyze_speakers(transcript):
    """Estimate number of speakers using simple heuristics"""
    # Look for speaker indicators and conversation patterns
    speaker_patterns = [
        r'\b(I think|I believe|In my opinion)\b',
        r'\b(You mentioned|You said|You think)\b',
        r'\b(We should|We need|We can)\b',
        r'\b(Let\'s|Let us)\b',
        r'\?.*\b(Yes|No|Right|Correct|Exactly)\b'
    ]
    
    indicators = 0
    for pattern in speaker_patterns:
        indicators += len(re.findall(pattern, transcript, re.IGNORECASE))
    
    # Estimate speakers based on conversation indicators
    if indicators < 5:
        return 1
    elif indicators < 15:
        return 2
    elif indicators < 30:
        return 3
    else:
        return 4

def generate_summary_and_insights(transcript):
    """Generate meeting summary, key points, and topic using GPT"""
    try:
        prompt = f"""
        Please analyze this meeting transcript and provide:
        1. A concise summary (2-3 sentences)
        2. Key bullet points (3-5 main points)
        3. The main topic/theme of the meeting
        
        Format your response as JSON:
        {{
            "summary": "Your summary here",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "main_topic": "Main topic here"
        }}
        
        Transcript:
        {transcript}
        """
        
        response =client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes meeting transcripts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return {
            "summary": "Unable to generate summary",
            "key_points": ["Unable to extract key points"],
            "main_topic": "Unable to determine topic"
        }

def main():
    # Initialize OpenAI
    initialize_openai()
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1 class="app-title">🎙️ Meeting Audio Summarizer</h1>
        <p class="app-subtitle">Upload your meeting audio and get AI-powered summaries, key points, and insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("""
    <div class="upload-section">
        <h2>📁 Upload Your Meeting Audio</h2>
        <p>Supported formats: MP3, WAV, M4A • Max file size: 25MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a'],
        help="Upload your meeting recording to get started"
    )
    
    if uploaded_file is not None:
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📄 **File:** {uploaded_file.name}")
        with col2:
            st.info(f"📊 **Size:** {uploaded_file.size / 1024 / 1024:.1f} MB")
        with col3:
            st.info(f"🕒 **Uploaded:** {datetime.now().strftime('%H:%M:%S')}")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            
            # Processing indicator
            with st.spinner("🔄 Processing your audio file..."):
                progress_bar = st.progress(0)
                
                # Step 1: Transcription
                st.write("🎯 **Step 1:** Transcribing audio...")
                progress_bar.progress(25)
                transcript = transcribe_audio(uploaded_file)
                
                if transcript:
                    # Step 2: Speaker Analysis
                    st.write("👥 **Step 2:** Analyzing speakers...")
                    progress_bar.progress(50)
                    speaker_count = analyze_speakers(transcript)
                    
                    # Step 3: Generate insights
                    st.write("🧠 **Step 3:** Generating insights...")
                    progress_bar.progress(75)
                    analysis = generate_summary_and_insights(transcript)
                    
                    # Step 4: Complete
                    st.write("✅ **Step 4:** Finalizing results...")
                    progress_bar.progress(100)
                    
                    st.success("🎉 Processing completed successfully!")
                    
                    # Results Section
                    st.markdown("---")
                    st.markdown("## 📋 Meeting Analysis Results")
                    
                    # Summary Card
                    st.markdown(f"""
                    <div class="result-card summary-card">
                        <div class="section-header">
                            <span class="section-icon">📝</span>
                            <h3>Meeting Summary</h3>
                        </div>
                        <p style="color: black;">{analysis['summary']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key Points Card
                    key_points_html = ""
                    for point in analysis['key_points']:
                        key_points_html += f'<div class="bullet-point" style="color: black;">• {point}</div>'
                    
                    st.markdown(f"""
                    <div class="result-card keypoints-card">
                        <div class="section-header">
                            <span class="section-icon">🎯</span>
                            <h3>Key Points</h3>
                        </div>
                        {key_points_html}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Insights Card
                    st.markdown(f"""
                    <div class="result-card insights-card" style="color: black;">
                        <div class="section-header">
                            <span class="section-icon">🔍</span>
                            <h3>Meeting Insights</h3>
                        </div>
                        <div class="metric-container" style="color: black;">
                            <div class="metric-box">
                                <div class="metric-number">{speaker_count}</div>
                                <div class="metric-label">Speakers Detected</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-number">{len(transcript.split())}</div>
                                <div class="metric-label">Words Transcribed</div>
                            </div>
                        </div>
                        <div style="margin-top: 1rem;">
                            <strong>Main Topic:</strong> {analysis['main_topic']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Transcript Expander
                    with st.expander("📄 View Full Transcript"):
                        st.text_area("Transcript", transcript, height=200, disabled=True)
                
                else:
                    st.error("❌ Failed to process audio file. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🎙️ Meeting Audio Summarizer • Powered by OpenAI Whisper & GPT</p>
        <p style="font-size: 0.9rem;">Contact- Nipunchoudhary44@gmail.com</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()