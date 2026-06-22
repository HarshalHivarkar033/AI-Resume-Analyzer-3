import streamlit as st   
import plotly.graph_objects as go  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re       
from collections import Counter    
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk import pos_tag 
from google import genai  # Official Google GenAI SDK

# Download NLTK resources safely
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)

download_nltk_resources()

# Page Setup
st.set_page_config(page_title="ATS Resume Maximizer Pro", page_icon="📄", layout="wide")

# Custom CSS for Background Image and Glassmorphism Styling
st.markdown("""
    <style>
    /* Targets the main application background with highly visible image */
    .stApp {
        background-image: linear-gradient(rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.15)), 
                          url("https://images.unsplash.com/photo-1497366216548-37526070297c?q=80&w=1920");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Makes content containers stand out beautifully against a visible background */
    .stTabs, .stTextArea, .stFileUploader, [data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.85);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .main-title { font-size: 2.5rem; font-weight: bold; color: #1E3A8A; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #1F2937; margin-bottom: 2rem; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Premium ATS Resume Matcher & Optimizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Transform your application with mathematical keyword optimization and AI tailoring tools.</div>', unsafe_allow_html=True)

# Sidebar setup for API Configuration & About
with st.sidebar:
    st.header("🔑 Configuration")
    api_key = st.text_input("Enter Gemini API Key", type="password", help="Get a free key from Google AI Studio")
    st.info("💡 Pro-Tip: Enter your API key to unlock the AI-powered 'Fix-It' engine features!")
    
    st.header("🛠️ Premium Engine Active")
    st.write("- **Visible Background Interface**")
    st.write("- **Plotly Skill Categorization**")
    st.write("- **Gemini Bullet Optimizer**")
    st.write("- **AI Cover Letter Writer**")

# --- Helper Functions ---

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return " ".join([word for word in words if word not in stop_words])

def calculate_similarity(resume_text, job_description):
    resume_processed = remove_stopwords(clean_text(resume_text))
    job_processed = remove_stopwords(clean_text(job_description))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_processed, job_processed])
    score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    return round(score, 2), resume_processed, job_processed

def extract_keywords(text, num_keywords=20):
    words = word_tokenize(text)
    words = [w for w in words if len(w) > 2]
    tagged_words = pos_tag(words)
    keywords = [w for w, pos in tagged_words if pos.startswith('NN') or pos.startswith('JJ')]
    word_freq = Counter(keywords)
    return [word for word, freq in word_freq.most_common(num_keywords)]

def categorize_skills(keywords):
    """Categorizes keywords into broader skill categories for dynamic visual reporting."""
    categories = {"Technical & Core Skills": 0, "Tools & Frameworks": 0, "Professional Soft Skills": 0}
    
    # Simple rule-based keyword grouping dictionary
    tech_triggers = ['python', 'sql', 'regression', 'machine', 'learning', 'data', 'analysis', 'statistics', 'modeling', 'predictive']
    tool_triggers = ['pandas', 'numpy', 'plotly', 'matplotlib', 'scikit-learn', 'tableau', 'powerbi', 'git', 'dashboard']
    soft_triggers = ['communication', 'teamwork', 'leadership', 'problem-solving', 'analytical', 'management', 'presentation']
    
    for kw in keywords:
        if kw in tech_triggers:
            categories["Technical & Core Skills"] += 1
        elif kw in tool_triggers:
            categories["Tools & Frameworks"] += 1
        elif kw in soft_triggers or len(kw) > 8: # fallback heuristic for descriptive traits
            categories["Professional Soft Skills"] += 1
        else:
            categories["Technical & Core Skills"] += 1
            
    return categories

# --- Main Application Logic ---

def main():
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("📤 Step 1: Upload Resume")
        uploaded_file = st.file_uploader("Choose your resume file (PDF Only)", type=['pdf'])
        
    with col2:
        st.subheader("📋 Step 2: Target Job Description")
        job_description = st.text_area("Paste the job details here", height=150)

    st.markdown("---")
    
    if st.button("🚀 Analyze & Initialize Optimizer", type="primary", use_container_width=True):
        if not uploaded_file or not job_description:
            st.warning("⚠️ Please provide both a resume file and a job description.")
            return
        
        # Save state data so it persists when interacting with the premium tools
        st.session_state['resume_text'] = extract_text_from_pdf(uploaded_file)
        st.session_state['job_description'] = job_description
        
        # Run standard math matching pipeline
        score, r_proc, j_proc = calculate_similarity(st.session_state['resume_text'], st.session_state['job_description'])
        st.session_state['similarity_score'] = score
        
        # Extract Keyword gaps
        j_keywords = extract_keywords(j_proc)
        r_keywords = set(extract_keywords(r_proc, num_keywords=30))
        st.session_state['missing_keywords'] = [kw for kw in j_keywords if kw not in r_keywords]
        st.session_state['job_keywords'] = j_keywords
        st.session_state['analysis_done'] = True

    # Check if analysis has run successfully
    if st.session_state.get('analysis_done'):
        
        # --- TAB LAYOUT FOR CORE VS PREMIUM FEATURES ---
        tab1, tab2, tab3 = st.tabs(["📊 Analytics Dashboard", "🎯 AI Bullet Point Optimizer", "✉️ AI Cover Letter Generator"])
        
        with tab1:
            score_col, chart_col = st.columns([1, 1])
            
            with score_col:
                st.metric("ATS Match Score", f"{st.session_state['similarity_score']}%")
                if st.session_state['similarity_score'] < 50:
                    st.error("⚠️ Low alignment. Utilize the AI features in the next tabs to patch your skills.")
                else:
                    st.success("✨ Strong alignment! Ready for submission.")
                
                st.write("**Top Missing Key Phrases Found:**")
                st.write(", ".join([f"`{kw}`" for kw in st.session_state['missing_keywords'][:8]]))
                
            with chart_col:
                st.markdown("### Interactive Skill Category Alignment Matrix")
                cat_data = categorize_skills(st.session_state['job_keywords'])
                
                # Interactive Plotly horizontal bar chart
                fig = go.Figure(go.Bar(
                    x=list(cat_data.values()),
                    y=list(cat_data.keys()),
                    orientation='h',
                    marker_color=['#1E3A8A', '#3B82F6', '#60A5FA']
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("💡 Optimize Weak Bullet Points")
            st.write("Paste an existing experience bullet point from your resume. The AI engine will instantly reconstruct it to hit your missing metrics.")
            
            weak_bullet = st.text_input("Paste a weak experience line here:", placeholder="e.g., I worked on data processing and made a dashboard.")
            
            if st.button("✨ Optimize Bullet Point"):
                if not api_key:
                    st.error("🔒 Please input your Gemini API Key in the left sidebar to use this feature.")
                elif not weak_bullet:
                    st.warning("Please enter a bullet point to analyze.")
                else:
                    with st.spinner("Rewriting statement..."):
                        # Initialize official Client instance
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
                        You are an expert resume writer. Rewrite this weak bullet point: '{weak_bullet}'.
                        Make it highly professional, beginning with a strong industry action verb.
                        Strategically weave in some of these missing job keywords if contextually relevant: {st.session_state['missing_keywords'][:5]}.
                        Return only the optimized bullet point directly.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                        )
                        st.info(f"**Optimized Output:**\n\n {response.text}")

        with tab3:
            st.subheader("✉️ Custom Tailored Cover Letter Generator")
            st.write("Generate a high-conversion cover letter contextualized directly to the target employer's job description gaps.")
            
            if st.button("🤖 Draft Cover Letter with Gemini"):
                if not api_key:
                    st.error("🔒 Please input your Gemini API Key in the left sidebar to use this feature.")
                else:
                    with st.spinner("Drafting your letter..."):
                        client = genai.Client(api_key=api_key)
                        prompt = f"""
                        Generate a professional 3-paragraph cover letter based on this target position requirements:
                        '''{st.session_state['job_description'][:1500]}'''
                        Ensure it emphasizes these technical/soft competencies organically: {st.session_state['job_keywords'][:6]}.
                        Keep placeholder brackets like [Your Name] for contact sections. Clean text only.
                        """
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=prompt,
                        )
                        st.text_area("Generated Cover Letter", value=response.text, height=400)

if __name__ == "__main__":
    main()