import os
import streamlit as st

st.set_page_config(
    page_title="Job Search Copilot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load API key from environment or sidebar
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input(
        "Enter your Anthropic API Key",
        type="password",
        help="Get your API key from https://console.anthropic.com"
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = bool(api_key)

# Main content
st.title("🔍 Job Search Copilot")
st.markdown("""
Welcome to your AI-powered job search assistant. This tool helps you:
- **Fetch & Analyze** job descriptions
- **Research** companies and roles
- **Prepare** for interviews
- **Compare** opportunities
- **Track** your job search progress

Use the navigation menu on the left to get started.
""")

if st.session_state.authenticated:
    st.success("✓ API Key loaded successfully")
else:
    st.warning("⚠️ Please enter your API key in the sidebar to continue")

st.markdown("---")
st.markdown("**Pages:**")
st.markdown("""
1. **Add Job** - Add and fetch new job postings
2. **Add Interviewer** - Set up interview preparation
3. **Compare Opportunities** - Analyze multiple jobs side-by-side
4. **Dashboard** - View your job search insights
""")
