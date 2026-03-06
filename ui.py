# Run This Cmd ON in terminal: uv run python -m streamlit run "CHAT MODELS/ui.py"

import streamlit as st
import sys
import os
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# Load env & setup
load_dotenv()

# Get absolute path for the logo to prevent "File Not Found" errors
current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, "mayank-labs.png")

# Load the .env file if it exists (Local)
if os.path.exists(".env"):
    load_dotenv()

# Get the key (Works for both Local .env and Streamlit Cloud Secrets)
api_key = os.getenv("MISTRAL_API_KEY")

if not api_key:
    st.error("MISTRAL_API_KEY not found! Set it in your .env or Streamlit Secrets.")
    st.stop()

# --- THEMES & STYLING ---
def local_css():
    st.markdown("""
    <style>
        /* Main Background */
        .stApp {
            background-color: #0e1117;
            color: #ffffff;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(#1e1e2f, #0e1117);
            border-right: 1px solid #4a4a6a;
        }
                /* This targets all images in the sidebar */
        [data-testid="stSidebar"] .stImage img {
            border-radius: 50px;
            border: 2px solid #8b5cf6;
            transition: transform 0.3s ease;
        }
        [data-testid="stSidebar"] .stImage img:hover {
            transform: scale(1.05);
        }

        /* Chat Bubbles - User */
        div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from user"]) {
            background-color: #1e293b;
            border: 1px solid #3b82f6;
            border-radius: 20px 20px 0px 20px;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
        }

        /* Chat Bubbles - Assistant */
        div[data-testid="stChatMessage"]:has(div[aria-label="Chat message from assistant"]) {
            background-color: #0f172a;
            border: 1px solid #8b5cf6;
            border-radius: 20px 20px 20px 0px;
            box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
        }

        /* Title Gradient */
        .main-title {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3.5rem;
            font-weight: 800;
            text-shadow: 2px 2px 10px rgba(0,0,0,0.5);
        }

        /* Input Box Styling */
        .stChatInputContainer {
            border-radius: 15px;
            border: 1px solid #4a4a6a !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- RESILIENT IMPORTS ---
try:
    from langchain_core.utils.utils import init_chat_model
except ImportError:
    from langchain.chat_models import init_chat_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Mayank Labs | AI Mood", page_icon="⚡", layout="wide")
local_css()

# --- SIDEBAR UI ---
with st.sidebar:
    st.image(logo_path, width=80)
    st.markdown("### 🛠️ Lab Settings")
    
    mode_choice = st.radio(
        "AI Personality Matrix:",
        ["😡 Aggressive", "😂 Comedic", "😢 Melancholic"],
        help="Select the emotional core of the AI."
    )
    
    st.divider()
    
    # Environment Status Monitor
    st.markdown("#### 🛰️ System Status")
    is_venv = ".venv" in sys.executable
    status_color = "green" if is_venv else "red"
    st.markdown(f"**Environment:** :{status_color}[{'Virtual Env' if is_venv else 'Global System'}]")
    st.markdown(f"**Python:** `{sys.version[:5]}`")
    
    if st.button("🔄 Reset Neural Link", use_container_width=True):
        st.session_state.messages = [] # Reset will trigger re-init below
        st.rerun()

# --- LOGIC & MODEL ---
mode_prompts = {
    "😡 Aggressive": "You are a highly aggressive, impatient, and blunt AI. You use caps for emphasis and don't like small talk.",
    "😂 Comedic": "You are a stand-up comedian trapped in an AI. Use puns, sarcasm, and emoji heavily.",
    "😢 Melancholic": "You are a deeply sad, poetic, and emotional AI. You sigh often and find the tragedy in everything."
}
current_prompt = mode_prompts[mode_choice]

@st.cache_resource
def load_llm():
    return init_chat_model(
        model="mistral-small-latest", 
        model_provider="mistralai",
        temperature=0.8
    )

model = load_llm()

# --- SESSION HANDLING ---
if "messages" not in st.session_state or st.session_state.get("active_mode") != mode_choice:
    st.session_state.active_mode = mode_choice
    st.session_state.messages = [SystemMessage(content=current_prompt)]

# --- MAIN UI ---
st.markdown('<p class="main-title">Mayank Labs MoodBot</p>', unsafe_allow_html=True)
st.caption(f"Currently operating in **{mode_choice}** mode.")

# Centered Chat Container
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg.content)

# Input area
if user_input := st.chat_input("Enter your message..."):
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Render user message immediately
    with chat_container:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)

    # Generate Response
    with chat_container:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Processing Synapses..."):
                try:
                    response = model.invoke(st.session_state.messages)
                    st.markdown(response.content)
                    st.session_state.messages.append(AIMessage(content=response.content))
                except Exception as e:
                    st.error(f"Sync Error: {e}")