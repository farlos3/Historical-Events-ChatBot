import streamlit as st
import os
import sys
from typing import Dict, List
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the project directory to the path to import backend
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from backend import create_historical_events_db, HistoricalEventsVectorDB

# Load configuration from environment variables
APP_TITLE = os.getenv("APP_TITLE", "Historical Events ChatBot")
PAGE_ICON = os.getenv("PAGE_ICON", "📚")
DATA_FILENAME = os.getenv("DATA_PATH", "descriptions.txt")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATA_PATH = os.path.join(os.path.dirname(__file__), DATA_FILENAME)

# Validate required environment variables
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables!")
    st.error("Please set your Groq API key in the .env file")
    st.stop()

@st.cache_resource
def initialize_database():
    """Initialize the vector database with caching"""
    with st.spinner("🔄 Initializing Historical Events Database..."):
        # Check if data file exists
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Data file not found: {DATA_PATH}")
            st.error("Please upload descriptions.txt file to the app directory")
            st.info("You can upload the file using the file uploader below:")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload your descriptions.txt file", 
                type=['txt'],
                help="Upload a text file with one historical event per line"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                with open(DATA_PATH, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                st.success("✅ File uploaded successfully! Reloading...")
                st.rerun()
            else:
                st.stop()
        
        # Show file info for debugging
        st.info(f"Loading data from: {DATA_PATH}")
        
        # Count lines for debugging
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            total_lines = len([line for line in f if line.strip()])
        st.info(f"Found {total_lines} events in file")
        
        db = create_historical_events_db(
            groq_api_key=GROQ_API_KEY,
            data_path=DATA_PATH,
            load_saved=False  # Force fresh load to avoid cache issues
        )
    return db

def display_stats(db: HistoricalEventsVectorDB):
    """Display database statistics in sidebar"""
    stats = db.get_stats()
    
    if stats["status"] == "ready":
        
        with st.sidebar.expander("Database Statistics", expanded=False):
            st.metric("Total Events", stats["total_events"])
            st.metric("Embedding Dimension", stats["embedding_dimension"])
            st.metric("Memory Usage (MB)", f"{stats['memory_usage_mb']:.1f}")
            st.metric("Avg Event Length", f"{stats['average_event_length']:.0f} chars")
            
            st.write("**Event Length Range:**")
            st.write(f"• Min: {stats['min_event_length']} chars")
            st.write(f"• Max: {stats['max_event_length']} chars")
    else:
        st.sidebar.error("❌ Database Not Ready")

def main():
    # Header
    st.title(f"{PAGE_ICON} {APP_TITLE}")
    st.markdown("---")
    
    # Initialize database
    try:
        db = initialize_database()
    except Exception as e:
        st.error(f"❌ Failed to initialize database: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Display database stats
        display_stats(db)
        
        # Search parameters
        st.subheader("⚙️ Chat Settings")
        chat_top_k = st.slider("Number of Sources (top_k)", min_value=3, max_value=10, value=DEFAULT_TOP_K)
        
        # Temperature setting
        temperature = st.slider("Response Creativity", min_value=0.0, max_value=1.0, value=TEMPERATURE, step=0.1)
        st.caption("Lower = More factual, Higher = More creative")
        
        # Model parameters
        with st.expander("Model Configuration", expanded=False):
            st.info(f"**Current Model:** {os.getenv('LLM_MODEL', 'Llama-3.3-70B-Versatile')}")
            st.info(f"**Temperature:** {temperature}")
            st.info(f"**Embedding Model:** {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    
    # Quick chat input section
    st.subheader("🚀 Quick Question")
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            quick_query = st.text_input(
                "Ask about historical events:",
                placeholder="e.g., What caused World War I? Tell me about the Russian Revolution...",
                key="quick_chat"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add space
            quick_ask = st.button("💬 Ask", use_container_width=True, key="quick_ask")
    
    # Process quick query
    if quick_ask and quick_query.strip():
        # Add user message to history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            
        st.session_state.chat_history.append({
            "role": "user",
            "content": quick_query,
            "timestamp": time.time()
        })
        
        with st.spinner("AI is thinking..."):
            start_time = time.time()
            response_data = db.chat(quick_query, top_k=chat_top_k)
            response_time = time.time() - start_time
        
        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_data['response'],
            "sources": response_data['relevant_events'],
            "sources_count": response_data['sources_count'],
            "response_time": response_time,
            "timestamp": time.time()
        })
        
        # Display response immediately
        st.markdown("### 🤖 AI Response:")
        st.markdown(response_data['response'])
        
        # Show sources
        if response_data['sources_count'] > 0:
            with st.expander(f"📚 Sources Used ({response_data['sources_count']} events)", expanded=False):
                for i, event in enumerate(response_data['relevant_events'], 1):
                    st.markdown(f"**Source {i}** (Similarity: {event['similarity']:.3f})")
                    st.text_area(
                        f"Content", 
                        value=event['content'],
                        height=80,
                        key=f"quick_source_{i}_{int(time.time())}"
                    )
        
        st.info(f"⏱️ Response generated in {response_time:.2f} seconds")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        chat_input = st.text_area(
            "Ask a detailed question about historical events:",
            placeholder="You can ask complex questions here...",
            height=100
        )
        submit_chat = st.form_submit_button("🚀 Send", use_container_width=True)
    
    if submit_chat and chat_input.strip():
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": chat_input,
            "timestamp": time.time()
        })
        
        # Generate AI response
        with st.spinner("🤔 AI is thinking..."):
            start_time = time.time()
            response_data = db.chat(chat_input, top_k=chat_top_k)
            response_time = time.time() - start_time
        
        # Add AI response to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response_data['response'],
            "sources": response_data['relevant_events'],
            "sources_count": response_data['sources_count'],
            "response_time": response_time,
            "timestamp": time.time()
        })
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Chat History")
        
        # Clear history button at top
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Clear History", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Display messages in chronological order (newest at bottom)
        for i, message in enumerate(st.session_state.chat_history[-10:]):  # Show last 10 messages
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show metadata and sources for AI responses
                    col_time, col_sources = st.columns([1, 1])
                    with col_time:
                        st.caption(f"⏱️ Response time: {message.get('response_time', 0):.2f}s")
                    with col_sources:
                        st.caption(f"📚 Sources used: {message.get('sources_count', 0)} events")
                    
                    # Show sources in expander
                    if message.get('sources_count', 0) > 0 and 'sources' in message:
                        with st.expander(f"📖 View {message['sources_count']} Sources", expanded=False):
                            for j, source in enumerate(message['sources'], 1):
                                st.markdown(f"**Source {j}** (Similarity: {source.get('similarity', 0):.3f})")
                                st.text_area(
                                    f"Event Content",
                                    value=source.get('content', '')[:200] + "..." if len(source.get('content', '')) > 200 else source.get('content', ''),
                                    height=60,
                                    key=f"history_source_{i}_{j}",
                                    disabled=True
                                )

if __name__ == "__main__":
    main()