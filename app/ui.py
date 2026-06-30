import streamlit as st
import time
from agent import HybridAgent
st.set_page_config(
    page_title="Agentic RAG System ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.title("System Architecture")
    st.markdown("---")
    st.markdown("### Developer Info")
    st.markdown("**Name:** Subhan Shahid\n\n**ID:** 23F-6109\n\n**Section:** 6A")
    st.markdown("---")
    st.markdown("### Core Components")
    st.markdown("- **Generative LLM:** Fine-Tuned DialoGPT")
    st.markdown("- **Vector Database:** FAISS (L2 Distance)")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **External Tools:** Python Runtime APIs")
    st.markdown("---")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
st.title("Hybrid Agentic Dialogue System")
st.caption("Combining Retrieval-Augmented Generation (RAG) with Lightweight Tool Integration")
st.markdown("---")
@st.cache_resource
def load_agent():
    return HybridAgent()
try:
    agent = load_agent()
except Exception as e:
    st.error(f"Failed to load the AI Agent. Please check your model paths.\nError: {e}")
    st.stop()
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to the Agentic AI Interface. I am equipped with a RAG memory and external computational tools. How may I assist you today?"}
    ]
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Type your message... (Try asking for the time, a math problem, or a fact)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and generating response..."):
            try:
                time.sleep(0.5) 
                response = agent.get_response(prompt)       
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")