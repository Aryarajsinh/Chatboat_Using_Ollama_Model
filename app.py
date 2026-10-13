import os
import json
import warnings
import urllib.request
import urllib.error
from dotenv import load_dotenv
import streamlit as st

# Suppress deprecation warnings
warnings.filterwarnings("ignore")

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# Page & Environment Configuration
# ---------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Ollama AI Playground",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LangSmith Tracing setup
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Ollama-Streamlit-App")

# ---------------------------------------------------------
# Premium Custom Styling (CSS)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global polish */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(168, 85, 247, 0.05) 0%, transparent 40%);
    }

    /* Hero header */
    .hero-container {
        padding: 1.2rem 1.5rem;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(12px);
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        padding: 0;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-online {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .badge-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-model {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Quick Prompt Cards */
    .starter-header {
        color: #cbd5e1;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Sidebar cards */
    .sidebar-section {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* Chat avatars & message containers */
    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 0.75rem;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        background-color: rgba(15, 23, 42, 0.35);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions: Ollama Detection
# ---------------------------------------------------------
@st.cache_data(ttl=10)
def check_ollama_status(base_url="http://localhost:11434"):
    """Check Ollama connectivity and fetch pulled local models."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", headers={"User-Agent": "Streamlit-Ollama-UI"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", []) if "embed" not in m.get("name", "").lower()]
                return True, models, "Connected"
    except Exception as e:
        pass
    return False, [], "Disconnected"


# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------
# Sidebar: Power Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ **Control Center**")
    
    # Ollama Host URL
    base_url = st.text_input("Ollama Server URL", value="http://localhost:11434", help="Default local Ollama port is 11434")
    
    # Check connection
    is_online, installed_models, status_text = check_ollama_status(base_url)
    
    if is_online:
        st.markdown(f'<span class="badge badge-online">● Ollama: {status_text}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge badge-offline">○ Ollama: {status_text}</span>', unsafe_allow_html=True)
        st.warning("Ollama is not running. Start it with `ollama serve` in your terminal.")

    st.markdown("---")
    
    # Model Selector
    st.markdown("#### 🧠 **Model Configuration**")
    model_options = installed_models if installed_models else ["gemma:2b", "llama3", "mistral", "phi3"]
    if "gemma:2b" not in model_options:
        model_options.insert(0, "gemma:2b")

    selected_model = st.selectbox(
        "Select Model",
        options=model_options,
        index=0 if "gemma:2b" in model_options else 0,
        help="Models pulled locally in Ollama"
    )

    custom_model = st.text_input("Or enter custom model tag", placeholder="e.g. deepseek-r1:1.5b")
    active_model = custom_model.strip() if custom_model.strip() else selected_model

    # Model Hyperparameters
    col_temp, col_top = st.columns(2)
    with col_temp:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05, help="0.0 = Deterministic/Focused, 1.0 = Creative")
    with col_top:
        st.metric(label="Model Type", value="Local GGUF")

    st.markdown("---")
    
    # System Persona Presets
    st.markdown("#### 🎭 **System Persona**")
    persona_options = {
        "Helpful Assistant": "You are a helpful, accurate, and friendly AI assistant. Answer queries clearly with structured markdown.",
        "Senior Code Expert": "You are a Principal Software Engineer. Provide optimal, clean, well-commented code snippets with concise architectural explanations.",
        "Data Science & ML Tutor": "You are an expert Data Science and Machine Learning instructor. Explain concepts intuitively with mathematical insights and practical Python examples.",
        "Concise Summarizer": "You are an executive summary specialist. Provide bullet points and key takeaways in the most concise format possible.",
        "Custom": ""
    }
    
    selected_persona_name = st.selectbox("Persona Preset", list(persona_options.keys()), index=0)
    if selected_persona_name == "Custom":
        system_instruction = st.text_area("Custom System Prompt", value="You are a helpful AI assistant.")
    else:
        system_instruction = persona_options[selected_persona_name]
        with st.expander("View Prompt", expanded=False):
            st.caption(system_instruction)

    st.markdown("---")
    
    # Chat management & Metrics
    st.markdown("#### 💬 **Chat Management**")
    num_messages = len(st.session_state.messages)
    col_count, col_clear = st.columns(2)
    with col_count:
        st.metric("Messages", f"{num_messages}")
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Chat Export
    if num_messages > 0:
        chat_transcript = "# Ollama Chat Export\n\n"
        for m in st.session_state.messages:
            role = "User" if m["role"] == "user" else f"Assistant ({active_model})"
            chat_transcript += f"### {role}\n{m['content']}\n\n"
        st.download_button(
            label="📥 Export Chat (.md)",
            data=chat_transcript,
            file_name="ollama_chat_history.md",
            mime="text/markdown",
            use_container_width=True
        )

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(f"""
<div class="hero-container">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
        <div>
            <h1 class="hero-title">⚡ Ollama AI Playground</h1>
            <p class="hero-subtitle">High-performance local LLM chat powered by LangChain & Streamlit</p>
        </div>
        <div>
            <span class="badge badge-model">🤖 {active_model}</span>
            <span class="badge {('badge-online' if is_online else 'badge-offline')}">
                {('🟢 Connected' if is_online else '🔴 Offline')}
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Empty State: Starter Prompt Suggestions
# ---------------------------------------------------------
starter_prompt = None
if len(st.session_state.messages) == 0:
    st.markdown('<p class="starter-header">💡 Try an instant prompt or ask anything below:</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Explain Transformers Architecture", use_container_width=True):
            starter_prompt = "Explain the Transformer neural network architecture in simple terms with key components."
    with c2:
        if st.button("🐍 Python Script for LangChain RAG", use_container_width=True):
            starter_prompt = "Write a complete Python code example demonstrating how to build a basic RAG pipeline with LangChain."
    with c3:
        if st.button("🧠 Machine Learning Interview Questions", use_container_width=True):
            starter_prompt = "Provide 5 essential Machine Learning interview questions on overfitting and regularisation with clear answers."

# ---------------------------------------------------------
# Display Chat Messages
# ---------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = "👤" if role == "user" else "🦙"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# LangChain Chain Builder
# ---------------------------------------------------------
def create_chain(model_name: str, sys_prompt: str, temp: float, host_url: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", sys_prompt),
        ("user", "Question: {question}")
    ])
    
    llm = Ollama(
        model=model_name,
        base_url=host_url,
        temperature=temp
    )
    
    output_parser = StrOutputParser()
    return prompt_template | llm | output_parser

# ---------------------------------------------------------
# Input Handling & Real-time Streaming
# ---------------------------------------------------------
user_input = st.chat_input("What question you have in mind? (e.g. Explain quantum computing...)")
prompt_to_run = starter_prompt or user_input

if prompt_to_run:
    # 1. Add and render User message
    st.session_state.messages.append({"role": "user", "content": prompt_to_run})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_run)

    # 2. Render and stream Assistant response
    with st.chat_message("assistant", avatar="🦙"):
        try:
            chain = create_chain(
                model_name=active_model,
                sys_prompt=system_instruction,
                temp=temperature,
                host_url=base_url
            )
            
            # Stream tokens live into the Streamlit container
            def response_generator():
                for chunk in chain.stream({"question": prompt_to_run}):
                    yield chunk

            response_text = st.write_stream(response_generator())
            
            # Save assistant response to state
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except urllib.error.URLError:
            err_msg = (
                f"❌ **Connection Error**: Could not reach Ollama at `{base_url}`.\n\n"
                "**Quick Fixes:**\n"
                "1. Make sure Ollama is running (`ollama serve` in terminal).\n"
                "2. Verify that the server URL in the sidebar matches your Ollama instance."
            )
            st.error(err_msg)
        except Exception as e:
            err_msg = (
                f"⚠️ **Execution Error**: `{str(e)}`\n\n"
                f"**Suggestion:** Verify that model `{active_model}` has been pulled via terminal: `ollama run {active_model}`."
            )
            st.error(err_msg)

#understnding purpose  this below code was logic 

# import os
# from dotenv import load_dotenv
# from langchain_community.llms import Ollama
# import streamlit as st
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# load_dotenv()
# ## Langsmith Tracking
# os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
# ## Prompt Template
# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system","You are a helpful assistant. Please respond to the question asked"),
#         ("user","Question:{question}")
#     ]
# )
# ## streamlit framework
# st.title("Langchain Demo With Gemma Model")
# input_text=st.text_input("What question you have in mind?")
# ## Ollama Llama2 model
# llm=Ollama(model="gemma:2b")
# output_parser=StrOutputParser()
# chain=prompt|llm|output_parser
# if input_text:
#     st.write(chain.invoke({"question":input_text}))

