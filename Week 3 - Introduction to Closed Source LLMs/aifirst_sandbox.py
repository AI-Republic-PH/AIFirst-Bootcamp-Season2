import streamlit as st
from openai import OpenAI

# Page config
st.set_page_config(page_title="AIFirst Sandbox", page_icon="🤖", layout="wide")

default_system_prompt = """
You are an assistant that performs a specific task. Follow the user's input instructions accurately, be concise, and focus on the intended goal of the task.
"""

# State init
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "temp_api_key" not in st.session_state:
    st.session_state.temp_api_key = ""
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = default_system_prompt
if "temp_prompt" not in st.session_state:
    st.session_state.temp_prompt = default_system_prompt
if "api_status" not in st.session_state:
    st.session_state.api_status = None
if "prompt_loaded" not in st.session_state:
    st.session_state.prompt_loaded = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# Validate API key
def validate_api_key():
    try:
        client = OpenAI(api_key=st.session_state.temp_api_key)
        client.models.list()
        st.session_state.api_key = st.session_state.temp_api_key
        st.session_state.api_status = "valid"
    except Exception:
        st.session_state.api_status = "invalid"

# Load prompt
def load_prompt():
    st.session_state.system_prompt = st.session_state.temp_prompt
    st.session_state.prompt_loaded = True

# Sidebar UI
with st.sidebar:
    st.title("🤖 AIFirst Sandbox")
    
    st.text_input("🔑 OpenAI API Key", type="password", key="temp_api_key")
    if st.button("Enter API Key"):
        validate_api_key()

    if st.session_state.api_status == "valid":
        st.success("API key is valid! ✅")
    elif st.session_state.api_status == "invalid":
        st.error("Invalid API key ❌")

    st.text_area("📝 System Prompt", key="temp_prompt", height=200)
    if st.button("Enter Prompt"):
        load_prompt()

    if st.session_state.prompt_loaded:
        st.info("Prompt loaded successfully! ✨")

    st.caption("Customize your assistant prompt above and start testing!")

# Main layout
st.title("🚀Playground")
st.markdown("Paste your input below and see how your assistant responds based on your custom prompt.")

st.text_area(
    "💬 Your Input", 
    key="user_input", 
    height=200, 
    placeholder="Paste your content or task..."
)

if st.button("Run AI", use_container_width=True, disabled=not st.session_state.api_key or not st.session_state.user_input.strip()):
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        with st.spinner("🤔 Thinking..."):
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": st.session_state.system_prompt},
                    {"role": "user", "content": st.session_state.user_input}
                ],
                temperature=0.7,
            )
            result = response.choices[0].message.content
            st.subheader("🧾 Output")
            st.write(result)

            st.markdown("""
                <button
                    onclick="navigator.clipboard.writeText(document.querySelector('.stMarkdown p').innerText); this.innerText='Copied!'; setTimeout(() => this.innerText='Copy to clipboard', 2000)"
                    style="background-color:#4CAF50;color:white;border:none;padding:6px 12px;border-radius:5px;cursor:pointer;">
                    Copy to clipboard
                </button>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
