import streamlit as st
from openai import OpenAI
import base64
from PIL import Image
import io

# Page config
st.set_page_config(page_title="AIFirst Multimodal Sandbox", page_icon="🤖", layout="wide")

# Default prompt
default_system_prompt = """
You are an assistant that performs a specific task. Follow the user's input instructions accurately, be concise, and focus on the intended goal of the task.
"""

# State initialization
for key, default in {
    "api_key": "",
    "temp_api_key": "",
    "system_prompt": default_system_prompt,
    "temp_prompt": default_system_prompt,
    "api_status": None,
    "prompt_loaded": False,
    "user_input": "",
    "image_prompt": "",
    "image_result": None,
    "uploaded_image": None,
    "image_analysis_result": None,
    "audio_text": "",
    "tts_output": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# OpenAI Client
client = None

def validate_api_key():
    try:
        global client
        client = OpenAI(api_key=st.session_state.temp_api_key)
        client.models.list()
        st.session_state.api_key = st.session_state.temp_api_key
        st.session_state.api_status = "valid"
    except Exception:
        st.session_state.api_status = "invalid"

def load_prompt():
    st.session_state.system_prompt = st.session_state.temp_prompt
    st.session_state.prompt_loaded = True

# Sidebar UI
with st.sidebar:
    st.title("🤖 AIFirst Multimodal")

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

# Tabs for different modes
mode = st.selectbox("Choose a mode:", ["Text", "Image Generation", "Image Analysis", "Text-to-Speech"])

# Text Mode
if mode == "Text":
    st.header("🚀 Text Interaction")
    st.text_area("💬 Your Input", key="user_input", height=200, placeholder="Paste your content or task...")
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
                st.subheader("🧒 Output")
                st.write(result)
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Image Generation
elif mode == "Image Generation":
    st.header("🎨 DALL•E Image Generator")
    prompt = st.text_input("Enter prompt for image generation", key="image_prompt")
    if st.button("Generate Image", disabled=not st.session_state.api_key or not prompt.strip()):
        try:
            client = OpenAI(api_key=st.session_state.api_key)
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                n=1,
                size="1024x1024"
            )
            img_url = response.data[0].url
            st.image(img_url, caption="Generated Image")
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Image Analysis
elif mode == "Image Analysis":
    st.header("📷 Upload an Image for Analysis")
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Analyze Image", disabled=not st.session_state.api_key):
        try:
            img_bytes = uploaded_image.read()
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            response = OpenAI(api_key=st.session_state.api_key).chat.completions.create(
                model="gpt-4o",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see in this image?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                    ]
                }]
            )
            st.subheader("🫀 Analysis Result")
            st.write(response.choices[0].message.content.strip())
        except Exception as e:
            st.error(f"❌ Error: {e}")

# Text-to-Speech
elif mode == "Text-to-Speech":
    st.header("🎤 Text to Speech")
    tts_input = st.text_area("Enter text to synthesize into speech", key="tts_text")
    if st.button("Generate Speech", disabled=not st.session_state.api_key or not tts_input.strip()):
        try:
            response = OpenAI(api_key=st.session_state.api_key).audio.speech.create(
                model="tts-1",
                voice="nova",
                input=tts_input
            )
            audio_data = response.content
            st.audio(audio_data, format='audio/mp3')
        except Exception as e:
            st.error(f"❌ Error: {e}")