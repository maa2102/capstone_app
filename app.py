import streamlit as st
import requests
import json
import logging
import numpy as np
from typing import Optional
from PIL import Image
import tensorflow as tf


# Constant
BASE_API_URL = "https://cbda-175-139-159-165.ngrok-free.app"
FLOW_ID = "3e733bf0-7649-4b45-af59-989397f3e2e6"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings
TWEAKS = {
  "ChatInput-feB0F": {},
  "ChatOutput-vikhM": {},
  "GoogleGenerativeAIModel-imboU": {},
  "Prompt-upBoA": {},
  "Memory-1YMJe": {},
  "TextInput-KlD03": {},
  "TextInput-ZmmJM": {}
}

# Function to run the flow
# Initialize logging
logging.basicConfig(level=logging.INFO)


# Function to run the flow
def run_flow(message: str,
             endpoint: str = FLOW_ID,
             output_type: str = "chat",
             input_type: str = "chat",
             tweaks: Optional[dict] = None,
             api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }

    if tweaks:
        payload["tweaks"] = tweaks

    headers = {"x-api-key": api_key} if api_key else None
    response = requests.post(api_url, json=payload, headers=headers)

    # Log the response for debugging
    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}


# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."
    
@st.cache_resource  # Cache the model to avoid reloading on each run
def load_model():
    model = tf.keras.models.load_model('model.keras')
    return model

model = load_model()

# Function to run the flow
def main():
    st.title(" KitarBot 🤖♻️")
    st.write("### 🤔 Unsure about recycling your items? Just text me your questions or upload a picture of your item here!")
    st.markdown('''📢 While KitarBot strives to provide accurate and useful information, it may not always be accurate or completely up to date. 
                Your patience and understanding are greatly appreciated as the KitarBot does its best to assist you.''')
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar for file uploader
    with st.sidebar:
        st.header("Upload your image here!")
        st.markdown("For the best and most accurate results, please make sure the image is clear and contains only one type of item at a time. Thanks! 😊")
        image = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if image is not None:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            image = Image.open(image)
            image = image.resize((160, 160), Image.Resampling.LANCZOS) # or other resampling methods
            image = np.array(image)
            image = np.expand_dims(image, axis=0)

            # Predict image class
            predict = np.argmax(model.predict(image))
            class_names = ['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper', 'plastic', 'shoes', 'trash']
            predicted_class = class_names[predict]
            #st.sidebar.success(f"**Predicted Class:** {predicted_class}")

            # ✅ Check if the class info was already added to avoid duplicates
            if not any(msg["content"] == f"{predicted_class}?" for msg in st.session_state.messages):
                # Save predicted class as a user message
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"{predicted_class}?",
                    "avatar": "💬"
                })

                # Get assistant response and save it
                assistant_response = extract_message(run_flow(f"{predicted_class}?", tweaks=TWEAKS))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response,
                    "avatar": "🤖"
                })

    # Display previous messages with avatars
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message["avatar"]):
            st.write(message["content"])

    # Input box for user message
    if query := st.chat_input("Ask me anything..."):
        # Add user message to session state
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query,
                "avatar": "💬",  # Emoji for user
            }
        )
        with st.chat_message("user", avatar="💬"):  # Display user message
            st.write(query)

         # Call the Langflow API and get the assistant's response
        with st.chat_message("assistant", avatar="🤖"):  # Emoji for assistant
            message_placeholder = st.empty()  # Placeholder for assistant response
            with st.spinner("Thinking..."):
                # Fetch response from Langflow with updated TWEAKS and using `query`
                assistant_response = extract_message(run_flow(query, tweaks=TWEAKS))
                message_placeholder.write(assistant_response)

        # Add assistant response to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "avatar": "🤖",  # Emoji for assistant
            }
        )

if __name__ == "__main__":
    main()