import json
import logging
import streamlit as st  # 1.34.0
import extra_streamlit_components as stx
import time
import tiktoken
import urllib
import sys
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from streamlit_chat import message
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.evaluation import RelevancyEvaluator
from datetime import datetime
from streamlit_float import *
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from functools import wraps
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger()
logging.basicConfig(encoding="UTF-8", level=logging.INFO)

st.set_page_config(page_title="Streamlit Chat Interface Improvement", page_icon="🤩")

st.title("🤩 Improved Streamlit Chat UI")

cookie_manager = stx.CookieManager(key="cookie_manager")

float_init()

# Secrets to be stored in /.streamlit/secrets.toml
# OPENAI_API_ENDPOINT = "https://xxx.openai.azure.com/"
# OPENAI_API_KEY = "efpgishhn2kwlnk9928avd6vrh28wkdj" (this is a fake key 😉)

# To be used with standard OpenAI API
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# To be used with standard Azure OpenAI API
# Setup llms and embedding models


sources = [
    "./FAQs_v2.csv",
    # "./chatai/data/ANZ-FAQ.pdf",
]


@st.cache_resource(show_spinner=False)
def setup():
    llm = Ollama(model="llama3-groq-tool-use", request_timeout=120.0)
    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
    )

    Settings.llm = llm
    Settings.embed_model = embed_model


setup()


@st.cache_resource(show_spinner=False)
def create_retriever():
    with st.spinner(text="Loading tools! This should take 1-2 minutes."):
        documents = SimpleDirectoryReader(input_files=sources).load_data()
    documents = Document(text="\n\n".join([doc.text for doc in documents]))
    index = VectorStoreIndex.from_documents([documents], show_progress=True)
    return index.as_query_engine(streaming=True, similarity_top_k=1)


query_engine = create_retriever()


# This function logs the last question and answer in the chat messages
def log_feedback(icon):
    # We display a nice toast
    st.toast("Thanks for your feedback!", icon="👌")

    # We retrieve the last question and answer
    last_messages = json.dumps(st.session_state["messages"][-2:])

    # We record the timestamp
    activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": "

    # And include the messages
    activity += "positive" if icon == "👍" else "negative"
    activity += ": " + last_messages

    # And log everything
    logger.info(activity)


@st.dialog("🎨 Upload a picture")
def upload_document():
    st.warning(
        "This is a demo dialog window. You need to process the file afterwards.",
        icon="💡",
    )
    picture = st.file_uploader(
        "Choose a file", type=["jpg", "png", "bmp"], label_visibility="hidden"
    )
    if picture:
        st.session_state["uploaded_pic"] = True
        st.rerun()


def retry(times=3, delay_seconds=1):
    """
    Retry decorator that retries the function `times` times with `delay_seconds` delay between retries.
    """

    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            attempts = times
            while attempts > 0:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {times - attempts + 1} failed: {e}")
                    if attempts > 1:
                        print(f"Retrying in {delay_seconds} second(s)...")
                        time.sleep(delay_seconds)
                attempts -= 1
            print(f"Function {func.__name__} failed after {times} attempts.")

        return wrapper_retry

    return decorator_retry


@retry(times=3, delay_seconds=1)
def chat(message, bot):
    return bot.query(message)


def get_conversation_title():
    return "Coach On Tap Chat Bot"


if "uploaded_pic" in st.session_state and st.session_state["uploaded_pic"]:
    st.toast("Picture uploaded!", icon="📥")
    del st.session_state["uploaded_pic"]

# Model Choice - Name to be adapted to your deployment
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-35-turbo"

# Adapted from https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps
if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_avatar = "👩‍💻"
assistant_avatar = "🤖"

# We rebuild the previous conversation stored in st.session_state["messages"] with the corresponding emojis
for message in st.session_state["messages"]:
    with st.chat_message(
        message["role"],
        avatar=assistant_avatar if message["role"] == "assistant" else user_avatar,
    ):
        st.markdown(message["content"])

# A chat input will add the corresponding prompt to the st.session_state["messages"]
if prompt := st.chat_input("How can I help you?"):

    st.session_state["messages"].append({"role": "user", "content": prompt})

    # and display it in the chat history
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)


# If the prompt is initialized or if the user is asking for a rerun, we
# launch the chat completion by the LLM
if prompt or ("rerun" in st.session_state and st.session_state["rerun"]):

    with st.chat_message("assistant", avatar=assistant_avatar):
        stream = chat(prompt, query_engine)
        if stream.response_gen:
            response = st.write_stream(stream.response_gen)

    st.session_state["messages"].append({"role": "assistant", "content": response})

    # In case this is a rerun, we set the "rerun" state back to False
    if "rerun" in st.session_state and st.session_state["rerun"]:
        st.session_state["rerun"] = False

st.write("")

# If there is at least one message in the chat, we display the options
if len(st.session_state["messages"]) > 0:

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = "history_" + datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

    action_buttons_container = st.container()
    action_buttons_container.float(
        "bottom: 7.2rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
    )

    # We set the space between the icons thanks to a share of 100
    cols_dimensions = [7, 14.9, 14.5, 9.1, 9, 8.6, 8.7]
    cols_dimensions.append(100 - sum(cols_dimensions))

    col0, col1, col2, col3, col4, col5, col6, col7 = action_buttons_container.columns(
        cols_dimensions
    )

    with col1:

        # Converts the list of messages into a JSON Bytes format
        json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        # And the corresponding Download button
        st.download_button(
            label="📥 Save!",
            data=json_messages,
            file_name="chat_conversation.json",
            mime="application/json",
        )

    with col2:

        # We set the message back to 0 and rerun the app
        # (this part could probably be improved with the cache option)
        if st.button("Clear 🧹"):
            st.session_state["messages"] = []
            del st.session_state["conversation_id"]

            if "uploaded_pic" in st.session_state:
                del st.session_state["uploaded_pic"]

            st.rerun()

    with col3:

        if st.button("🎨"):
            upload_document()

    with col4:
        icon = "🔁"
        if st.button(icon):
            st.session_state["rerun"] = True
            st.rerun()

    with col5:
        icon = "👍"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    with col6:
        icon = "👎"

        # The button will trigger the logging function
        if st.button(icon):
            log_feedback(icon)

    with col7:

        # We initiate a tokenizer
        enc = tiktoken.get_encoding("cl100k_base")

        # We encode the messages
        tokenized_full_text = enc.encode(
            " ".join([item["content"] for item in st.session_state["messages"]])
        )

        # And display the corresponding number of tokens
        label = f"💬 {len(tokenized_full_text)} tokens"
        st.link_button(label, "https://platform.openai.com/tokenizer")

else:

    # At the first run of a session, we temporarly display a message
    if "disclaimer" not in st.session_state:
        with st.empty():
            for seconds in range(3):
                st.warning(
                    "‎ You can click on 👍 or 👎 to provide feedback regarding the quality of responses.",
                    icon="💡",
                )
                time.sleep(1)
            st.write("")
            st.session_state["disclaimer"] = True

st.sidebar.header("💬 Past Conversations")

sc1, sc2 = st.sidebar.columns((6, 1))

history_keys = [
    key for key in reversed(list(st.context.cookies)) if key.startswith("history")
]

for key, conversation_id in enumerate(history_keys):

    content = json.loads(urllib.parse.unquote(st.context.cookies.get(conversation_id)))

    title = list(content.keys())[0]

    if sc1.button(title, key=f"c{key}"):
        st.sidebar.info(f'Reload "{title}"', icon="💬")

    if sc2.button("❌", key=f"x{key}"):
        st.sidebar.info("Conversation removed", icon="❌")
        cookie_manager.delete(conversation_id)

if "conversation_id" in st.session_state:

    conversation_title = get_conversation_title()
    cookie_manager.set(
        st.session_state["conversation_id"],
        val={conversation_title: st.session_state["messages"]},
    )
