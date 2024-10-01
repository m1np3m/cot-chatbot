import streamlit as st
from llama_index.core.agent import ReActAgent
from utils import get_doc_tools
import GPUtil
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.mistralai import MistralAI
from pathlib import Path
from streamlit_chat import message
import os, json
from datetime import datetime
import time
from streamlit_float import *
from loguru import logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "cuda" if len(GPUtil.getAvailable()) >= 1 else "cpu"

# Define a variable to enable/disable chat_input()
if "is_chat_input_disabled" not in st.session_state:
    st.session_state.is_chat_input_disabled = False


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


@st.cache_resource(show_spinner=False)
def setup():

    llm = MistralAI(
        api_key="jiSxvwweunDg9qY8LasnngBrqPVaPMGb",
        temperature=0.1,
    )

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device=device
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    all_tools = prepare_tools()
    return ReActAgent.from_tools(all_tools, verbose=False)


def prepare_tools():
    sources = [
        ("./data/COT.csv", "Coach On Tap platform"),
        # ("./data/ANZ.pdf", "ANZ Bank"),
    ]
    source_to_tools_dict = {}
    for source, desc in sources:
        print(f"Getting tools for source: {source}")
        vector_tool, summary_tool = get_doc_tools(source, Path(source).stem, desc)
        source_to_tools_dict[source] = [vector_tool, summary_tool]

    # Create all tools
    all_tools = [t for s, _ in sources for t in source_to_tools_dict[s]]
    for i in all_tools:
        print(i.metadata)
    return all_tools


def main():
    st.title("💬 FAQs Chatbot")
    st.caption("[Coach On Tap](https://coachontap.co)")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}
        ]
    agent = setup()
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input(key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        response_gen = None
        try:
            response_gen = agent.stream_chat(prompt.strip().lower()).response_gen
        except Exception as e:
            response_gen = None
        if response_gen:
            response = st.chat_message("assistant").write_stream(response_gen)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            response = "Could you please ask me again."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)
    # If there is at least one message in the chat, we display the options
    if len(st.session_state["messages"]) > 0:
        action_buttons_container = st.container()
        action_buttons_container.float(
            "bottom: 7.2rem;background-color: var(--default-backgroundColor); padding-top: 1rem;"
        )

        # We set the space between the icons thanks to a share of 100
        cols_dimensions = [14.5, 8.6, 8.7, 14.9, 7, 9.1, 14.9]
        cols_dimensions.append(100 - sum(cols_dimensions))

        col0, col1, col2, col3, col4, col5, col6, col7 = (
            action_buttons_container.columns(cols_dimensions)
        )

        # with col1:

        #     # Converts the list of messages into a JSON Bytes format
        #     json_messages = json.dumps(st.session_state["messages"]).encode("utf-8")

        #     # And the corresponding Download button
        #     st.download_button(
        #         label="📥 Save!",
        #         data=json_messages,
        #         file_name="chat_conversation.json",
        #         mime="application/json",
        #     )

        with col0:

            # We set the message back to 0 and rerun the app
            # (this part could probably be improved with the cache option)
            if st.button("Clear 🧹"):
                del st.session_state["messages"]
                if "uploaded_pic" in st.session_state:
                    del st.session_state["uploaded_pic"]
                agent.reset()
                st.cache_resource.clear()
                st.rerun()

        # with col3:

        #     if st.button("🎨"):
        #         upload_document()

        # with col4:
        #     icon = "🔁"
        #     if st.button(icon):
        #         st.session_state["rerun"] = True
        #         st.rerun()

        with col1:
            icon = "👍"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        with col2:
            icon = "👎"

            # The button will trigger the logging function
            if st.button(icon):
                log_feedback(icon)

        # with col7:

        #     # We initiate a tokenizer
        #     enc = tiktoken.get_encoding("cl100k_base")

        #     # We encode the messages
        #     tokenized_full_text = enc.encode(
        #         " ".join([item["content"] for item in st.session_state["messages"]])
        #     )

        #     # And display the corresponding number of tokens
        #     label = f"💬 {len(tokenized_full_text)} tokens"
        #     st.link_button(label, "https://platform.openai.com/tokenizer")

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


if __name__ == "__main__":
    main()
