import streamlit as st
from llama_index.core.agent import ReActAgent
from utils import get_doc_tools
import GPUtil
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import os, json
from streamlit_float import *
from loguru import logger


os.environ["TOKENIZERS_PARALLELISM"] = "false"
if st.secrets.get("OPENAI_API_KEY") is not None:
    logger.debug(f"Reading openai key from streamlit secret...")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
device = "cuda" if len(GPUtil.getAvailable()) >= 1 else "cpu"

# Streamlit UI Setup
st.set_page_config(initial_sidebar_state="collapsed")
ss = st.session_state
st.markdown(
    """ <style> #MainMenu {visibility: hidden;} footer {visibility: hidden;} </style> """,
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)


def save_feedback(index):
    feedback = ss[f"feedback_{index}"]
    trace = ss.get(f"trace_{index}")
    ss.messages[index]["feedback"] = feedback
    if int(feedback) == 0:
        ss.show_comment_box = index
    elif trace:
        ss.show_comment_box = None


def save_comment(index):
    comment = ss.get(f"comment_{index}", "")
    ss.messages[index]["comment"] = comment
    ss.show_comment_box = None  # Close comment box
    feedback = ss.messages[index]["feedback"]
    trace = ss.get(f"trace_{index}")


from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI


@st.cache_resource(show_spinner="Model loading...")
def create_agent():
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.5,
    )

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3", device=device)
    Settings.llm = llm
    Settings.embed_model = embed_model

    all_tools = prepare_tools()
    return OpenAIAgent.from_tools(
        all_tools,
        system_prompt=""" \
You are an agent designed to answer queries about the documentation.\n
Important Notes:\n
1/ Always try to using the same language with user to answer.\n
2/ Please always use the tools provided to answer a question. Do not rely on prior knowledge.

""",
        verbose=True,
    )


def prepare_tools():
    sources = [
        # ("./data/COT.csv", "Coach On Tap platform"),
        ("./data/faqs_docs.pkl", "Coach On Tap platform"),
    ]
    source_to_tools_dict = {}
    for source, desc in sources:
        logger.info(f"Getting tools for source: {source}")
        vector_tool, summary_tool = get_doc_tools(
            source,
            Path(source).stem,
            desc,
            extra_sources=["https://www.coachontap.co/about-us"],
        )
        source_to_tools_dict[source] = [vector_tool, summary_tool]

    # Create all tools
    all_tools = [t for s, _ in sources for t in source_to_tools_dict[s]]
    for i in all_tools:
        logger.info(i.metadata)
    return all_tools


def main():
    # Define variables in state
    if "show_comment_box" not in ss:
        ss.show_comment_box = None
    if "trace_id" not in ss:
        ss.trace_id = None
    if "is_show_feedback" not in ss:
        ss.is_show_feedback = False
    if "messages" not in ss:
        ss["messages"] = [
            {
                "role": "assistant",
                "content": "Xin chào, tôi có thể giúp gì cho bạn ?",
            }
        ]

    agent = create_agent()

    ## Display messages
    for i, message in enumerate(ss.messages):
        st.chat_message(message["role"]).write(message["content"])
        if message["role"] == "assistant":
            feedback = message.get("feedback", None)
            ss[f"feedback_{i}"] = feedback
            st.feedback(
                "thumbs",
                key=f"feedback_{i}",
                disabled=feedback is not None,
                on_change=save_feedback,
                args=[i],
            )
            # Show comment box if "👎" is clicked
            if ss.show_comment_box == i:
                st.text_area(
                    "Có thể cải thiện được điều gì?",
                    key=f"comment_{i}",
                    placeholder="Góp ý ở đây...",
                    on_change=save_comment,
                    args=[i],
                )
    if prompt := st.chat_input(key="chat_input"):
        st.chat_message("user").write(prompt)
        ss.messages.append({"role": "user", "content": prompt})

        response_gen = None
        try:
            with st.spinner(text="Thinking..."):
                response_gen = agent.stream_chat(prompt.strip().lower()).response_gen
        except Exception as e:
            response_gen = None
        if response_gen:
            response = st.chat_message("assistant").write_stream(response_gen)
        else:
            response = "Could you please ask me again."
            st.chat_message("assistant").write(response)

        n_messages = len(ss.messages)

        st.feedback(
            "thumbs",
            key=f"feedback_{n_messages}",
            on_change=save_feedback,
            args=[n_messages],
        )
        ss.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
