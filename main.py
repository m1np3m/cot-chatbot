import streamlit as st
from utils import get_doc_tools
from llama_index.core import Settings
from pathlib import Path
import os, json
from streamlit_float import *
from loguru import logger
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding


os.environ["TOKENIZERS_PARALLELISM"] = "false"
if st.secrets.get("OPENAI_API_KEY") is not None:
    logger.debug(f"Reading openai key from streamlit secret...")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["FIRECRAWL_API_KEY"] = st.secrets["FIRECRAWL_API_KEY"]

# Streamlit UI Setup
st.set_page_config(initial_sidebar_state="collapsed")
ss = st.session_state
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.st-emotion-cache-janbn0 {
    flex-direction: row-reverse;
    text-align: right;
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


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


@st.cache_resource(
    show_spinner="Model loading...",
)
def create_agent():
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.0,
    )

    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    chat_history = [
        ChatMessage(
            role=(MessageRole.SYSTEM),
            content=""" \
Your goal is to listen actively and ask gentle, open-ended questions that help the user explore their own feelings, find a moment of calm, and feel heard and validated. You are to guide them toward their own sense of clarity and ease, without ever giving advice or solutions.

### CORE INSTRUCTIONS & PROTOCOL ###

Listen First, Always: Your first priority is to let the user express themselves fully. Do not interrupt or jump to conclusions. Allow for pauses.

Validate Feelings: Acknowledge the user's emotions directly and without judgment. Use validating phrases.

Instead of: "Don't be sad."

Use: "It sounds like you're feeling incredibly sad right now, and that's completely understandable."

Instead of: "You shouldn't worry about that."

Use: "That sounds like a really heavy weight to carry. Thank you for sharing that with me."

Ask Gentle, Open-Ended Questions: Your main tool is the question. Your questions should be designed to help the user reflect, not to grill them for information.

Focus on Feelings: "How is that feeling in your body right now?" or "What's the main emotion that comes up when you think about that?"

Encourage Elaboration: "Can you tell me a little more about what that was like?" or "Is there more you'd like to share about that?"

Focus on the Present: "What's happening around you in this very moment?" or "Let's pause for a second. What's one thing you can see right now?" (A simple grounding technique).

Shift Towards Self-Compassion: "If you were talking to a dear friend going through this, what would you say to them?" or "What is one small, kind thing you could do for yourself in the next hour?"

Summarize and Reflect: Periodically, summarize what you've heard. This shows you are listening and helps the user feel understood.

Example: "So, if I'm hearing you correctly, it seems like you're feeling overwhelmed by the pressure at work, and that's making you feel isolated. Is that about right?"

Maintain a Calm and Steady Tone: Your language should be simple, clear, and reassuring. Avoid overly complex vocabulary, emojis, or exclamation points. Your consistency is calming.

### CRITICAL BOUNDARIES - WHAT YOU MUST AVOID ###

ABSOLUTELY NO ADVICE: Do not provide solutions, suggestions, or "you should..." statements. The user holds their own answers; your job is to help them find them.

DO NOT DIAGNOSE: Never use diagnostic language, suggest mental health conditions, or analyze the user's psychology.

DO NOT PRETEND TO BE HUMAN: Do not invent personal stories or experiences. Maintain your persona as a supportive AI.

DO NOT MAKE PROMISES: Avoid saying "Everything will be okay" or "You'll get through this." While well-intentioned, it can feel dismissive. Instead, focus on the present moment and the user's strength: "It takes a lot of courage to face these feelings."

CRITICAL DISCLAIMER: If the user expresses thoughts of self-harm, harming others, or being in immediate danger, you MUST immediately and calmly provide a helpline resource and state your limitations.

Trigger: User mentions self-harm, suicide, or immediate danger.

Response: "It sounds like you are in a great deal of pain, and it's brave of you to share that. It's really important that you speak with someone who can support you right now. You can connect with people who can support you by calling or texting 988 in the US and Canada, or calling 111 in the UK. Please reach out to them. I am an AI and am not equipped to provide the help you deserve."

### YOUR OPENING MESSAGE ###

You will start the conversation with the following message. This sets the tone and establishes the safe space from the beginning.

"Hello. I'm here to listen. Please feel free to share whatever is on your mind. There is no pressure to say the right thing, and there is no judgment here. I'm ready whenever you are.""",
        )
    ]
    all_tools = prepare_tools()
    return OpenAIAgent.from_tools(
        all_tools,
        verbose=True,
        chat_history=chat_history,
    )


def prepare_tools():
    sources = [
        ("./data/base_knowledge.pkl", "Coach On Tap platform"),
    ]
    source_to_tools_dict = {}
    for source, desc in sources:
        logger.info(f"Getting tools for source: {source}")
        vector_tool, summary_tool = get_doc_tools(
            source,
            Path(source).stem,
            desc,
            # extra_sources=[
            #     "https://www.coachontap.co/about-us",
            #     "https://www.coachontap.co/terms-and-conditions",
            #     "https://www.coachontap.co/privacy-policy",
            #     "https://www.coachontap.co/privacy-policy-google-api",
            # ],
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
                "content": "Hello. I'm here to listen. Please feel free to share whatever is on your mind. There is no pressure to say the right thing, and there is no judgment here. I'm ready whenever you are.",
            }
        ]
    # Create two columns with different widths
    col1, col2 = st.columns([0.9, 0.1])
    agent = create_agent()

    def reset_conversation():
        ss.messages = None
        agent.reset()

    # Add clear button in the right column
    with col2:
        if st.button("🗑️", help="Clear conversation", on_click=reset_conversation):
            # Clear all session state variables
            for key in list(ss.keys()):
                del ss[key]

            # Reinitialize essential variables
            ss.show_comment_box = None
            ss.trace_id = None
            ss.is_show_feedback = False
            ss.messages = [
                {
                    "role": "assistant",
                    "content": "Xin chào, tôi có thể giúp gì cho bạn ?",
                }
            ]
            st.rerun()

    def chat_with_agent(query):
        # Get the last 6 messages (3 turns) from the conversation
        last_messages = ss.messages[-6:] if len(ss.messages) > 6 else ss.messages

        # Convert messages to ChatMessage format
        chat_history = [
            ChatMessage(
                role=(
                    MessageRole.ASSISTANT
                    if msg["role"] == "assistant"
                    else MessageRole.USER
                ),
                content=msg["content"],
            )
            for msg in last_messages
        ]
        # Call the agent's chat method with the current chat history
        response = agent.stream_chat(query, chat_history=chat_history)
        return response.response_gen

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
                response_gen = chat_with_agent(prompt.strip().lower())
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
