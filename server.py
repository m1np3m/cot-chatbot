from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.memory import ChatMemoryBuffer



from llama_index.core import Settings
from pathlib import Path
import os, json
from loguru import logger
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from utils import get_doc_tools

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI()


def create_agent():
    llm = OpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
    )

    Settings.llm = llm
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    chat_history = [
        ChatMessage(
            role=(MessageRole.SYSTEM),
            content=""" \
You are an agent designed to answer queries about the documentation.\n
Important Notes:\n
1/ Always try to answer in the same language with user.\n
2/ Please always use the tools provided to answer a question. Do not rely on prior knowledge.""",
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
        ("./data/faqs_docs.pkl", "Coach On Tap platform"),
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


agent = create_agent()


class ChatRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = []


@app.post("/chat")
async def chat(request: ChatRequest):
    chat_history = [
        ChatMessage(
            role=(
                MessageRole.ASSISTANT
                if msg["role"] == "assistant"
                else MessageRole.USER
            ),
            content=msg["content"],
        )
        for msg in request.history
    ]
    response = agent.stream_chat(request.query, chat_history=chat_history)

    def response_generator():
        for token in response.response_gen:
            yield token

    return StreamingResponse(response_generator(), media_type="text/event-stream")
