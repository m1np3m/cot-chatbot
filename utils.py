# TODO: abstract all of this into a function that takes in a PDF file name

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool


def get_doc_tools(file_path: str, name: str, desc: str) -> str:
    """Get vector query and summary query tools from a document."""
    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    index = VectorStoreIndex.from_documents(documents)

    summary_index = SummaryIndex(nodes)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    query_engine = index.as_query_engine()

    summary_tool = QueryEngineTool.from_defaults(
        name=f"{name.lower()}_summary_tool",
        query_engine=summary_query_engine,
        description=(f"Useful if you want to get a summary of {desc}."),
    )

    rag_tool = QueryEngineTool.from_defaults(
        query_engine,
        name=f"{name.lower()}_qna_tool",
        description=f"A RAG engine with some facts about {desc}.",
    )
    return summary_tool, rag_tool
