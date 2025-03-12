# TODO: abstract all of this into a function that takes in a PDF file name

from logging import Logger
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from pymongo import MongoClient
from llama_index.core import Document, Settings
import joblib as jl
from loguru import logger
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.web import FireCrawlWebReader
import os

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

mongo_uri = "mongodb://admin-prod:419Gjjkw084XeqXU@ec2-52-3-143-218.compute-1.amazonaws.com:27017/"

# MongoDB connection details
database_name = "COT"
faq_item_collection = "ItemFAQs"
faq_topic_collection = "TopicFAQs"

# Create a MongoClient
client = MongoClient(mongo_uri)

# # # Access the specific database and collection
db = client[database_name]
faq_items = db[faq_item_collection]
faq_topic = db[faq_topic_collection]


def generate_faqs_document(doc_path: str):
    # Perform the join using $lookup
    pipeline = [
        {
            "$lookup": {
                "from": faq_topic_collection,  # Collection to join (Topic)
                "localField": "topic_id",  # Field from the Item collection
                "foreignField": "_id",  # Field from the Topic collection
                "as": "faq_topic",  # Name of the output array field
            }
        },
        {"$unwind": "$faq_topic"},  # Unwind the array to get a flat structure
        {
            "$project": {
                "question": 1,  # Include title from the Topic
                "answer": 1,  # Include title from the Topic
                "faq_topic.title": 1,  # Include title from the Topic
                "faq_topic.topicType": 1,  # Include title from the Topic
                "updated_at": 1,  # Include title from the Topic
            }
        },
    ]

    # Run the aggregation pipeline
    results = faq_items.aggregate(pipeline)

    docs = []
    # Display the joined documents
    for doc in results:
        docs.append(
            Document(
                id_=str(doc["_id"]),
                text=f'question: {doc["question"]}\nanswer: {doc["answer"]}',
                metadata={
                    **doc["faq_topic"],
                    **{"updated_at": doc["updated_at"].isoformat()},
                },
            )
        )
    jl.dump(docs, doc_path)
    return docs


def get_doc_tools(file_path: str, name: str, desc: str, **kwargs) -> str:
    """Get vector query and summary query tools from a document."""
    # load documents
    import joblib

    extra_sources = kwargs.get("extra_sources")
    file_extention = file_path.split(".")[-1]
    if file_extention == "pkl":
        try:
            documents = joblib.load(file_path)
        except:
            documents = generate_faqs_document(doc_path=file_path)
    else:
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    if extra_sources and len(extra_sources) > 0:
        # Initialize FireCrawlWebReader to crawl a website
        firecrawl_reader = FireCrawlWebReader(
            api_key=os.environ[
                "FIRECRAWL_API_KEY"
            ],  # Rep;lace with your actual API key from https://www.firecrawl.dev/
            mode="scrape",  # Choose between "crawl" and "scrape" for single page scraping
        )
        for source in extra_sources:
            # Load documents from a single page URL
            ex_documents = firecrawl_reader.load_data(url=source)
            documents.extend(ex_documents)

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
