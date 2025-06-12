from gpt_researcher import GPTResearcher
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_aws import BedrockEmbeddings

from qdrant_client.http.models import Distance, VectorParams
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

import asyncio
from enum import Enum
import json
import tempfile
import os


class Provider(Enum):
    OPENAI = {
        "FAST_LLM": "openai:gpt-4.1-mini-2025-04-14",
        "SMART_LLM": "openai:gpt-4.1-mini-2025-04-14",
        "STRATEGIC_LLM": "openai:gpt-4.1-mini-2025-04-14"
    }
    ANTHROPIC = {
        "FAST_LLM": "anthropic:claude-3-5-sonnet-latest",
        "SMART_LLM": "anthropic:claude-3-5-sonnet-latest",
        "STRATEGIC_LLM": "anthropic:claude-3-5-sonnet-latest"
    }
    GEMINI = {
        "FAST_LLM": "google_genai:gemini-2.0-flash-001",
        "SMART_LLM": "google_genai:gemini-2.0-flash-001",
        "STRATEGIC_LLM": "google_genai:gemini-2.0-flash-001"
    }

class ReportSource(Enum):
    Web = "web"
    Internal = "langchain_vectorstore"
    Hybrid = "hybrid"

def get_researcher(
        query: str,
        report_source: ReportSource,
        provider: Provider
    ) -> GPTResearcher:

    config_dict = {
        "RETRIEVER": "tavily",
        "EMBEDDING": "bedrock:amazon.titan-embed-text-v2:0",
        "EMBEDDING_KWARGS": {"region_name": "us-east-1"}
    }
    config_dict.update(provider.value)
    config_dict["REPORT_SOURCE"] = report_source.value

    temp_config_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
    try:
        json.dump(config_dict, temp_config_file)
        temp_config_path = temp_config_file.name
    finally:
        temp_config_file.close()

    vector_store = None
    if report_source != ReportSource.Web:
        client = QdrantClient(
            url="QDRANT_URL",
            api_key="QDRANT_API_KEY",
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="synchrony_collection",
            content_payload_key="content",
            embedding=BedrockEmbeddings(region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0"),
        )

        # vector_store = setup_store()

    researcher = GPTResearcher(
        query=query,
        config_path=temp_config_path,
        vector_store=vector_store,
        report_source=config_dict["REPORT_SOURCE"])
    
    # print(json.dumps(researcher.cfg.__dict__, indent=2))
    
    os.remove(temp_config_path)

    return researcher

# def setup_store(file_path: str = "./notebooks/DA/storage/docs/Nvidia-stock.md"):
#     try:
#         with open(file_path, 'r', encoding='utf-8') as file:
#             essay = file.read()
#         print(f"Successfully read essay from {file_path}")
#     except FileNotFoundError:
#         print(f"Essay file not found at {file_path}. Using a placeholder text.")
#         essay = "This is a placeholder text because the essay file wasn't found."
#     except Exception as e:
#         print(f"Error reading essay file: {str(e)}. Using a placeholder text.")
#         essay = "This is a placeholder text due to error reading the essay file."

#     document = [Document(page_content=essay)]
#     text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30, separator="\n")
#     docs = text_splitter.split_documents(documents=document)

#     client = QdrantClient(location=":memory:")

#     try:
#         client.create_collection(
#             collection_name="demo_collection",
#             vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
#         )
#         collection_created = True
#     except ValueError:
#         print("Using existing collection...")
#         collection_created = False
        
#     vector_store = QdrantVectorStore(
#         client=client,
#         collection_name="demo_collection",
#         embedding=BedrockEmbeddings(region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0"),
#     )

#     if collection_created:
#         vector_store.add_documents(documents=docs)
#         print("Documents added to the collection.")
    
#     return vector_store


async def get_report(query: str, in_json: dict):
    provider_name = in_json.get("model_provider", "openai").upper()
    try:
        provider = Provider[provider_name]
    except KeyError:
        provider = Provider.OPENAI
    
    report_source_name = in_json.get("report_source", "web").lower()
    if report_source_name == "internal":
        report_source = ReportSource.Internal
    elif report_source_name == "hybrid":
        report_source = ReportSource.Hybrid
    else:
        report_source = ReportSource.Web
    
    researcher = get_researcher(query, report_source, provider)
        
    await researcher.conduct_research()
    report = await researcher.write_report()

    print(report)

if __name__ == "__main__":
    query = "What company is the document about?"
    raw_json = """
{
  "model_provider": "openai",
  "report_source": "internal"
}
"""
    in_json = json.loads(raw_json)
    
    asyncio.run(get_report(query, in_json))