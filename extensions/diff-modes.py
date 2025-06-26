from gpt_researcher import GPTResearcher
import gpt_researcher
print(gpt_researcher.__file__)

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_aws import BedrockEmbeddings

import asyncio
from enum import Enum
import json
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    Hybrid = "dual"

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
        og_client = QdrantClient(
            url="https://ft-vdb.epiphaiplatform.com:443",
            api_key="Y3NMMGK3Okzt7rzho88jhmzPZl5Mhhnd98i39bLG4OJPGRtNV7pH8sOlVNtveGce",
        )
        client = QdrantClient(location=":memory:")

        og_client.migrate(
            dest_client=client,
            collection_names=["synchrony_collection"],
        )

        vector_store = QdrantVectorStore(
            client=client,
            collection_name="synchrony_collection",
            content_payload_key="content",
            # metadata_payload_key="document_key",
            embedding=BedrockEmbeddings(region_name="us-east-1", model_id="amazon.titan-embed-text-v2:0"),
        )

    researcher = GPTResearcher(
        query=query,
        config_path=temp_config_path,
        vector_store=vector_store,
        report_source=config_dict["REPORT_SOURCE"],
        verbose=True)

    # print(json.dumps(researcher.cfg.__dict__, indent=2))
    
    log_researcher_config(researcher)
    
    os.remove(temp_config_path)

    return researcher

def log_researcher_config(researcher):
    try:
        config_info = {
            'report_source': researcher.report_source,
            'report_type': researcher.report_type,
            'embedding': researcher.cfg.embedding,
            'fast llm': researcher.cfg.fast_llm_model,
            'strategic llm': researcher.cfg.strategic_llm_model,
            'smart llm': researcher.cfg.smart_llm_model,
        }
        
        logger.info(f"GPTResearcher Configuration: {json.dumps(config_info, indent=2)}")
        
    except AttributeError as e:
        logger.warning(f"Could not access some researcher attributes: {e}")

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
    filename = f"./extensions/research_report.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(report)
    print(f"Report saved to {filename}")

if __name__ == "__main__":
    query = "how are the financials of the company which the file is attached of compared to in 2018. also put the name of the company wherever necessary in the final report."
    raw_json = """
{
  "model_provider": "gemini",
  "report_source": "hybrid"
}
"""
    in_json = json.loads(raw_json)
    
    asyncio.run(get_report(query, in_json))