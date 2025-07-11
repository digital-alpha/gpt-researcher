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
logger.propagate = False


class Provider(Enum):
    OPENAI = {
        "FAST_LLM": "openai:gpt-4.1-mini-2025-04-14",
        "SMART_LLM": "openai:gpt-4.1-2025-04-14",
        "STRATEGIC_LLM": "openai:o4-mini-2025-04-16",
        "llm_kwargs": {"api_key": "<openai-api-key>"}
    }
    ANTHROPIC = {
        "FAST_LLM": "anthropic:claude-3-5-sonnet-latest",
        "SMART_LLM": "anthropic:claude-3-5-sonnet-latest",
        "STRATEGIC_LLM": "anthropic:claude-3-5-sonnet-latest"
    }
    GEMINI = {
        "FAST_LLM": "google_genai:gemini-2.0-flash-001",
        "SMART_LLM": "google_genai:gemini-2.0-flash-001",
        "STRATEGIC_LLM": "google_genai:gemini-2.0-flash-001",
        "llm_kwargs": {"google_api_key": "<google-api-key>"}
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
        "EMBEDDING_KWARGS": {"region_name": "us-east-1"},
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
            api_key="<qdrant-api-key>",
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
        verbose=True,
        )
    researcher.retrievers[0].topic = "finance" # type: ignore #default "general"
    researcher.retrievers[0].api_key = "<tavily-api-key>" # type: ignore #default "general"

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
    researcher.set_verbose(True)
    await researcher.conduct_research()
    report = await researcher.write_report()

    print(report)
    filename = f"./extensions/research_report.txt"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(report)
    print(f"Report saved to {filename}")

if __name__ == "__main__":
    # query = "how are the financials of the company which the file is attached of compared to in 2018. also put the name of the company wherever necessary in the final report."
    # query = "how are the financials of jetblue? compare the latest one with one from 2018."
    
    query = """
    Replicate the spirit of Cohen-Nguyen (2024) for NVIDIA (NVDA)  Identify performance metrics / targets that disappear from management discourse and assess whether those disappearances foreshadow weaker fundamentals or share-price under-performance.
List stated performance “targets” for each call
Focus on metrics explicitly described as goals, guidance, or outlook
Record each target in a structured table: Quarter, Target phrase, Type (financial / operating / product), Exact wording, Source URL.
Detect dropped targets
For each quarter t, compare its target list to the same quarter a year earlier (t – 4).
Mark any metric that disappears as Dropped = Yes.
Compute: MovingTargetRatio_t = DroppedCount / TotalTargets_{t-4}.
Cross-check analyst attention
Scan corresponding Q&A text for each dropped target phrase.
Flag AnalystFollowUp = Yes if analysts explicitly ask about that metric.
Gauge short-run market reaction
Pull NVIDIA’s closing prices from Yahoo Finance: day-before, day-of, and day-after the call.
Compute simple ±1-day return around each call; note if price reaction is muted (< ±1 %).
Gauge medium-term drift
Record 3-month and 6-month forward price change (call-to-call).
Compare average drift for high vs low MovingTargetRatio quarters.
Narrative synthesis
Highlight notable dropped metrics (specific product lines).
Link each to subsequent news: revenue misses, segment slow-downs, guidance cuts.
Comment on whether analyst follow-up mitigated any negative drift."""
    
    raw_json = """
{
  "model_provider": "openai",
  "report_source": "web"
}
"""
    in_json = json.loads(raw_json)
    
    asyncio.run(get_report(query, in_json))