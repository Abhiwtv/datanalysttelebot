import json
import re
from google import genai
from prompts import PLANNER_PROMPT
from groq import Groq
import logging
import os
logger=logging.getLogger(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def planner(user_query: str) -> dict:
    """Uses Groq to break down the user query into an actionable JSON plan."""
    
    # 1. Provide the exact, strict schema to Groq
    system_prompt = """You are a precise data analysis planner. 
You MUST output valid JSON exactly matching this schema:
{
    "dataset_source": "Name of the organization (e.g., RBI, MOSPI, etc.)",
    "dataset_name": "Name of the dataset",
    "expected_domain": "Official domain (e.g., rbi.org.in). Leave empty if unknown.",
    "dataset_link": "Extract ANY direct http/https URL provided in the user's prompt. If no URL is provided, leave this empty.",
    "search_keywords": ["Combine source, dataset, and current year into a strict search string, e.g., 'RBI Handbook of Statistics on Indian States 2022-23'"],
    "task_type": "The operation to perform (e.g., argmin, sum)",
    "aggregation": "Any aggregation required",
    "requested_fields": ["field1", "field2"],
    "filters": {"year": "2021-22", "price_type": "current prices"}
}"""

    try:
        # 2. Call Groq
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Plan the following query:\n{user_query}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"} 
        )
        
        output = response.choices[0].message.content.strip()
        plan = json.loads(output)
        
        # 3. INDESTRUCTIBLE FALLBACKS
        # If Groq somehow returns an empty keyword list, force the original query as the keyword
        if "search_keywords" not in plan or not plan["search_keywords"]:
            plan["search_keywords"] = [user_query]
            
        # Ensure expected_domain exists so retriever doesn't crash
        if "expected_domain" not in plan:
            plan["expected_domain"] = ""
            
        # Ensure filters exists
        if "filters" not in plan:
            plan["filters"] = {}
            
        logger.info(f"Generated Plan: {json.dumps(plan, indent=2)}")
        return plan

    except Exception as e:
        logger.error(f"Planner API error or Parse failure: {e}")
        # Absolute last resort fallback to keep the pipeline moving
        return {
            "search_keywords": [user_query],
            "dataset_name": "",
            "expected_domain": "",
            "requested_fields": [],
            "filters": {}
        }