import logging
from pathlib import Path
import pandas as pd
from google import genai
from prompts import INTERPRETER_PROMPT
from groq import Groq
import os

logger = logging.getLogger(__name__)


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))  # <-- Use the environment variable for security

def load_data_context(file_path: Path, max_rows: int = 250) -> str:
    """Loads a dataset and converts it into a markdown string for the LLM context."""
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".csv":
            # Using engine='python' or trying different encodings helps with messy internet CSVs
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1')
        elif ext in [".xls", ".xlsx"]:
            xls = pd.read_excel(file_path, sheet_name=None)
            data_str = ""
            
            # Force pandas to not truncate columns in the markdown output
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)

            for sheet_name, sheet_df in xls.items():
                data_str += f"\n--- Sheet: {sheet_name} ---\n"
                
                # Drop completely empty columns/rows to save LLM context space
                sheet_df = sheet_df.dropna(how='all', axis=1).dropna(how='all', axis=0)
                
                if len(sheet_df) > max_rows:
                    data_str += sheet_df.head(max_rows).to_markdown(index=False)
                else:
                    data_str += sheet_df.to_markdown(index=False)
                    
            pd.reset_option('display.max_columns') # Reset after we are done
            return data_str
        elif ext == ".parquet":
            df = pd.read_parquet(file_path)
        elif ext == ".json":
            df = pd.read_json(file_path)
        else:
            return f"Unsupported dataset file type: {ext}"

        # Truncate row count to prevent exceeding the model's token limits
        if len(df) > max_rows:
            data_str = f"Data truncated (showing first {max_rows} of {len(df)} rows):\n\n"
            data_str += df.head(max_rows).to_markdown(index=False)
        else:
            data_str = df.to_markdown(index=False)
            
        return data_str

    except Exception as e:
        logger.error(f"Failed to load dataset for interpretation: {e}")
        return f"Error reading the dataset: {str(e)}"


def interpret_data(user_query: str, file_path: Path, log_url: str) -> str:
    """Feeds the dataset and user query to the LLM for final interpretation."""
    logger.info(f"Ingesting {file_path.name} for interpretation...")
    
    data_context = load_data_context(file_path)
    
    prompt = INTERPRETER_PROMPT.format(
        question=user_query,
        file_name=file_path.name,
        data_context=data_context,
        log_url=log_url  # <-- Injecting the real URL here
    )

    logger.info("Sending data context to Gemini for analysis...")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a precise data science assistant that responds strictly in valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="llama-3.3-70b-versatile",  # Fast and highly accurate for reasoning
        temperature=0.1                  # Low temperature for strict adherence to formatting
    )

    # Clean potential markdown wrappers if the model outputs strict JSON
    output = response.choices[0].message.content.strip()
    if output.startswith("```json") and output.endswith("```"):
        output = output[7:-3].strip()
    elif output.startswith("```") and output.endswith("```"):
        output = output[3:-3].strip()

    return output