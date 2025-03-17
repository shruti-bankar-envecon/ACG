
# Here is the LLM, GPT and AI codes for ACG.

import os
import json
import pytesseract
import pandas as pd
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# OpenAI API Key Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"Loaded API Key: {OPENAI_API_KEY}")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found! Set OPENAI_API_KEY as an environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)
print("OPENAI_API_KEY="+OPENAI_API_KEY)

# Extract text from PDF using OCR
def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        return "\n".join(pytesseract.image_to_string(img, config="--psm 6") for img in images).strip()
    except Exception as e:
        print(f"[Error] Failed to extract text from PDF: {e}")
        return ""

# Process text using GPT-4o with improved error handling and detailed prompting
def process_text_with_gpt4o(extracted_text):
    if not extracted_text.strip():
        print("[Warning] Extracted text is empty. Skipping GPT processing.")
        return {}

    try:
        prompt = f"""
        Extract the following tariff data into structured JSON format. Ensure precise data mapping for:
        - Category (Import, Export, General Charges)
        - Job Description
        - 20Ft Charges
        - 40Ft Charges
        - Currency (INR assumed if not specified)

        Each job description should be accurate and detailed. Ensure all numerical values are captured correctly. Avoid summarization. 

        Data:
        {extracted_text}
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data extraction assistant. Ensure precise JSON output."},
                {"role": "user", "content": prompt}
            ]
        )

        # Improved Response Handling with Debug Info
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            print("[DEBUG] Raw GPT-4o Response Content:")
            print(content)

            try:
                return json.loads(content)
            except json.JSONDecodeError:
                print("[Warning] Attempting to extract JSON from mixed text...")
                match = re.search(r'({.*})', content, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                else:
                    print("[Error] Unable to extract valid JSON from response.")
                    return {}
        else:
            print("[Error] GPT-4o returned an empty or invalid response.")
            return {}

    except Exception as e:
        print(f"[Error] GPT-4o processing failed: {e}")
        return {}

# Save results to JSON 
def save_results(data, json_path):
    df = pd.DataFrame(data)
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# File Paths
pdf_path = "Chennai-CFS.pdf"
text_output_path = "Extracted_Tariff_Text.txt"
json_path = "Categorized_Tariff_Report.json"

# Main Execution
if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(pdf_path)

    with open(text_output_path, "w", encoding="utf-8") as text_file:
        text_file.write(extracted_text)

    categorized_data = process_text_with_gpt4o(extracted_text)

    if categorized_data:
        save_results(categorized_data, json_path)
        print(f"[Success] Processing complete. Files saved: {text_output_path}, {json_path}")
    else:
        print("[Warning] GPT processing failed. No data saved.")
        
