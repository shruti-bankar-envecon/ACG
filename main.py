## FastApi for ACG Project

import os
import json
import pytesseract
import re
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# OpenAI API Key Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found! Set OPENAI_API_KEY as an environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

# Function to extract text from PDF
async def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        images = convert_from_path(pdf_path)
        extracted_text = "\n".join(pytesseract.image_to_string(img, config="--psm 6") for img in images).strip()
        return extracted_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {e}")

# Function to process text using GPT-4o
async def process_text_with_gpt4o(extracted_text: str) -> dict:
    if not extracted_text.strip():
        raise HTTPException(status_code=400, detail="Extracted text is empty. Cannot process with GPT-4o.")

    try:
        prompt = f"""
        Extract the following tariff data into structured JSON format. Ensure precise data mapping for:
        - Category (Import, Export, General Charges)
        - Job Description
        - 20Ft Charges
        - 40Ft Charges
        - Currency (INR assumed if not specified)

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
        content = response.choices[0].message.content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'({.*})', content, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                raise HTTPException(status_code=500, detail="Failed to extract valid JSON from GPT-4o response.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPT-4o processing failed: {e}")

# Save results to JSON
async def save_results(data: dict, json_path: str):
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

# FastAPI Endpoints
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        temp_pdf_path = Path("temp") / file.filename
        temp_pdf_path.parent.mkdir(exist_ok=True)
        with temp_pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        extracted_text = await extract_text_from_pdf(str(temp_pdf_path))
        categorized_data = await process_text_with_gpt4o(extracted_text)
        json_path = Path("output") / f"{file.filename}.json"
        json_path.parent.mkdir(exist_ok=True)
        await save_results(categorized_data, str(json_path))
        
        return {"message": "Processing complete", "json_data": categorized_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
