from openai import OpenAI
import os
from dotenv import load_dotenv  
load_dotenv()

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def ask_llm(question, context):
    prompt = f"""
Please answer the following question based on the provided notes:

Notes:
{context}

Question:
{question}
"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
        {"role": "system", "content": "You are a helpful assistant who answers based on the given notes."},
        {"role": "user", "content": f"Notes:\n{context}\n\nQuestion: {question}"}
    ]
    )

    return response.choices[0].message.content
    