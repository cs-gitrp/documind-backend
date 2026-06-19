from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

PROMPT_TEMPLATES = {
    "detailed": "Answer comprehensively using only the context below. Include all relevant details.",
    "balanced": "Answer clearly and concisely using only the context below.",
    "concise": "Answer in 2-3 sentences only using the context below."
}

def build_prompt(query: str, chunks: list, response_mode: str = "detailed"):
    instruction = PROMPT_TEMPLATES.get(response_mode, PROMPT_TEMPLATES["balanced"])
    context = "\n\n".join([f"[Chunk {i+1}]: {c['chunk']}" for i, c in enumerate(chunks)])
    return f"""{instruction}
If the answer is not in the context, say: "I cannot find this information in the provided document."

Context:
{context}

Question: {query}
Answer:"""

def stream_response(prompt: str, temperature: float = 0.7):
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta