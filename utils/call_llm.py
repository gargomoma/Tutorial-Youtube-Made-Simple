import os
from openai import OpenAI
from anthropic import AnthropicVertex

def call_llm(prompt: str) -> str:
    """
    Calls the appropriate LLM based on environment variables.
    """
    if os.getenv("OPENAI_API_URL") and OpenAI:
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_API_URL", "")
        )
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_API_MODEL", ""),
            n=1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    elif os.getenv("ANTHROPIC_PROJECT_ID") and AnthropicVertex:
        client = AnthropicVertex(
            region=os.getenv("ANTHROPIC_REGION", "us-east5"),
            project_id=os.getenv("ANTHROPIC_PROJECT_ID", "")
        )
        response = client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            model="claude-3-7-sonnet@20250219"
        )
        return response.content[0].text
    else:
        print("Error: No LLM provider configured.  Set either OPENAI_API_URL or ANTHROPIC_PROJECT_ID.")
        return None  # Indicate failure


if __name__ == "__main__":
    test_prompt = "Hello, how are you?"
    response = call_llm(test_prompt)
    if response:
        print(f"Test successful. Response: {response}")
    else:
        print("Test failed.")