import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import openai
from openai import OpenAI


def get_llm_response(pre_prompt, prompt, temperature=0.0, stop="END"):
    SEED = 0
    client = OpenAI()
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    openai_response = client.chat.completions.create(
        # model="gpt-4-1106-preview",
        model="gpt-4",
        messages=[
            {"role": "system", "content": pre_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        stop=stop,
        seed=SEED,
    )
    # text = openai_response['choices'][0]['message']['content']
    text = openai_response.choices[0].message.content
    text = text.strip()
    return text
