import os
nspl_root_dir = os.environ.get("NSPL_REPO")
import openai
from openai import OpenAI


def get_vlm_response(pre_prompt, prompt, image_url, stop="END"):
    client = OpenAI()
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    try:
        openai_response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {"role": "system", "content": pre_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                            },
                        },
                    ],
                }
            ],
            max_tokens=4000,
            stop=stop,
        )
        text = openai_response.choices[0].message.content
        text = text.strip()
    except:
        text = "MyText: Could not get response from VLM."
    return text
