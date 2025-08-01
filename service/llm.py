from openai import OpenAI
import os
from dotenv import load_dotenv
from service.utils.config import get_logger

load_dotenv()

logger = get_logger()
OPENAI_MODEL = "gpt-4.1"


def load_prompt(path):
    """
    Loads a prompt file from a path relative to the project root (e.g. 'resource/prompt/system/xyz.txt').
    I need to improve this... let me see what other scripts require llm
    """
    # Start from this file's location (llm.py)
    this_file = os.path.abspath(__file__)

    # Assume 'service/' is one level under the project root
    project_root = os.path.abspath(os.path.join(this_file, "..", ".."))

    # Build the final absolute path
    abs_path = os.path.join(project_root, path)

    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def fill_prompt(template, variables):
    for key, value in variables.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    return template


class Llm:
    def __init__(self):
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.exception(
                f"Failed to initialize OpenAI client: {e}. API calls will fail."
            )
            self.client = None

    def _call_openai_chat(self, messages, temperature=0.2, max_tokens=None):
        if not self.client:
            logger.error("OpenAI client not initialized.")
            return None
        try:
            kwargs = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = int(max_tokens)
            response = self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            logger.exception(f"LLM call failed: {e}")
            return None

    def call_with_text(self, system_prompt, user_prompt, temperature=0.2, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self._call_openai_chat(messages, temperature, max_tokens)
        if response is None:
            return None
        return response.choices[0].message.content.strip()

    def call_with_image(self, system_prompt, user_prompt, img_url, temperature=0.2, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": img_url},
                    },
                ],
            },
        ]
        response = self._call_openai_chat(messages, temperature, max_tokens)
        if response is None:
            return None
        return response.choices[0].message.content.strip()
