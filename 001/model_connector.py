import gc
import os

import anthropic
import boto3
import torch
from google import genai
from ollama import ChatResponse, chat
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelConnector:
    """様々なAIモデルプロバイダーへの接続を管理するクラス"""

    def __init__(self):
        self._current_model = None
        self.openai_client = OpenAI
        self.google_client = genai.Client()
        self.anthropic_client = anthropic.Anthropic()
        self.amazon_client = boto3.client("bedrock-runtime", region_name="us-east-1")
        self.x_client = OpenAI(
            api_key=os.environ["XAI_API_KEY"], base_url="https://api.x.ai/v1"
        )
        self.openrouter_client = OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )

    def generate_response(self, model_info, system_prompt, user_prompt):
        provider = model_info["provider"]
        self._current_model = model_info

        try:
            if provider == "openai":
                return self._call_openai_api(model_info, system_prompt, user_prompt)
            elif provider == "google":
                return self._call_google_api(model_info, system_prompt, user_prompt)
            elif provider == "anthropic":
                return self._call_anthropic_api(model_info, system_prompt, user_prompt)
            elif provider == "amazon":
                return self._call_amazon_api(model_info, system_prompt, user_prompt)
            elif provider == "x":
                return self._call_x_api(model_info, system_prompt, user_prompt)
            elif provider == "openrouter":
                return self._call_openrouter_api(model_info, system_prompt, user_prompt)
            elif provider == "ollama":
                return self._call_ollama_api(model_info, system_prompt, user_prompt)
            elif provider == "huggingface":
                return self._call_huggingface_api(
                    model_info, system_prompt, user_prompt
                )
            else:
                raise ValueError(f"Unknown provider: {provider}")
        except Exception as e:
            print(f"Error calling {model_info['name']}: {str(e)}")
        finally:
            self._clear_model()

    def _clear_model(self):
        """モデルのメモリを解放"""
        self._current_model = None
        gc.collect()

    def _call_openai_api(self, model_info, system_prompt, user_prompt):
        response = self.openai_client.responses.create(
            model=model_info["model"],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

    def _call_google_api(self, model_info, system_prompt, user_prompt):
        response = self.google_client.models.generate_content(
            model=model_info["model"],
            contents=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.text

    def _call_anthropic_api(self, model_info, system_prompt, user_prompt):
        response = self.anthropic_client.messages.create(
            model=model_info["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.content

    def _call_amazon_api(self, model_info, system_prompt, user_prompt):
        response = self.amazon_client.converse(
            modelId=model_info["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["output"]["message"]["content"][0]["text"]

    def _call_x_api(self, model_info, system_prompt, user_prompt):
        completion = self.x_client.chat.completions.create(
            model=model_info["model"],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content

    def _call_openrouter_api(self, model_info, system_prompt, user_prompt):
        completion = self.openrouter_client.chat.completions.create(
            model=model_info["model"],
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content

    def _call_ollama_api(self, model_info, system_prompt, user_prompt):
        response: ChatResponse = chat(
            model=model_info["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.message.content

    def _call_huggingface_api(self, model_info, system_prompt, user_prompt):
        tokenizer = AutoTokenizer.from_pretrained(model_info["model"])
        model = AutoModelForCausalLM.from_pretrained(
            model_info["model"], torch_dtype="auto", device_map="auto"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        tokenized_input = tokenizer.apply_chat_template(
            messages, tokenizer=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(tokenized_input)[0]

        text = tokenizer.decode(output)
        return text
