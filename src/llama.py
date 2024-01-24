# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

int8 gptq model need: pip install optimum auto-gptq
"""

from loguru import logger

from src.base_model import BaseLLMModel
from src.presets import LOCAL_MODELS


class LLaMAClient(BaseLLMModel):
    def __init__(self, model_name, user_name=""):
        super().__init__(model_name=model_name, user=user_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_generation_token = 1000
        logger.info(f"Loading model from {model_name}")
        if model_name in LOCAL_MODELS:
            model_path = LOCAL_MODELS[model_name]
        else:
            model_path = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype='auto').eval()
        logger.info(f"Model loaded from {model_path}")
        self.stop_str = self.tokenizer.eos_token or "</s>"

    def _get_chat_input(self):
        messages = []
        logger.debug(f"{self.history}")
        for conv in self.history:
            if conv["role"] == "system":
                messages.append({'role': 'system', 'content': conv["content"]})
            elif conv["role"] == "user":
                messages.append({'role': 'user', 'content': conv["content"]})
            else:
                messages.append({'role': 'assistant', 'content': conv["content"]})
        input_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt'
        )

        return input_ids.to(self.model.device)

    def get_answer_at_once(self):
        input_ids = self._get_chat_input()
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=self.max_generation_token,
            top_p=self.top_p,
            temperature=self.temperature,
        )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        return response, len(response)

    def get_answer_stream_iter(self):
        from transformers import TextIteratorStreamer
        from threading import Thread
        input_ids = self._get_chat_input()
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        thread = Thread(
            target=self.model.generate,
            kwargs={"input_ids": input_ids,
                    "max_new_tokens": self.max_generation_token,
                    "top_p": self.top_p,
                    "temperature": self.temperature,
                    "streamer": streamer}
        )
        thread.start()
        generated_text = ""
        for new_text in streamer:
            stop = False
            pos = new_text.find(self.stop_str)
            if pos != -1:
                new_text = new_text[:pos]
                stop = True
            generated_text += new_text
            yield generated_text
            if stop:
                break
