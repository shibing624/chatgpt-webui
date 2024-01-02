# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description:

Llama:
pip install llama-cpp-python

"""
import os

from loguru import logger

from src.base_model import BaseLLMModel
from src.presets import MODEL_METADATA

SYS_PREFIX = "<<SYS>>\n"
SYS_POSTFIX = "\n<</SYS>>\n\n"
INST_PREFIX = "<s>[INST] "
INST_POSTFIX = " "
OUTPUT_PREFIX = "[/INST] "
OUTPUT_POSTFIX = "</s>"


def download(repo_id, filename, retry=3):
    from huggingface_hub import hf_hub_download
    model_path = ''
    while retry > 0:
        try:
            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                resume_download=True,
            )
            break
        except:
            logger.error(f"Error downloading model, repo_id: {repo_id}, filename: {filename}, retrying...")
            retry -= 1
    if retry == 0:
        raise Exception("Error downloading model, please try again later.")
    return model_path


class LLaMAClient(BaseLLMModel):
    def __init__(self, model_name, lora_path=None, user_name=""):
        super().__init__(model_name=model_name, user=user_name)
        self.max_generation_token = 1000
        if model_name in MODEL_METADATA:
            path_to_model = download(
                MODEL_METADATA[model_name]["repo_id"],
                MODEL_METADATA[model_name]["filelist"][0],
            )
        else:
            dir_to_model = os.path.join("models", model_name)
            # look for nay .gguf file in the dir_to_model directory and its subdirectories
            path_to_model = None
            for root, dirs, files in os.walk(dir_to_model):
                for file in files:
                    if file.endswith(".gguf"):
                        path_to_model = os.path.join(root, file)
                        break
                if path_to_model is not None:
                    break
        logger.info(f"Loading model from {path_to_model}")
        self.system_prompt = ""
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "Please install llama_cpp by running `pip install llama-cpp-python`"
            )
        if lora_path is not None:
            lora_path = os.path.join("lora", lora_path)
            self.model = Llama(model_path=path_to_model, lora_path=lora_path)
        else:
            self.model = Llama(model_path=path_to_model)

    def _get_llama_style_input(self):
        context = []
        for conv in self.history:
            if conv["role"] == "system":
                context.append(SYS_PREFIX + conv["content"] + SYS_POSTFIX)
            elif conv["role"] == "user":
                context.append(
                    INST_PREFIX + conv["content"] + INST_POSTFIX + OUTPUT_PREFIX
                )
            else:
                context.append(conv["content"] + OUTPUT_POSTFIX)
        return "".join(context)

    def get_answer_at_once(self):
        context = self._get_llama_style_input()
        response = self.model(
            context,
            max_tokens=self.max_generation_token,
            stop=[],
            echo=False,
            stream=False,
        )
        return response, len(response)

    def get_answer_stream_iter(self):
        context = self._get_llama_style_input()
        iter = self.model(
            context,
            max_tokens=self.max_generation_token,
            stop=[SYS_PREFIX, SYS_POSTFIX, INST_PREFIX, OUTPUT_PREFIX, OUTPUT_POSTFIX],
            echo=False,
            stream=True,
        )
        partial_text = ""
        for i in iter:
            response = i["choices"][0]["text"]
            partial_text += response
            yield partial_text
