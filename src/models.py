# -*- coding: utf-8 -*-
"""
Get model client from model name
"""

import json

import colorama
import gradio as gr
from loguru import logger

from src import config
from src.base_model import ModelType
from src.utils import (
    hide_middle_chars,
    i18n,
)


def get_model(
        model_name,
        lora_model_path=None,
        access_key=None,
        temperature=None,
        top_p=None,
        system_prompt=None,
        user_name="",
        original_model=None,
):
    msg = i18n("模型设置为了：") + f" {model_name}"
    model_type = ModelType.get_type(model_name)
    lora_choices = ["No LoRA"]
    if model_type != ModelType.OpenAI:
        config.local_embedding = True
    model = original_model
    chatbot = gr.Chatbot.update(label=model_name)
    try:
        if model_type == ModelType.OpenAI:
            logger.info(f"正在加载OpenAI模型: {model_name}")
            from src.openai_client import OpenAIClient
            model = OpenAIClient(
                model_name=model_name,
                api_key=access_key,
                system_prompt=system_prompt,
                user_name=user_name,
            )
        elif model_type == ModelType.OpenAIVision:
            logger.info(f"正在加载OpenAI Vision模型: {model_name}")
            from src.openai_client import OpenAIVisionClient
            model = OpenAIVisionClient(model_name, api_key=access_key, user_name=user_name)
        elif model_type == ModelType.ChatGLM:
            logger.info(f"正在加载ChatGLM模型: {model_name}")
            from src.chatglm import ChatGLMClient
            model = ChatGLMClient(model_name, user_name=user_name)
        elif model_type == ModelType.LLaMA:
            logger.info(f"正在加载LLaMA模型: {model_name}")
            from src.llama import LLaMAClient
            model = LLaMAClient(model_name, user_name=user_name)
        elif model_type == ModelType.Unknown:
            raise ValueError(f"未知模型: {model_name}")
    except Exception as e:
        logger.error(e)
    logger.info(msg)
    presudo_key = hide_middle_chars(access_key)
    if original_model is not None and model is not None:
        model.history = original_model.history
        model.history_file_path = original_model.history_file_path
    return model, msg, chatbot, gr.Dropdown.update(choices=lora_choices, visible=False), access_key, presudo_key


if __name__ == "__main__":
    with open("../config.json", "r") as f:
        openai_api_key = json.load(f)["openai_api_key"]
    print('key:', openai_api_key)
    client = get_model(model_name="gpt-3.5-turbo", access_key=openai_api_key)[0]
    chatbot = []
    stream = False
    # 测试账单功能
    logger.info(colorama.Back.GREEN + "测试账单功能" + colorama.Back.RESET)
    logger.info(client.billing_info())
    # 测试问答
    logger.info(colorama.Back.GREEN + "测试问答" + colorama.Back.RESET)
    question = "巴黎是中国的首都吗？"
    for i in client.predict(inputs=question, chatbot=chatbot, stream=stream):
        logger.info(i)
    logger.info(f"测试问答后history : {client.history}")
    # 测试记忆力
    logger.info(colorama.Back.GREEN + "测试记忆力" + colorama.Back.RESET)
    question = "我刚刚问了你什么问题？"
    for i in client.predict(inputs=question, chatbot=chatbot, stream=stream):
        logger.info(i)
    logger.info(f"测试记忆力后history : {client.history}")
    # 测试重试功能
    logger.info(colorama.Back.GREEN + "测试重试功能" + colorama.Back.RESET)
    for i in client.retry(chatbot=chatbot, stream=stream):
        logger.info(i)
    logger.info(f"重试后history : {client.history}")
