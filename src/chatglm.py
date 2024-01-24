# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import platform

from loguru import logger

from src.base_model import BaseLLMModel
from src.presets import LOCAL_MODELS


class ChatGLMClient(BaseLLMModel):
    def __init__(self, model_name, user_name=""):
        super().__init__(model_name=model_name, user=user_name)
        import torch
        from transformers import AutoModel, AutoTokenizer
        system_name = platform.system()
        logger.info(f"Loading model from {model_name}")
        if model_name in LOCAL_MODELS:
            model_path = LOCAL_MODELS[model_name]
        else:
            model_path = model_name
        self.CHATGLM_TOKENIZER = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        quantified = False
        if "int4" in model_name:
            quantified = True
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map='auto', torch_dtype='auto')
        if torch.cuda.is_available():
            logger.info("CUDA is available, using CUDA")
            model = model.half().cuda()
        # mps加速还存在一些问题，暂时不使用
        elif system_name == "Darwin" and model_path is not None and not quantified:
            logger.info("Running on macOS, using MPS")
            # running on macOS and model already downloaded
            model = model.half().to("mps")
        else:
            logger.info("GPU is not available, using CPU")
            model = model.float()
        model = model.eval()
        logger.info(f"Model loaded from {model_path}")
        self.CHATGLM_MODEL = model

    def _get_glm3_style_input(self):
        history = self.history
        query = history.pop()["content"]
        return history, query

    def _get_glm2_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        logger.debug(f"{history}")
        assert len(history) % 2 == 0, f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]]
                   for i in range(0, len(history), 2)]
        return history, query

    def _get_glm_style_input(self):
        if "glm2" in self.model_name:
            return self._get_glm2_style_input()
        else:
            return self._get_glm3_style_input()

    def get_answer_at_once(self):
        history, query = self._get_glm_style_input()
        logger.debug(f"{history}")
        response, _ = self.CHATGLM_MODEL.chat(
            self.CHATGLM_TOKENIZER, query, history=history)
        return response, len(response)

    def get_answer_stream_iter(self):
        history, query = self._get_glm_style_input()
        logger.debug(f"{history}")
        for response, history in self.CHATGLM_MODEL.stream_chat(
                self.CHATGLM_TOKENIZER,
                query,
                history,
                max_length=self.token_upper_limit,
                top_p=self.top_p,
                temperature=self.temperature,
        ):
            yield response
