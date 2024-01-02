import datetime
import json
import os
import platform

import colorama
import gradio as gr
import requests
from loguru import logger

from src import shared, config
from src.base_model import BaseLLMModel, ModelType
from src.presets import (
    INITIAL_SYSTEM_PROMPT,
    TIMEOUT_ALL,
    TIMEOUT_STREAMING,
    STANDARD_ERROR_MSG,
    CONNECTION_TIMEOUT_MSG,
    READ_TIMEOUT_MSG,
    ERROR_RETRIEVE_MSG,
    GENERAL_ERROR_MSG,
    CHAT_COMPLETION_URL,
    SUMMARY_CHAT_SYSTEM_PROMPT,
)
from src.utils import (
    hide_middle_chars,
    count_token,
    construct_system,
    construct_user,
    get_last_day_of_month,
    i18n,
    replace_special_symbols,
)


class OpenAIClient(BaseLLMModel):
    def __init__(
            self,
            model_name,
            api_key,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            temperature=1.0,
            top_p=1.0,
            user_name="",
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user=user_name,
        )
        self.api_key = api_key
        self.need_api_key = True
        self._refresh_header()

    def get_answer_stream_iter(self):
        if not self.api_key:
            raise ValueError("API key is not set")
        response = self._get_response(stream=True)
        if response is not None:
            iter = self._decode_chat_response(response)
            partial_text = ""
            for i in iter:
                partial_text += i
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        if not self.api_key:
            raise ValueError("API key is not set")
        response = self._get_response()
        response = json.loads(response.text)
        content = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]
        return content, total_token_count

    def count_token(self, user_input):
        input_token_count = count_token(construct_user(user_input))
        if self.system_prompt is not None and len(self.all_token_counts) == 0:
            system_prompt_token_count = count_token(
                construct_system(self.system_prompt)
            )
            return input_token_count + system_prompt_token_count
        return input_token_count

    def billing_info(self):
        try:
            curr_time = datetime.datetime.now()
            last_day_of_month = get_last_day_of_month(
                curr_time).strftime("%Y-%m-%d")
            first_day_of_month = curr_time.replace(day=1).strftime("%Y-%m-%d")
            usage_url = f"{shared.state.usage_api_url}?start_date={first_day_of_month}&end_date={last_day_of_month}"
            try:
                usage_data = self._get_billing_data(usage_url)
            except Exception as e:
                logger.warning(f"获取API使用情况失败:" + str(e))
                return i18n("**获取API使用情况失败**")
            rounded_usage = "{:.5f}".format(usage_data["total_usage"] / 100)
            return i18n("**本月使用金额** ") + f"\u3000 ${rounded_usage}"
        except requests.exceptions.ConnectTimeout:
            status_text = (
                    STANDARD_ERROR_MSG + CONNECTION_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            )
            return status_text
        except requests.exceptions.ReadTimeout:
            status_text = STANDARD_ERROR_MSG + READ_TIMEOUT_MSG + ERROR_RETRIEVE_MSG
            return status_text
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(i18n("获取API使用情况失败:") + str(e))
            return STANDARD_ERROR_MSG + ERROR_RETRIEVE_MSG

    def set_token_upper_limit(self, new_upper_limit):
        pass

    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        openai_api_key = self.api_key
        system_prompt = self.system_prompt
        history = self.history
        logger.debug(f"{history}")
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        if system_prompt is not None:
            history = [construct_system(system_prompt), *history]

        payload = {
            "model": self.model_name,
            "messages": history,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": self.n_choices,
            "stream": stream,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }

        if self.max_generation_token is not None:
            payload["max_tokens"] = self.max_generation_token
        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence
        if self.logit_bias is not None:
            payload["logit_bias"] = self.logit_bias
        if self.user_identifier is not None:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.chat_completion_url != CHAT_COMPLETION_URL:
            logger.info(f"使用自定义API URL: {shared.state.chat_completion_url}")

        with config.retrieve_proxy():
            try:
                response = requests.post(
                    shared.state.chat_completion_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=timeout,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                response = None
        return response

    def _refresh_header(self):
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _get_billing_data(self, billing_url):
        with config.retrieve_proxy():
            response = requests.get(
                billing_url,
                headers=self.headers,
                timeout=TIMEOUT_ALL,
            )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

    def _decode_chat_response(self, response):
        error_msg = ""
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode()
                chunk_length = len(chunk)
                try:
                    chunk = json.loads(chunk[6:])
                except Exception as e:
                    print(i18n("JSON解析错误,收到的内容: ") + f"{chunk}")
                    error_msg += chunk
                    continue
                try:
                    if chunk_length > 6 and "delta" in chunk["choices"][0]:
                        if "finish_reason" in chunk["choices"][0]:
                            finish_reason = chunk["choices"][0]["finish_reason"]
                        else:
                            finish_reason = chunk["finish_reason"]
                        if finish_reason == "stop":
                            break
                        try:
                            yield chunk["choices"][0]["delta"]["content"]
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            continue
                except:
                    print(f"ERROR: {chunk}")
                    continue
        if error_msg and not error_msg == "data: [DONE]":
            raise Exception(error_msg)

    def set_key(self, new_access_key):
        ret = super().set_key(new_access_key)
        self._refresh_header()
        return ret

    def _single_query_at_once(self, history, temperature=1.0):
        timeout = TIMEOUT_ALL
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "temperature": f"{temperature}",
        }
        payload = {
            "model": self.model_name,
            "messages": history,
        }
        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.chat_completion_url != CHAT_COMPLETION_URL:
            logger.debug(f"使用自定义API URL: {shared.state.chat_completion_url}")

        with config.retrieve_proxy():
            response = requests.post(
                shared.state.chat_completion_url,
                headers=headers,
                json=payload,
                stream=False,
                timeout=timeout,
            )

        return response

    def auto_name_chat_history(self, name_chat_method, user_question, chatbot, single_turn_checkbox):
        if len(self.history) == 2 and not single_turn_checkbox and not config.hide_history_when_not_logged_in:
            user_question = self.history[0]["content"]
            if name_chat_method == i18n("模型自动总结（消耗tokens）"):
                ai_answer = self.history[1]["content"]
                try:
                    history = [
                        {"role": "system", "content": SUMMARY_CHAT_SYSTEM_PROMPT},
                        {"role": "user",
                         "content": f"Please write a title based on the following conversation:\n---\nUser: {user_question}\nAssistant: {ai_answer}"}
                    ]
                    response = self._single_query_at_once(history, temperature=0.0)
                    response = json.loads(response.text)
                    content = response["choices"][0]["message"]["content"]
                    filename = replace_special_symbols(content) + ".json"
                except Exception as e:
                    logger.info(f"自动命名失败。{e}")
                    filename = replace_special_symbols(user_question)[:16] + ".json"
                return self.rename_chat_history(filename, chatbot)
            elif name_chat_method == i18n("第一条提问"):
                filename = replace_special_symbols(user_question)[:16] + ".json"
                return self.rename_chat_history(filename, chatbot)
            else:
                return gr.update()
        else:
            return gr.update()


class ChatGLMClient(BaseLLMModel):
    def __init__(self, model_name) -> None:
        super().__init__(model_name=model_name)
        from transformers import AutoTokenizer, AutoModel
        import torch
        global CHATGLM_TOKENIZER, CHATGLM_MODEL
        if CHATGLM_TOKENIZER is None or CHATGLM_MODEL is None:
            system_name = platform.system()
            model_path = None
            if os.path.exists("models"):
                model_dirs = os.listdir("models")
                if model_name in model_dirs:
                    model_path = f"models/{model_name}"
            if model_path is not None:
                model_source = model_path
            else:
                model_source = f"THUDM/{model_name}"
            CHATGLM_TOKENIZER = AutoTokenizer.from_pretrained(
                model_source, trust_remote_code=True
            )
            quantified = False
            if "int4" in model_name:
                quantified = True
            model = AutoModel.from_pretrained(
                model_source, trust_remote_code=True
            )
            if torch.cuda.is_available():
                # run on CUDA
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
            CHATGLM_MODEL = model

    def _get_glm_style_input(self):
        history = [x["content"] for x in self.history]
        query = history.pop()
        logger.debug(f"{history}")
        assert (
                len(history) % 2 == 0
        ), f"History should be even length. current history is: {history}"
        history = [[history[i], history[i + 1]]
                   for i in range(0, len(history), 2)]
        return history, query

    def get_answer_at_once(self):
        history, query = self._get_glm_style_input()
        response, _ = CHATGLM_MODEL.chat(
            CHATGLM_TOKENIZER, query, history=history)
        return response, len(response)

    def get_answer_stream_iter(self):
        history, query = self._get_glm_style_input()
        for response, history in CHATGLM_MODEL.stream_chat(
                CHATGLM_TOKENIZER,
                query,
                history,
                max_length=self.token_upper_limit,
                top_p=self.top_p,
                temperature=self.temperature,
        ):
            yield response


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
    lora_selector_visibility = False
    lora_choices = ["No LoRA"]
    if model_type != ModelType.OpenAI:
        config.local_embedding = True
    model = original_model
    chatbot = gr.Chatbot.update(label=model_name)
    try:
        if model_type == ModelType.OpenAI:
            logger.info(f"正在加载OpenAI模型: {model_name}")
            model = OpenAIClient(
                model_name=model_name,
                api_key=access_key,
                system_prompt=system_prompt,
                user_name=user_name,
            )
            logger.info(f"OpenAI模型加载完成: {model_name}")
        elif model_type == ModelType.ChatGLM:
            logger.info(f"正在加载ChatGLM模型: {model_name}")
            model = ChatGLMClient(model_name)
        elif model_type == ModelType.Unknown:
            raise ValueError(f"未知模型: {model_name}")
    except Exception as e:
        logger.error(e)
    logger.info(msg)
    presudo_key = hide_middle_chars(access_key)
    if original_model is not None and model is not None:
        model.history = original_model.history
        model.history_file_path = original_model.history_file_path
    return model, msg, chatbot, gr.Dropdown.update(choices=lora_choices,
                                                   visible=lora_selector_visibility), access_key, presudo_key


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
