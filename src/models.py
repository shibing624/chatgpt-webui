# -*- coding: utf-8 -*-
"""
Get model client from model name
"""

import base64
import datetime
import json
import os
from io import BytesIO

import colorama
import gradio as gr
import requests
from PIL import Image
from loguru import logger

from src import shared, config
from src.base_model import BaseLLMModel, ModelType
from src.chatglm import ChatGLMClient
from src.llama import LLaMAClient
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
    SUMMARY_CHAT_SYSTEM_PROMPT
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


def decode_chat_response(response):
    try:
        error_msg = ""
        for chunk in response.iter_lines():
            if chunk:
                try:
                    chunk = chunk.decode('utf-8')
                    chunk_length = len(chunk)
                    chunk = json.loads(chunk[6:])

                    if chunk_length > 6 and "delta" in chunk["choices"][0]:
                        if "finish_reason" in chunk["choices"][0]:
                            finish_reason = chunk["choices"][0]["finish_reason"]
                        else:
                            finish_reason = chunk["finish_reason"]
                        if finish_reason == "stop":
                            break
                        yield chunk["choices"][0]["delta"]["content"]
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {str(e)}, content received: {chunk}")
                    error_msg += chunk
                    continue
                except Exception as e:
                    logger.error(f"Error while processing chunk: {str(e)}, Content: {chunk}")
                    continue
        if error_msg and not error_msg.endswith("[DONE]"):
            raise Exception(error_msg)
    except GeneratorExit:
        logger.warning("decode_chat_response: GeneratorExit")
    except Exception as e:
        logger.error(f"Error in decode_chat_response: {str(e)}")


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
            iter = decode_chat_response(response)
            partial_text = ""
            try:
                for i in iter:
                    partial_text += i
                    yield partial_text
            except Exception as e:
                logger.error(f"Error while generating answers: {str(e)}")
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
        if self.user_identifier:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.chat_completion_url != CHAT_COMPLETION_URL:
            logger.debug(f"使用自定义API URL: {shared.state.chat_completion_url}")

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


class OpenAIVisionClient(BaseLLMModel):
    def __init__(
            self,
            model_name,
            api_key,
            system_prompt=INITIAL_SYSTEM_PROMPT,
            temperature=1.0,
            top_p=1.0,
            user_name=""
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            system_prompt=system_prompt,
            user=user_name
        )
        self.api_key = api_key
        self.need_api_key = True
        self.max_generation_token = 4096
        self.images = []
        self._refresh_header()

    def get_answer_stream_iter(self):
        response = self._get_response(stream=True)
        if response is not None:
            iter = decode_chat_response(response)
            partial_text = ""
            try:
                for i in iter:
                    partial_text += i
                    yield partial_text
            except Exception as e:
                logger.error(f"Error while generating answers: {str(e)}")
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def get_answer_at_once(self):
        response = self._get_response()
        response = json.loads(response.text)
        content = response["choices"][0]["message"]["content"]
        total_token_count = response["usage"]["total_tokens"]
        return content, total_token_count

    def try_read_image(self, filepath):
        def is_image_file(filepath):
            # 判断文件是否为图片
            valid_image_extensions = [
                ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
            file_extension = os.path.splitext(filepath)[1].lower()
            return file_extension in valid_image_extensions

        def image_to_base64(image_path):
            # 打开并加载图片
            img = Image.open(image_path)

            # 获取图片的宽度和高度
            width, height = img.size

            # 计算压缩比例，以确保最长边小于4096像素
            max_dimension = 2048
            scale_ratio = min(max_dimension / width, max_dimension / height)

            if scale_ratio < 1:
                # 按压缩比例调整图片大小
                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # 将图片转换为jpg格式的二进制数据
            buffer = BytesIO()
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(buffer, format='JPEG')
            binary_image = buffer.getvalue()

            # 对二进制数据进行Base64编码
            base64_image = base64.b64encode(binary_image).decode('utf-8')

            return base64_image

        if is_image_file(filepath):
            logger.info(f"读取图片文件: {filepath}")
            base64_image = image_to_base64(filepath)
            self.images.append({
                "path": filepath,
                "base64": base64_image,
            })

    def handle_file_upload(self, files, chatbot, language):
        """if the model accepts multi modal input, implement this function"""
        if files:
            for file in files:
                if file.name:
                    self.try_read_image(file.name)
        if self.images is not None:
            chatbot = chatbot + [([image["path"] for image in self.images], None)]
        return None, chatbot, None

    def prepare_inputs(self, real_inputs, use_websearch, files, reply_language, chatbot,
                       load_from_cache_if_possible=True):
        fake_inputs = real_inputs
        display_append = ""
        limited_context = False
        return limited_context, fake_inputs, display_append, real_inputs, chatbot

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

    @shared.state.switching_api_key  # 在不开启多账号模式的时候，这个装饰器不会起作用
    def _get_response(self, stream=False):
        openai_api_key = self.api_key
        system_prompt = self.system_prompt
        history = self.history
        if self.images:
            self.history[-1]["content"] = [
                {"type": "text", "text": self.history[-1]["content"]},
                *[{"type": "image_url", "image_url": "data:image/jpeg;base64," + image["base64"]} for image in
                  self.images]
            ]
            self.images = []
        logger.debug(colorama.Fore.YELLOW +
                     f"{history}" + colorama.Fore.RESET)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_api_key}",
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
            "max_tokens": 4096
        }

        if self.stop_sequence is not None:
            payload["stop"] = self.stop_sequence
        if self.logit_bias is not None:
            payload["logit_bias"] = self.encoded_logit_bias()
        if self.user_identifier:
            payload["user"] = self.user_identifier

        if stream:
            timeout = TIMEOUT_STREAMING
        else:
            timeout = TIMEOUT_ALL

        # 如果有自定义的api-host，使用自定义host发送请求，否则使用默认设置发送请求
        if shared.state.chat_completion_url != CHAT_COMPLETION_URL:
            logger.debug(f"使用自定义API URL: {shared.state.chat_completion_url}")

        with config.retrieve_proxy():
            try:
                response = requests.post(
                    shared.state.chat_completion_url,
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=timeout,
                )
            except:
                return None
        return response

    def _refresh_header(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
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
            model = OpenAIClient(
                model_name=model_name,
                api_key=access_key,
                system_prompt=system_prompt,
                user_name=user_name,
            )
            logger.info(f"OpenAI模型加载完成: {model_name}")
        elif model_type == ModelType.OpenAIVision:
            logger.info(f"正在加载OpenAI Vision模型: {model_name}")
            access_key = os.environ.get("OPENAI_API_KEY", access_key)
            model = OpenAIVisionClient(
                model_name, api_key=access_key, user_name=user_name)
        elif model_type == ModelType.ChatGLM:
            logger.info(f"正在加载ChatGLM模型: {model_name}")
            model = ChatGLMClient(model_name, user_name=user_name)
        elif model_type == ModelType.LLaMA:
            logger.info(f"正在加载LLaMA模型: {model_name}")
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
