import json

import requests
from loguru import logger

from modules.models.base_model import BaseLLMModel
from modules.presets import STANDARD_ERROR_MSG, GENERAL_ERROR_MSG, i18n


class MOSS_Client(BaseLLMModel):
    def __init__(self, model_name='', user_name="", system_prompt=None):
        super().__init__(model_name=model_name, user=user_name)
        self.headers = {
            'Authorization': "Bearer sk-3AmBmgGILenGk3N2514a7e3004354870Ac1aE0A45a1a7e64",
            'Content-Type': 'application/json',
        }
        self.url = 'https://apejhvxcd.cloud.sealos.io/v1/chat/completions'
        self.history = []
        self.system_prompt = system_prompt

    def request_gpt_wandou(self, prompt):
        json_data = {
            # 'model': 'gpt-4',
            'model': 'gpt-3.5-turbo',
            'messages': [
                {
                    'role': 'user',
                    'content': prompt,
                },
            ],
            'stream': False,
        }

        response = requests.post('https://apejhvxcd.cloud.sealos.io/v1/chat/completions',
                                 headers=self.headers,
                                 json=json_data)
        logger.info(response)
        logger.info(response.text)
        result = response.json()
        return result['choices'][0]['message']['content']

    def get_answer_at_once(self):
        request_body = {
            "model": 'gpt-4',
            'messages': [{"role": "user", "content": self.history[-1]['content']}],
            'stream': False,
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=request_body)
            result = response.json()
            response = result['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error: {e}")
            response = str(e)
        return response, len(response)

    def get_answer_stream_iter(self):
        messages = []
        for msg in self.history:
            if msg['role'] == 'user':
                messages.append({"role": "user", "content": msg['content']})
            else:
                messages.append({"role": "assistant", "content": msg['content']})

        request_body = {
            "model": 'gpt-4',
            'messages': messages,
            'stream': True,
        }

        try:
            response = requests.post(
                self.url,
                headers=self.headers,
                json=request_body,
            )
            logger.debug(f"response: {response}")
        except Exception as e:
            logger.error(f"Error: {e}")
            response = None

        if response is not None:
            iter = self._decode_chat_response(response)
            partial_text = ""
            for i in iter:
                partial_text += i
                yield partial_text
        else:
            yield STANDARD_ERROR_MSG + GENERAL_ERROR_MSG

    def _decode_chat_response(self, response):
        error_msg = ""
        c = 0
        for chunk in response.iter_lines():
            if chunk:
                chunk = chunk.decode()
                chunk_length = len(chunk)
                print(chunk)
                c += 1
                try:
                    chunk = json.loads(chunk[6:])
                except json.JSONDecodeError:
                    print(i18n("JSON解析错误,收到的内容: ") + f"{chunk}")
                    error_msg += chunk
                    continue
                if chunk_length > 6 and "delta" in chunk["choices"][0]:
                    if "finish_reason" in chunk["choices"][0] and chunk["choices"][0]["finish_reason"] == "stop":
                        self.all_token_counts.append(c)
                        break
                    try:
                        yield chunk["choices"][0]["delta"]['content']
                    except Exception as e:
                        logger.error(f"Error: {e}")
                        continue
        if error_msg:
            try:
                error_msg = json.loads(error_msg)
                if 'base_resp' in error_msg:
                    status_code = error_msg['base_resp']['status_code']
                    status_msg = error_msg['base_resp']['status_msg']
                    raise Exception(f"{status_code} - {status_msg}")
            except json.JSONDecodeError:
                pass
            raise Exception(error_msg)


if __name__ == "__main__":
    model = MOSS_Client()
    r = model.request_gpt_wandou('南极为啥没有北极熊')
    print(r)
