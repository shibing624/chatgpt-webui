from collections import defaultdict
from contextlib import contextmanager
import os
import sys
import commentjson as json

from src import shared, presets
from loguru import logger
pwd_path = os.path.abspath(os.path.dirname(__file__))

# 添加一个统一的config文件
config_file = os.path.join(pwd_path, "../config.json")
config = {}
if os.path.exists(config_file):
    with open(config_file, "r", encoding='utf-8') as f:
        config = json.load(f)
if config:
    logger.info(f"加载配置文件成功, config: {config}")
lang_config = config.get("language", "auto")
language = os.environ.get("LANGUAGE", lang_config)

hide_history_when_not_logged_in = config.get("hide_history_when_not_logged_in", False)

## 处理docker if we are running in Docker
dockerflag = config.get("dockerflag", False)
if os.environ.get("dockerrun") == "yes":
    dockerflag = True

## 处理 api-key 以及 允许的用户列表
my_api_key = config.get("openai_api_key", "")
my_api_key = my_api_key or os.environ.get("OPENAI_API_KEY", "")

xmchat_api_key = config.get("xmchat_api_key", "")
os.environ["XMCHAT_API_KEY"] = xmchat_api_key

minimax_api_key = config.get("minimax_api_key", "")
os.environ["MINIMAX_API_KEY"] = minimax_api_key
minimax_group_id = config.get("minimax_group_id", "")
os.environ["MINIMAX_GROUP_ID"] = minimax_group_id


usage_limit = os.environ.get("USAGE_LIMIT", config.get("usage_limit", 120))

## 多账户机制
multi_api_key = config.get("multi_api_key", False) # 是否开启多账户机制
if multi_api_key:
    api_key_list = config.get("api_key_list", [])
    if len(api_key_list) == 0:
        logger.error("多账号模式已开启，但api_key_list为空，请检查config.json")
        sys.exit(1)
    shared.state.set_api_key_queue(api_key_list)

auth_list = config.get("users", []) # 实际上是使用者的列表
authflag = len(auth_list) > 0  # 是否开启认证的状态值，改为判断auth_list长度

# 处理自定义的api_host，优先读环境变量的配置，如果存在则自动装配
api_host = os.environ.get("OPENAI_API_BASE", config.get("openai_api_base", None))
if api_host is not None:
    shared.state.set_api_host(api_host)

default_chuanhu_assistant_model = config.get("default_chuanhu_assistant_model", "gpt-3.5-turbo")

@contextmanager
def retrieve_openai_api(api_key = None):
    old_api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key is None:
        os.environ["OPENAI_API_KEY"] = my_api_key
        yield my_api_key
    else:
        os.environ["OPENAI_API_KEY"] = api_key
        yield api_key
    os.environ["OPENAI_API_KEY"] = old_api_key

## 处理代理：
http_proxy = config.get("http_proxy", "")
https_proxy = config.get("https_proxy", "")
http_proxy = os.environ.get("HTTP_PROXY", http_proxy)
https_proxy = os.environ.get("HTTPS_PROXY", https_proxy)

# 重置系统变量，在不需要设置的时候不设置环境变量，以免引起全局代理报错
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

local_embedding = config.get("local_embedding", False) # 是否使用本地embedding

@contextmanager
def retrieve_proxy(proxy=None):
    """
    1, 如果proxy = NONE，设置环境变量，并返回最新设置的代理
    2，如果proxy ！= NONE，更新当前的代理配置，但是不更新环境变量
    """
    global http_proxy, https_proxy
    if proxy is not None:
        http_proxy = proxy
        https_proxy = proxy
        yield http_proxy, https_proxy
    else:
        old_var = os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"]
        os.environ["HTTP_PROXY"] = http_proxy
        os.environ["HTTPS_PROXY"] = https_proxy
        yield http_proxy, https_proxy # return new proxy

        # return old proxy
        os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"] = old_var


## 处理advance docs
advance_docs = defaultdict(lambda: defaultdict(dict))
advance_docs.update(config.get("advance_docs", {}))

## 处理gradio.launch参数
server_name = config.get("server_name", None)
server_port = config.get("server_port", None)
if server_name is None:
    if dockerflag:
        server_name = "0.0.0.0"
    else:
        server_name = "127.0.0.1"
if server_port is None:
    if dockerflag:
        server_port = 7860

assert server_port is None or type(server_port) == int, "要求port设置为int类型"

# 设置默认model
default_model = config.get("default_model", "")
try:
    presets.DEFAULT_MODEL = presets.MODELS.index(default_model)
except ValueError:
    pass

share = config.get("share", False)
