import os
import sys
from collections import defaultdict
from contextlib import contextmanager

import commentjson as json
from loguru import logger

from src import shared, presets

pwd_path = os.path.abspath(os.path.dirname(__file__))

# 添加一个统一的config文件
config_file = os.path.join(pwd_path, "../config.json")
config = {}
if os.path.exists(config_file):
    with open(config_file, "r", encoding='utf-8') as f:
        config = json.load(f)
if config:
    logger.info(f"加载配置文件成功, config: {config}")
language = config.get("language", "") or os.environ.get("LANGUAGE", "auto")

hide_history_when_not_logged_in = config.get("hide_history_when_not_logged_in", False)
show_api_billing = config.get("show_api_billing", False)
# 选择对话名称的方法。0: 使用日期时间命名；1: 使用第一条提问命名，2: 使用模型自动总结
chat_name_method_index = config.get("chat_name_method_index", 2)

hide_local_models = config.get("hide_local_models", False)
if hide_local_models:
    presets.MODELS = presets.ONLINE_MODELS
    logger.info(f"已设置隐藏本地模型，可用模型：{presets.MODELS}")
else:
    local_models = config.get("local_models", None)
    if local_models:
        presets.LOCAL_MODELS = local_models
        logger.info(f"已设置本地模型：{local_models}")
        presets.MODELS = presets.ONLINE_MODELS + list(presets.LOCAL_MODELS.keys())
if "available_models" in config:
    presets.MODELS = config["available_models"]
    logger.info(f"已设置可用模型：{config['available_models']}")

# 处理docker if we are running in Docker
dockerflag = config.get("dockerflag", False)

xmchat_api_key = config.get("xmchat_api_key", "")
minimax_api_key = config.get("minimax_api_key", "")
minimax_group_id = config.get("minimax_group_id", "")

usage_limit = config.get("usage_limit", 120)

# 多账户机制
multi_api_key = config.get("multi_api_key", False)  # 是否开启多账户机制
if multi_api_key:
    api_key_list = config.get("api_key_list", [])
    if len(api_key_list) == 0:
        logger.error("多账号模式已开启，但api_key_list为空，请检查config.json")
        sys.exit(1)
    shared.state.set_api_key_queue(api_key_list)

auth_list = config.get("users", [])  # 实际上是使用者的列表
authflag = len(auth_list) > 0  # 是否开启认证的状态值，改为判断auth_list长度

# 处理自定义的api_host，优先读环境变量的配置，如果存在则自动装配
api_host = config.get("openai_api_base", None)
if api_host is not None:
    shared.state.set_api_host(api_host)
    logger.info(f"OpenAI API Base set to: {shared.state.openai_api_base}")
# 处理 api-key 以及 允许的用户列表
my_api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))

@contextmanager
def retrieve_openai_api(api_key=None):
    if api_key is None:
        yield my_api_key
    else:
        yield api_key


# 处理代理：
http_proxy = config.get("http_proxy", "")
https_proxy = config.get("https_proxy", "")
http_proxy = os.environ.get("HTTP_PROXY", http_proxy)
https_proxy = os.environ.get("HTTPS_PROXY", https_proxy)

# 重置系统变量，在不需要设置的时候不设置环境变量，以免引起全局代理报错
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""

local_embedding = config.get("local_embedding", False)  # 是否使用本地embedding
chunk_size = config.get("chunk_size", 500)
chunk_overlap = config.get("chunk_overlap", 50)
hf_emb_model_name = config.get("hf_emb_model_name", "shibing624/text2vec-base-multilingual")


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
        yield http_proxy, https_proxy  # return new proxy

        # return old proxy
        os.environ["HTTP_PROXY"], os.environ["HTTPS_PROXY"] = old_var


# 处理latex options
user_latex_option = config.get("latex_option", "default")
if user_latex_option == "default":
    latex_delimiters_set = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]
elif user_latex_option == "strict":
    latex_delimiters_set = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]
elif user_latex_option == "all":
    latex_delimiters_set = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
        {"left": "\\begin{equation}", "right": "\\end{equation}", "display": True},
        {"left": "\\begin{align}", "right": "\\end{align}", "display": True},
        {"left": "\\begin{alignat}", "right": "\\end{alignat}", "display": True},
        {"left": "\\begin{gather}", "right": "\\end{gather}", "display": True},
        {"left": "\\begin{CD}", "right": "\\end{CD}", "display": True},
    ]
elif user_latex_option == "disabled":
    latex_delimiters_set = []
else:
    latex_delimiters_set = [
        {"left": "$$", "right": "$$", "display": True},
        {"left": "$", "right": "$", "display": False},
        {"left": "\\(", "right": "\\)", "display": False},
        {"left": "\\[", "right": "\\]", "display": True},
    ]

# 处理advance docs
advance_docs = defaultdict(lambda: defaultdict(dict))
advance_docs.update(config.get("advance_docs", {}))


def update_doc_config(two_column_pdf):
    global advance_docs
    advance_docs["pdf"]["two_column"] = two_column_pdf


# 处理gradio.launch参数
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
    logger.error("你填写的默认模型" + default_model + "不存在！请从下面的列表中挑一个填写：" + str(presets.MODELS))

share = config.get("share", False)

# avatar
bot_avatar = config.get("bot_avatar", "default")
user_avatar = config.get("user_avatar", "default")
if bot_avatar == "" or bot_avatar == "none" or bot_avatar is None:
    bot_avatar = None
elif bot_avatar == "default":
    bot_avatar = os.path.join(pwd_path, "../assets/chatbot.png")
if user_avatar == "" or user_avatar == "none" or user_avatar is None:
    user_avatar = None
elif user_avatar == "default":
    user_avatar = os.path.join(pwd_path, "../assets/user.png")

websearch_engine = config.get("websearch_engine", "duckduckgo")
# 设置websearch engine api key
bing_search_api_key = config.get("bing_search_api_key", "") or os.environ.get("BING_SEARCH_API_KEY", "")
google_search_api_key = config.get("google_search_api_key", "") or os.environ.get("GOOGLE_SEARCH_API_KEY", "")
google_search_cx = config.get("google_search_cx", "") or os.environ.get("GOOGLE_SEARCH_CX", "")
serper_search_api_key = config.get("serper_search_api_key", "") or os.environ.get("SERPER_SEARCH_API_KEY", "")
searchapi_api_key = config.get("searchapi_api_key", "") or os.environ.get("SEARCHAPI_API_KEY", "")

autobrowser = config.get("autobrowser", True)
