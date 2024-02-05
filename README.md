[**🇨🇳中文**](https://github.com/shibing624/chatgpt-webui/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/chatgpt-webui/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/chatgpt-webui/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/chatgpt-webui">
    <img src="https://github.com/shibing624/chatgpt-webui/blob/main/assets/icon.png" height="100" alt="Logo">
  </a>
</div>

-----------------

# ChatGPT WebUI: ChatGPT webui using gradio
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/chatgpt-webui.svg)](https://github.com/shibing624/chatgpt-webui/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**chatgpt-webui**: ChatGPT webui using gradio. 为ChatGPT等多种LLM提供了一个轻快好用的Web图形界面

![img](https://github.com/shibing624/chatgpt-webui/blob/main/docs/chat.png)

## ✨ Features
本项目基于 [ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT) 简化而来，主要改动如下：
1. 简化了WebUI页面，只保留核心的ChatGPT对话(LLM)、文档检索对话（RAG）功能，去除了midjourney等功能
2. 重构了代码逻辑和结构，规范python语法，轻量化项目依赖库
3. 保留本地大语言模型对话，方便扩展本地模型
4. 支持nginx反向代理，静态文件使用相对路径，方便部署
5. 提升RAG检索准确率
   - 引入jieba分词的rank_BM25，提升对query关键词的字面匹配，使用字面相似度+sentence embedding向量相似度加权获取corpus候选集
   - RAG底模优化，使用基于RAG微调的模型
   - Chinese chunk切分优化
6. 优化了在线搜索功能，支持DuckDuckGo、Google、Bing、Serper等搜索引擎，提升搜索准确率

## 支持模型

- [ChatGPT(GPT-4)](https://chat.openai.com) 
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [ChatGLM3-6b](https://github.com/THUDM/ChatGLM3)
- [Chinese-LLaMA-Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [YI](https://github.com/01-ai/Yi)
## 使用技巧

### 💪 强力功能
- **GPT4 API**：基于代理配置GPT4 API，支持多用户对话；
- **本地部署LLM**：一键部署，获取属于你自己的大语言模型。

### 🤖 System Prompt
- 通过 System Prompt 设定前提条件，可以很有效地进行角色扮演；
- 川虎Chat 预设了Prompt模板，点击`加载Prompt模板`，先选择 Prompt 模板集合，然后在下方选择想要的 Prompt。

### 💬 基础对话
- 如果回答不满意，可以使用 `重新生成` 按钮再试一次，或者直接 `删除这轮对话`;
- 输入框支持换行，按 <kbd>Shift</kbd> + <kbd>Enter</kbd>即可；
- 在输入框按 <kbd>↑</kbd> <kbd>↓</kbd> 方向键，可以在发送记录中快速切换；
- 每次新建一个对话太麻烦，试试 `单论对话` 功能；
- 回答气泡旁边的小按钮，不仅能 `一键复制`，还能 `查看Markdown原文`；
- 指定回答语言，让 ChatGPT 固定以某种语言回答。

### 📜 对话历史
- 对话历史记录会被自动保存，不用担心问完之后找不到了；
- 多用户历史记录隔离，除了你都看不到；
- 重命名历史记录，方便日后查找；
- <sup>New!</sup> 魔法般自动命名历史记录，让 LLM 理解对话内容，帮你自动为历史记录命名！
- <sup>New!</sup> 搜索历史记录，支持正则表达式！

### 🖼️ 小而美的体验
- 自研 Small-and-Beautiful 主题，带给你小而美的体验；
- 自动亮暗色切换，给你从早到晚的舒适体验；
- 完美渲染 LaTeX / 表格 / 代码块，支持代码高亮；
- <sup>New!</sup> 非线性动画、毛玻璃效果，精致得不像 Gradio！
- <sup>New!</sup> 适配 Windows / macOS / Linux / iOS / Android，从图标到全面屏适配，给你最合适的体验！
- <sup>New!</sup> 支持以 PWA应用程序 安装，体验更加原生！

### 👨‍💻 极客功能
- 大量 LLM 参数可调；
- 支持更换 api-host；
- 支持自定义代理；
- 支持多 api-key 负载均衡。

### ⚒️ 部署相关
- 部署到服务器：在 `config.json` 中设置 `"server_name": "0.0.0.0", "server_port": <你的端口号>,`。
- 获取公共链接：在 `config.json` 中设置 `"share": true,`。注意程序必须在运行，才能通过公共链接访问。
- 在Hugging Face上使用：建议在右上角 **复制Space** 再使用，这样App反应可能会快一点。

## 快速上手

在终端执行以下命令：

```shell
git clone https://github.com/shibing624/chatgpt-webui.git
cd chatgpt-webui
pip install -r requirements.txt
```

然后，在项目文件夹中复制一份 `config_example.json`，并将其重命名为 `config.json`，在其中填入 `API-Key` 等设置。

```shell
python main.py
```

一个浏览器窗口将会自动打开，可以与ChatGPT或其他模型进行对话。


## 疑难杂症解决

在遇到各种问题查阅相关信息前，您可以先尝试 **手动拉取本项目的最新更改<sup>1</sup>** 并 **更新依赖库<sup>2</sup>**，然后重试。步骤为：

1. 点击网页上的 `Download ZIP` 按钮，下载最新代码并解压覆盖，或
   ```shell
   git pull https://github.com/shibing624/chatgpt-webui.git main -f
   ```
2. 尝试再次安装依赖（可能本项目引入了新的依赖）
   ```
   pip install -r requirements.txt
   ```

很多时候，这样就可以解决问题。

如果问题仍然存在，请查阅该页面：[常见问题](https://github.com/shibing624/chatgpt-webui/wiki/常见问题)

该页面列出了**几乎所有**您可能遇到的各种问题，包括如何配置代理，以及遇到问题后您该采取的措施，**请务必认真阅读**。

## Contact

- Issue(建议)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/chatgpt-webui.svg)](https://github.com/shibing624/chatgpt-webui/issues)
- 邮件我：xuming: xuming624@qq.com
- 微信我： 加我*微信号：xuming624, 备注：姓名-公司名-NLP* 进NLP交流群（加我拉你进群）。

<img src="https://github.com/shibing624/chatgpt-webui/blob/main/docs/wechat.jpeg" width="200" />

<img src="https://github.com/shibing624/chatgpt-webui/blob/main/docs/wechat_group.png" width="200" />
