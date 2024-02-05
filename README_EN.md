[**🇨🇳中文**](https://github.com/shibing624/chatgpt-webui/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/chatgpt-webui/blob/main/README_EN.md) | [**📖文档/Docs**](https://github.com/shibing624/chatgpt-webui/wiki) | [**🤖模型/Models**](https://huggingface.co/shibing624) 

<div align="center">
  <a href="https://github.com/shibing624/chatgpt-webui">
    <img src="https://github.com/shibing624/chatgpt-webui/blob/main/assets/icon.png" height="100" alt="Logo">
  </a>
</div>

-----------------

# ChatGPT WebUI: ChatGPT webui by python gradio
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/chatgpt-webui.svg)](https://github.com/shibing624/chatgpt-webui/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**chatgpt-webui**: ChatGPT webui using gradio. 为ChatGPT等多种LLM提供了一个轻快好用的Web图形界面

![img](https://github.com/shibing624/chatgpt-webui/blob/main/docs/chat.png)

## ✨ Features
This project is based on [ChuanhuChatGPT](https://github.com/GaiZhenbiao/ChuanhuChatGPT), the main changes are as follows:
1. Simplified the WebUI page, retaining only the core ChatGPT dialogue (LLM) and document retrieval dialogue (RAG) functions, and removing online search/midjournal and other functions;
2. Refactored the code logic and structure, standardized Python syntax, and lightweight project dependency libraries;
3. Keep the local large language model dialogue for easy expansion of the local model;
4. Support nginx reverse proxy, use relative paths for static files, and facilitate deployment.
5. Optimized the online search function, supporting search engines such as DuckDuckGo, Google, Bing, Serper, etc., to improve the accuracy of searches.

## Usage Tips

- To better control the ChatGPT, use System Prompt.
- To use a Prompt Template, select the Prompt Template Collection file first, and then choose certain prompt from the drop-down menu.
- To try again if the response is unsatisfactory, use `🔄 Regenerate` button.
- To start a new line in the input box, press <kbd>Shift</kbd> + <kbd>Enter</kbd> keys.
- To quickly switch between input history, press <kbd>↑</kbd> and <kbd>↓</kbd> key in the input box.
- To deploy the program onto a server, change the last line of the program to `demo.launch(server_name="0.0.0.0", server_port=<your port number>)`.
- To get a public shared link, change the last line of the program to `demo.launch(share=True)`. Please be noted that the program must be running in order to be accessed via a public link.
- To use it in Hugging Face Spaces: It is recommended to **Duplicate Space** and run the program in your own Space for a faster and more secure experience.

## Installation

```shell
git clone https://github.com/shibing624/chatgpt-webui.git
cd chatgpt-webui
pip install -r requirements.txt
```

Then make a copy of `config_example.json`, rename it to `config.json`, and then fill in your API-Key and other settings in the file.

```shell
python main.py
```

A browser window will open and you will be able to chat with ChatGPT.

> **Note**
>
> Please check our [wiki page](https://github.com/shibing624/chatgpt-webui/wiki) for detailed instructions.

## Troubleshooting

When you encounter problems, you should try manually pulling the latest changes of this project first. The steps are as follows:

1. Download the latest code archive by clicking on `Download ZIP` on the webpage, or
   ```shell
   git pull https://github.com/shibing624/chatgpt-webui.git main -f
   ```
2. Try installing the dependencies again (as this project may have introduced new dependencies)
   ```
   pip install -r requirements.txt
   ```
3. Update Gradio
   ```
   pip install gradio --upgrade --force-reinstall
   ```

Generally, you can solve most problems by following these steps.

If the problem still exists, please refer to this page: [Frequently Asked Questions (FAQ)](https://github.com/shibing624/chatgpt-webui/wiki/常见问题)

This page lists almost all the possible problems and solutions. Please read it carefully.
