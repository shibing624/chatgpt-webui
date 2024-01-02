import os
from collections import namedtuple

import gradio as gr
from gradio.utils import validate_url
from gradio_client import utils as client_utils

from src.presets import chuanhu_path, assets_path
from src.utils import convert_bot_before_marked, convert_user_before_marked


def postprocess(
        self,
        y,
):
    """
    Parameters:
        y: List of lists representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.  It can also be a tuple whose first element is a string filepath or URL to an image/video/audio, and second (optional) element is the alt text, in which case the media file is displayed. It can also be None, in which case that message is not displayed.
    Returns:
        List of lists representing the message and response. Each message and response will be a string of HTML, or a dictionary with media information. Or None if the message is not to be displayed.
    """
    if y is None:
        return []
    processed_messages = []
    for message_pair in y:
        assert isinstance(
            message_pair, (tuple, list)
        ), f"Expected a list of lists or list of tuples. Received: {message_pair}"
        assert (
                len(message_pair) == 2
        ), f"Expected a list of lists of length 2 or list of tuples of length 2. Received: {message_pair}"

        processed_messages.append(
            [
                self._postprocess_chat_messages(message_pair[0], "user"),
                self._postprocess_chat_messages(message_pair[1], "bot"),
            ]
        )
    return processed_messages


def postprocess_chat_messages(
        self, chat_message, role: str
):
    if chat_message is None:
        return None
    elif isinstance(chat_message, (tuple, list)):
        file_uri = chat_message[0]
        if validate_url(file_uri):
            filepath = file_uri
        else:
            filepath = self.make_temp_copy_if_needed(file_uri)

        mime_type = client_utils.get_mimetype(filepath)
        return {
            "name": filepath,
            "mime_type": mime_type,
            "alt_text": chat_message[1] if len(chat_message) > 1 else None,
            "data": None,  # These last two fields are filled in by the frontend
            "is_file": True,
        }
    elif isinstance(chat_message, str):
        # chat_message = inspect.cleandoc(chat_message)
        # escape html spaces
        # chat_message = chat_message.replace(" ", "&nbsp;")
        if role == "bot":
            chat_message = convert_bot_before_marked(chat_message)
        elif role == "user":
            chat_message = convert_user_before_marked(chat_message)
        return chat_message
    else:
        raise ValueError(f"Invalid message for Chatbot component: {chat_message}")


def add_classes_to_gradio_component(comp):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    code from stable-diffusion-webui <AUTOMATIC1111/stable-diffusion-webui>
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(comp.elem_classes or [])]

    if getattr(comp, 'multiselect', False):
        comp.elem_classes.append('multiselect')


def IOComponent_init(self, *args, **kwargs):
    res = original_IOComponent_init(self, *args, **kwargs)
    add_classes_to_gradio_component(self)

    return res


original_IOComponent_init = gr.components.IOComponent.__init__
gr.components.IOComponent.__init__ = IOComponent_init


def BlockContext_init(self, *args, **kwargs):
    res = original_BlockContext_init(self, *args, **kwargs)
    add_classes_to_gradio_component(self)

    return res


original_BlockContext_init = gr.blocks.BlockContext.__init__
gr.blocks.BlockContext.__init__ = BlockContext_init


def get_html(filename):
    path = os.path.join(chuanhu_path, "assets", "html", filename)
    if os.path.exists(path):
        with open(path, encoding="utf8") as file:
            return file.read()
    return ""


def webpath(fn):
    if fn.startswith(assets_path):
        web_path = os.path.relpath(fn, chuanhu_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)
    return f'file={web_path}?{os.path.getmtime(fn)}'


ScriptFile = namedtuple("ScriptFile", ["basedir", "filename", "path"])


def list_scripts(scriptdirname, extension):
    scripts_list = []
    scripts_dir = os.path.join(chuanhu_path, "assets", scriptdirname)
    if os.path.exists(scripts_dir):
        for filename in sorted(os.listdir(scripts_dir)):
            scripts_list.append(ScriptFile(assets_path, filename, os.path.join(scripts_dir, filename)))
    scripts_list = [x for x in scripts_list if
                    os.path.splitext(x.path)[1].lower() == extension and os.path.isfile(x.path)]
    return scripts_list


def javascript_html():
    head = ""
    for script in list_scripts("javascript", ".js"):
        head += f'<script type="text/javascript" src="{webpath(script.path)}"></script>\n'
    for script in list_scripts("javascript", ".mjs"):
        head += f'<script type="module" src="{webpath(script.path)}"></script>\n'
    return head


def css_html():
    head = ""
    for cssfile in list_scripts("stylesheet", ".css"):
        head += f'<link rel="stylesheet" property="stylesheet" href="{webpath(cssfile.path)}">'
    return head


def reload_javascript():
    js = javascript_html()
    js += '<script async type="module" src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>'
    js += '<script async type="module" src="https://spin.js.org/spin.umd.js"></script><link type="text/css" href="https://spin.js.org/spin.css" rel="stylesheet" />'
    js += '<script async src="https://cdn.jsdelivr.net/npm/@fancyapps/ui@5.0/dist/fancybox/fancybox.umd.js"></script><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@fancyapps/ui@5.0/dist/fancybox/fancybox.css" />'

    meta = """
        <meta name="apple-mobile-web-app-title" content="ChatGPT-WebUI">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="application-name" content="ChatGPT-WebUI">
        <meta name='viewport' content='width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover'>
        <meta name="theme-color" content="#ffffff">
    """
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{meta}{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
