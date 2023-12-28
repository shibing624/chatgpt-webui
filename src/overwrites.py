import gradio as gr
from gradio.utils import validate_url
from gradio_client import utils as client_utils

from src.presets import custom_js_path, external_js_path
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


with open(custom_js_path, "r", encoding="utf-8") as f, \
        open(external_js_path, "r", encoding="utf-8") as f1:
    customJS = f.read()
    externalScripts = f1.read()


def reload_javascript():
    print("Reloading javascript...")
    js = f'<script>{customJS}</script><script async>{externalScripts}</script>'

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</html>', f'{js}</html>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
