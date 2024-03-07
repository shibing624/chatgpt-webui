import os
import re
from typing import List, Optional, Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger
from tqdm import tqdm

from src.config import local_embedding, retrieve_proxy, chunk_overlap, chunk_size, hf_emb_model_name
from src import shared
from src.utils import excel_to_string, get_files_hash, load_pkl, save_pkl

pwd_path = os.path.abspath(os.path.dirname(__file__))


class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """Recursive text splitter for Chinese text.
    copy from: https://github.com/chatchat-space/Langchain-Chatchat/tree/master
    """

    def __init__(
            self,
            separators: Optional[List[str]] = None,
            keep_separator: bool = True,
            is_separator_regex: bool = True,
            **kwargs: Any,
    ) -> None:
        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s"
        ]
        self._is_separator_regex = is_separator_regex

    @staticmethod
    def _split_text_with_regex_from_end(
            text: str, separator: str, keep_separator: bool
    ) -> List[str]:
        # Now that we have the separator, split the text
        if separator:
            if keep_separator:
                # The parentheses in the pattern keep the delimiters in the result.
                _splits = re.split(f"({separator})", text)
                splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]
                if len(_splits) % 2 == 1:
                    splits += _splits[-1:]
            else:
                splits = re.split(separator, text)
        else:
            splits = list(text)
        return [s for s in splits if s != ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1:]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = self._split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return [re.sub(r"\n{2,}", "\n", chunk.strip()) for chunk in final_chunks if chunk.strip() != ""]


def get_documents(file_paths):
    text_splitter = ChineseRecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = []
    logger.debug("Loading documents...")
    logger.debug(f"file_paths: {file_paths}")
    for file in file_paths:
        filepath = file.name
        filename = os.path.basename(filepath)
        file_type = os.path.splitext(filename)[1]
        logger.info(f"loading file: {filename}")
        texts = None
        try:
            if file_type == ".pdf":
                import PyPDF2
                logger.debug("Loading PDF...")
                try:
                    from src.pdf_func import parse_pdf
                    from src.config import advance_docs

                    two_column = advance_docs["pdf"].get("two_column", False)
                    pdftext = parse_pdf(filepath, two_column).text
                except:
                    pdftext = ""
                    with open(filepath, "rb") as pdfFileObj:
                        pdfReader = PyPDF2.PdfReader(pdfFileObj)
                        for page in tqdm(pdfReader.pages):
                            pdftext += page.extract_text()
                texts = [Document(page_content=pdftext,
                                  metadata={"source": filepath})]
            elif file_type == ".docx":
                logger.debug("Loading Word...")
                from langchain.document_loaders import UnstructuredWordDocumentLoader
                loader = UnstructuredWordDocumentLoader(filepath)
                texts = loader.load()
            elif file_type == ".pptx":
                logger.debug("Loading PowerPoint...")
                from langchain.document_loaders import UnstructuredPowerPointLoader
                loader = UnstructuredPowerPointLoader(filepath)
                texts = loader.load()
            elif file_type == ".epub":
                logger.debug("Loading EPUB...")
                from langchain.document_loaders import UnstructuredEPubLoader
                loader = UnstructuredEPubLoader(filepath)
                texts = loader.load()
            elif file_type == ".xlsx":
                logger.debug("Loading Excel...")
                text_list = excel_to_string(filepath)
                texts = []
                for elem in text_list:
                    texts.append(Document(page_content=elem,
                                          metadata={"source": filepath}))
            else:
                logger.debug("Loading text file...")
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(filepath, "utf8")
                texts = loader.load()
            logger.debug(f"text size: {len(texts)}, text top3: {texts[:3]}")
        except Exception as e:
            logger.error(f"Error loading file: {filename}, {e}")

        if texts is not None:
            texts = text_splitter.split_documents(texts)
            documents.extend(texts)
    logger.debug(f"Documents loaded. documents size: {len(documents)}, top3: {documents[:3]}")
    return documents


def construct_index(
        api_key,
        files,
        load_from_cache_if_possible=True,
):
    from langchain_community.vectorstores import FAISS
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"
    index_name = get_files_hash(files)
    index_dir = os.path.join(pwd_path, '../index')
    index_path = f"{index_dir}/{index_name}"
    doc_file = f"{index_path}/docs.pkl"
    if local_embedding:
        embeddings = HuggingFaceEmbeddings(model_name=hf_emb_model_name)
    else:
        from langchain_community.embeddings import OpenAIEmbeddings
        if os.environ.get("OPENAI_API_TYPE", "openai") == "openai":
            embeddings = OpenAIEmbeddings(
                openai_api_base=shared.state.openai_api_base,
                openai_api_key=os.environ.get("OPENAI_EMBEDDING_API_KEY", api_key)
            )
        else:
            embeddings = OpenAIEmbeddings(
                deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT_NAME"],
                openai_api_key=os.environ["AZURE_OPENAI_API_KEY"],
                model=os.environ["AZURE_EMBEDDING_MODEL_NAME"],
                openai_api_base=os.environ["AZURE_OPENAI_API_BASE_URL"],
                openai_api_type="azure"
            )
    if os.path.exists(index_path) and load_from_cache_if_possible:
        logger.info("找到了缓存的索引文件，加载中……")
        index = FAISS.load_local(index_path, embeddings)
        documents = load_pkl(doc_file)
        return index, documents
    else:
        try:
            documents = get_documents(files)
            logger.info("构建索引中……")
            with retrieve_proxy():
                index = FAISS.from_documents(documents, embeddings)
            logger.debug("索引构建完成！")
            os.makedirs(index_dir, exist_ok=True)
            index.save_local(index_path)
            logger.debug("索引已保存至本地!")
            save_pkl(documents, doc_file)
            logger.debug("索引文档已保存至本地!")
            return index, documents
        except Exception as e:
            logger.error(f"索引构建失败！error: {e}")
            return None
