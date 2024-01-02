import os

from loguru import logger
from tqdm import tqdm

from src.config import local_embedding, retrieve_proxy, chunk_overlap, chunk_size, hf_emb_model_name
from src.presets import OPENAI_API_BASE
from src.utils import excel_to_string, get_files_hash

pwd_path = os.path.abspath(os.path.dirname(__file__))


def get_documents(file_paths):
    import PyPDF2
    from langchain.schema import Document
    from langchain.text_splitter import TokenTextSplitter
    text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

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
                from langchain.document_loaders import TextLoader
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
        max_input_size=4096,
        num_outputs=5,
        max_chunk_overlap=20,
        chunk_size_limit=600,
        embedding_limit=None,
        separator=" ",
        load_from_cache_if_possible=True,
):
    from langchain.vectorstores import FAISS
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        os.environ["OPENAI_API_KEY"] = "sk-xxxxxxx"
    index_name = get_files_hash(files)
    index_dir = os.path.join(pwd_path, '../index')
    index_path = f"{index_dir}/{index_name}"
    if local_embedding:
        embeddings = HuggingFaceEmbeddings(model_name=hf_emb_model_name)
    else:
        from langchain.embeddings import OpenAIEmbeddings
        if os.environ.get("OPENAI_API_TYPE", "openai") == "openai":
            openai_api_base = os.environ.get("OPENAI_API_BASE", OPENAI_API_BASE)
            embeddings = OpenAIEmbeddings(
                openai_api_base=openai_api_base,
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
        return FAISS.load_local(index_path, embeddings)
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
            return index
        except Exception as e:
            logger.error(f"索引构建失败！error: {e}")
            return None
