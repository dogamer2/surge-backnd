import asyncio
import json
import threading
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


class IconFinderService:
    def __init__(self):
        self.collection_name = "icons"
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize_icons_collection(self):
        self.client = chromadb.PersistentClient(
            path="chroma", settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = ONNXMiniLM_L6_V2()
        self.embedding_function.DOWNLOAD_PATH = "chroma/models"
        self.embedding_function._download_model_if_not_exists()
        try:
            self.collection = self.client.get_collection(
                self.collection_name, embedding_function=self.embedding_function
            )
        except Exception:
            with open("assets/icons.json", "r") as f:
                icons = json.load(f)

            documents = []
            ids = []

            for i, each in enumerate(icons["icons"]):
                if each["name"].split("-")[-1] == "bold":
                    doc_text = f"{each['name']} {each['tags']}"
                    documents.append(doc_text)
                    ids.append(each["name"])

            if documents:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_function,
                    metadata={"hnsw:space": "cosine"},
                )
                self.collection.add(documents=documents, ids=ids)

    def _ensure_initialized(self):
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return
            print("Initializing icons collection...")
            self._initialize_icons_collection()
            self._initialized = True
            print("Icons collection initialized.")

    async def search_icons(self, query: str, k: int = 1):
        await asyncio.to_thread(self._ensure_initialized)
        result = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query],
            n_results=k,
        )
        return [f"/static/icons/bold/{each}.svg" for each in result["ids"][0]]


ICON_FINDER_SERVICE = IconFinderService()
