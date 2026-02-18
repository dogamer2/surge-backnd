import asyncio
import json
from pathlib import Path


class IconFinderService:
    def __init__(self):
        self.collection_name = "icons"
        self.client = None
        self.collection = None
        self.embedding_function = None
        self._chromadb_ready = False
        self._icon_entries = self._load_icon_entries()

    def _load_icon_entries(self):
        icons_path = Path(__file__).resolve().parents[1] / "assets" / "icons.json"
        with open(icons_path, "r") as f:
            icons = json.load(f)
        return [
            icon
            for icon in icons["icons"]
            if icon["name"].split("-")[-1] == "bold"
        ]

    def _initialize_icons_collection(self):
        if self._chromadb_ready:
            return
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        except ImportError:
            return

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
            documents = []
            ids = []

            for each in self._icon_entries:
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
        self._chromadb_ready = self.collection is not None

    def _fallback_search(self, query: str, k: int):
        query_tokens = [token for token in query.lower().split() if token]
        scored = []
        for entry in self._icon_entries:
            corpus = f"{entry['name']} {' '.join(entry.get('tags', []))}".lower()
            score = sum(1 for token in query_tokens if token in corpus)
            if score > 0:
                scored.append((score, entry["name"]))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [f"/static/icons/bold/{name}.svg" for _, name in scored[:k]]

    async def search_icons(self, query: str, k: int = 1):
        if not self._chromadb_ready:
            self._initialize_icons_collection()

        if self.collection is None:
            return self._fallback_search(query, k)

        result = await asyncio.to_thread(
            self.collection.query,
            query_texts=[query],
            n_results=k,
        )
        return [f"/static/icons/bold/{each}.svg" for each in result["ids"][0]]


ICON_FINDER_SERVICE = IconFinderService()
