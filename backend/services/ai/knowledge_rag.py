"""Backward-compatible facade for the multi-source knowledge RAG pipeline."""

from services.ai.rag import KnowledgeRAGPipeline


class KnowledgeRAG:
    """Legacy interface; retrieve() continues returning context text or None."""

    def __init__(self):
        self.pipeline = KnowledgeRAGPipeline()
        self.last_result = None

    def retrieve(self, query):
        self.last_result = self.pipeline.retrieve(query)
        return self.last_result.get("context")
