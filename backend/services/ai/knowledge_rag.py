import chromadb

from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)


class KnowledgeRAG:

    def __init__(self):

        embedding_function = SentenceTransformerEmbeddingFunction(

            model_name="BAAI/bge-small-en-v1.5"

        )

        client = chromadb.PersistentClient(

            path="vector_db"

        )

        self.collection = client.get_collection(

            name="knowledge",

            embedding_function=embedding_function

        )

        ##################################################
        # Similarity Threshold
        ##################################################

        self.threshold = 0.60

    def retrieve(

        self,

        query

    ):

        ##################################################
        # Search
        ##################################################

        results = self.collection.query(

            query_texts=[query],

            n_results=3,

            include=[

                "documents",

                "distances"

            ]

        )

        documents = results["documents"][0]

        distances = results["distances"][0]

        ##################################################
        # Nothing Retrieved
        ##################################################

        if len(documents) == 0:

            return None

        ##################################################
        # Best Match Too Far
        ##################################################

        if distances[0] > self.threshold:

            print("\n===== Knowledge RAG =====")
            print("No relevant knowledge found.")
            print(f"Distance: {distances[0]:.3f}")
            print("=========================\n")

            return None

        ##################################################
        # Build Knowledge
        ##################################################

        knowledge = ""

        for document, distance in zip(

            documents,

            distances

        ):

            if distance <= self.threshold:

                knowledge += document

                knowledge += "\n\n"

        ##################################################
        # Debug
        ##################################################

        print("\n===== Knowledge Retrieved =====")

        print(

            knowledge

        )

        print("Distances:", distances)

        print("===============================\n")

        return knowledge.strip()