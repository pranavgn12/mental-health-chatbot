from pathlib import Path
import uuid

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

import markdown
from bs4 import BeautifulSoup

##################################################
# Configuration
##################################################

KNOWLEDGE_DIR = Path("knowledge")

DB_DIR = "vector_db"

COLLECTION_NAME = "knowledge"

CHUNK_SIZE = 3500

OVERLAP = 500

##################################################
# Markdown -> Plain Text
##################################################


def markdown_to_text(md):

    html = markdown.markdown(md)

    soup = BeautifulSoup(html, "html.parser")

    return soup.get_text("\n")


##################################################
# Chunking
##################################################


def chunk_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunks.append(

            text[start:end]

        )

        start += CHUNK_SIZE - OVERLAP

    return chunks


##################################################
# Embedding Function
##################################################

embedding_function = SentenceTransformerEmbeddingFunction(

    model_name="BAAI/bge-small-en-v1.5"

)

##################################################
# Chroma Client
##################################################

client = chromadb.PersistentClient(

    path=DB_DIR

)

##################################################
# Rebuild Collection
##################################################

try:

    client.delete_collection(

        COLLECTION_NAME

    )

    print("Old collection deleted.")

except Exception:

    print("No previous collection found.")

collection = client.create_collection(

    name=COLLECTION_NAME,

    embedding_function=embedding_function

)

##################################################
# Build Knowledge Database
##################################################

total_chunks = 0

for file in KNOWLEDGE_DIR.glob("*.md"):

    print(f"Processing {file.name}")

    text = markdown_to_text(

        file.read_text(

            encoding="utf-8"

        )

    )

    chunks = chunk_text(

        text

    )

    for i, chunk in enumerate(chunks):

        collection.add(

            ids=[str(uuid.uuid4())],

            documents=[chunk],

            metadatas=[

                {

                    "topic": file.stem,

                    "source": file.name,

                    "chunk": i

                }

            ]

        )

        total_chunks += 1

##################################################
# Finished
##################################################

print()

print("========================================")
print("Knowledge database built successfully.")
print(f"Documents : {len(list(KNOWLEDGE_DIR.glob('*.md')))}")
print(f"Chunks    : {total_chunks}")
print("========================================")