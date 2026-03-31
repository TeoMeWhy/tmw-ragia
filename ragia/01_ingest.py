# %%
import os
import dotenv
import uuid

from tqdm import tqdm

dotenv.load_dotenv()

import qdrant_client
from qdrant_client import models

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

DENSE_MODEL = "intfloat/multilingual-e5-large"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")

client = qdrant_client.QdrantClient(
    url=QDRANT_CLUSTER_ENDPOINT,
    api_key=QDRANT_API_KEY,
)

doc_convert = DocumentConverter()

chunker = HybridChunker(
    tokenizer=AutoTokenizer.from_pretrained(DENSE_MODEL),
    max_tokens=450,
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

# %%
def doc_to_vectordb(path):

    doc = doc_convert.convert(path)
    chunks = list(chunker.chunk(doc.document))

    points = []
    for c in chunks:
        dense_embedding = list(dense_model.passage_embed(c.text))[0].tolist()
        sparse_embeding = list(sparse_model.passage_embed(c.text))[0].as_object() 
        colbert_embeding = list(colbert_model.passage_embed(c.text))[0].tolist()

        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_embedding,
                "colbert": colbert_embeding,
                "sparse": sparse_embeding,
            },
            payload={
                "metadata": {
                    "file_name": c.meta.origin.filename,
                },
                "text": c.text,
            }
        )
        points.append(point)
    
    client.upload_points(
        collection_name="ragia",
        points=points,
    )

# %%

collections =  [i.name for i in client.get_collections().collections]

if "ragia" not in collections:

    client.create_collection(
        collection_name="ragia",
        vectors_config={
            "dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
            "colbert": models.VectorParams(size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        }
    )

files = [i for i in os.listdir("../data") if i.endswith(".md")]

for file in tqdm(files):
    doc_to_vectordb(f"../data/{file}")