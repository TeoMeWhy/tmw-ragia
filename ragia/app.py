import flask

import os
import dotenv
import mlflow

from openai import OpenAI
import qdrant_client
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

dotenv.load_dotenv()

print("Carregando modelo de guardrails...")
mlflow.set_tracking_uri(os.getenv("MLFLOW_URI"))
model_infos = mlflow.search_registered_models(filter_string="name='ragia_guardrails'")[0]
MODEL_GUARDRAILS_CUTOFF = model_infos.tags["cutoff"]
MODEL_GUARDRAILS_NAME = model_infos.name
MODEL_GUARDRAILS_LAST_VERSION = max([ int(v.version) for v in model_infos.latest_versions])
MODEL_GUARDRAILS_URI = f"models:/{MODEL_GUARDRAILS_NAME}/{MODEL_GUARDRAILS_LAST_VERSION}"
MODEL_GUARDRAILS = mlflow.sklearn.load_model(MODEL_GUARDRAILS_URI)
print(f"Modelo de guardrails carregado: {MODEL_GUARDRAILS_URI}")


DENSE_MODEL = os.getenv("DENSE_MODEL")
SPARSE_MODEL = os.getenv("SPARSE_MODEL")
COLBERT_MODEL = os.getenv("COLBERT_MODEL")

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

qudrant_client = qdrant_client.QdrantClient(
    url = QDRANT_CLUSTER_ENDPOINT,
    api_key = QDRANT_API_KEY,
)

openai_client = OpenAI(
    api_key = GROQ_API_KEY,
    base_url = "https://api.groq.com/openai/v1",
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)


app = flask.Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():

    data = flask.request.json
    query = data.get("query", "")
    if query == "":
        return flask.jsonify({"error":"Query is required."}), 400

    dense_query = list(dense_model.passage_embed(query))[0].tolist()

    prob_guardrails = MODEL_GUARDRAILS.predict_proba([dense_query])[0][1]
    if prob_guardrails <= float(MODEL_GUARDRAILS_CUTOFF):
        return flask.jsonify({"error":"Pergunta fora do contexto. Tente reformular."}), 400

    sparse_query = list(sparse_model.passage_embed(query))[0].as_object() 
    colbert_query = list(colbert_model.passage_embed(query))[0].tolist()

    results = qudrant_client.query_points(
        collection_name="ragia",
        prefetch={
            "prefetch": [
                {"query": dense_query, "using":"dense", "limit": 10},
                {"query": sparse_query, "using":"sparse", "limit": 10},
            ],
            "query": models.FusionQuery(fusion=models.Fusion.RRF),
            "limit":20,
        },
        query = colbert_query,
        using="colbert",
        limit=3,
    )

    prompt = f"""
    Responda a seguinte pergunta usando os seguintes parágrafos de contexto:

    Pergunta: {query}

    Contexto:
    {'\n'.join([f'- {r.payload["text"]}\n' for r in results.points])}
    
    ---

    Responda de forma clara e objetivo com no máximo 300 caracteres.

    Caso você não tenha contexto suficiente para responder, retorne uma string vazia.
    """

    response = openai_client.responses.create(
        input=prompt,
        model="openai/gpt-oss-20b",
    )
    
    return flask.jsonify({"response": response.output_text}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)