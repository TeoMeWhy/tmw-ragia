# RagIA

Construção de um RAG usando documentos do projeto Téo Me Why

## Stack

### Chunking

Trataremos cada documento (vídeo, artigo, postagem) será "quebrado" em vários chunks. Para ter uma melhor separação desses textos evitando perda de contexto e semântica, utilizaremos o [HybridChunker](https://docling-project.github.io/docling/examples/hybrid_chunking/#hybrid-chunking) da biblioteca [docling](https://www.docling.ai/).

### Qdrant Cloud

Serviço de bancos vetoriais cloud para armazenamento e busca por documentos

Métodos de busca que utilizaremos:
- Densa
- Esparsa
- Colbert

Ou seja, para o mesmo documento, criaremos 3 vetores (embeding) distintos.

### Groq

Utilizaremos a API do [Groq](https://groq.com/) para inferência das LLMs.




