# RagIA

Construção de um RAG usando documentos do projeto Téo Me Why

## Índice
- [Etapas](#etapas)
  - [Dados](#dados)
    - [Vídeos](#vídeos)
    - [Artigos](#artigos)
  - [Chunking](#chunking)
  - [Vector Database](#vector-database)
  - [LLM](#llm)
- [Referência](#referencia)

## Etapas

### Dados

As fontes de dados utilizadas serão:

#### Vídeos

Alguns vídeos com o contexto necessário foram baixados e transcritos para arquivos Markdown (.md) utilizando o modelo [whisper](https://github.com/openai/whisper).

#### Artigos

Coletânea de artigos do nosso blog em formato markdown (.md).

Todos os dados acima passarão pelas próximas etapas.

### Chunking

Trataremos cada documento (vídeo, artigo, postagem) será "quebrado" em vários chunks. Para ter uma melhor separação desses textos evitando perda de contexto e semântica, utilizaremos o [HybridChunker](https://docling-project.github.io/docling/examples/hybrid_chunking/#hybrid-chunking) da biblioteca [docling](https://www.docling.ai/).

### Vector Database

Cada chunk de cada documento é inserido no banco de dados vetorial Qdrant. Por amor a simplicidade vamos utilizar o serviço cloud do Qdrand em sua versão gratuita.

Métodos de busca que utilizaremos:
- Densa
- Esparsa
- Colbert

Metadados do chunk:
- id
- nome do documento original (da onde o chunk pertence);

Ou seja, para o mesmo chunk do documento, criaremos 3 vetores (embeding) distintos, possibilitando assim todas as buscas necessárias.

### LLM

Utilizaremos a API do [Groq](https://groq.com/) para inferência das LLMs.

Utilizaremos os seguintes argumentos no prompt:
- consulta: pergunta do usuário
- contexto: lista de k-top documentos encontrados na busca vetorial
- instrução: detalhes de como o modelo de genAI deve responder

### Integração com Chat

A interface primária de utilização desse sistema será pelo chat da Twitch. Para isso devemos implementar guardrails para que apenas respostas dentro de nosso contexto sejam respondidas.

A priori a resposta só será gerada para Subs do canal.

## Referência

Nossa inspiração de projeto se deu pelos estudos no curso [deveficiente.com](https://deveficiente.com/oferta-20-especializacao-engenharia-ia?ref=E103343389X&gad_source=1&gad_campaignid=23529177118&gbraid=0AAAABCphGGuicDSEPCRE6hUy782iwutXB&gclid=Cj0KCQjw4PPNBhD8ARIsAMo-icytIETcbuMhL6cViG7JgY6MhpGOsRi_KrVxqJVDEHwyfgVa6kDdqJMaAgUiEALw_wcB)
