import numpy as np
import pandas as pd
import spacy

# Carregar modelo
nlp = spacy.load("pt_core_news_lg")

# Exemplo de dataframe existente
df = pd.DataFrame({
    "comentario": [
        "Gostei muito do vídeo",
        "Conteúdo fraco e mal explicado",
        "Excelente tutorial, muito didático",
        "Não recomendo, perdi meu tempo"
    ]
})
# Processamento em batch (mais eficiente que for loop simples)
docs = list(nlp.pipe(df["comentario"], batch_size=32))

embeddings = np.array([doc.vector for doc in docs])
print(embeddings)  # Deve ser (n_comentarios, dimensao_embedding)

df_embeddings = pd.DataFrame(
    embeddings,
    columns=[f"dim_{i}" for i in range(embeddings.shape[1])]
)

print(df_embeddings.shape)
print(df_embeddings.head())
