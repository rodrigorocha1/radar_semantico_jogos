import numpy as np
import spacy

from src.modelo.som_keras import SOMKeras

nlp = spacy.load("pt_core_news_lg")

comentarios = [
    "ótimo vídeo!",
    "não gostei desse conteúdo",
    "excelente explicação",
    "ruim, não recomendo"
]

# Criar embeddings
embeddings = np.array(
    [nlp(texto).vector for texto in comentarios], dtype=np.float32)

# Criar SOM
dim = embeddings.shape[1]
som = SOMKeras(linhas=10, colunas=10, dimensao=dim, log_interval=100)

# Treinar SOM com log informativo
som.fit_embeddings(embeddings, passos=500, batch_size=2)

# Gerar mapa semântico
mapa = som.get_grid_map(comentarios)
for k, v in mapa.items():
    print(f"{k}: {v}")
