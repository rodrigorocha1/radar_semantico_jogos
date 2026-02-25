import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from minisom import MiniSom
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar SpaCy para tokenização e lematização
nlp = spacy.load("pt_core_news_lg")

# Exemplo de DataFrame
data = {
    "id_texto": [218261480, 218139992, 218042630, 218041328, 218038914],
    "codigo_steam": [105600]*5,
    "nome_jogo": ["terraria"]*5,
    "texto_comentario": [
        "E POGGERS",
        "oi",
        "Better than life",
        "tuffo",
        "E MUITO BOM E ME DA UM POUCI DE MEDO NA CAVERNA"
    ]
}
df = pd.DataFrame(data)

# Lista de jogos (apenas exemplo simplificado)
lista_jogos = [
    (105600, "terraria")
]

# ----------------------------
# 1. Pré-processamento
# ----------------------------


def preprocess_text(texts):
    processed_texts = []
    for doc in nlp.pipe(texts, batch_size=20):
        tokens = [token.lemma_.lower() for token in doc
                  if not token.is_stop and token.is_alpha]
        processed_texts.append(" ".join(tokens))
    return processed_texts


df['texto_limpo'] = preprocess_text(df['texto_comentario'])

# ----------------------------
# 2. Vetorização
# ----------------------------
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(df['texto_limpo']).toarray()
print(X.shape)

# # ----------------------------
# # 3. Treinamento do SOM
# # ----------------------------
# som_size = 5  # tamanho do grid 5x5
# som = MiniSom(som_size, som_size, X.shape[1], sigma=1.0, learning_rate=0.5)
# som.random_weights_init(X)
# som.train_random(X, 100)  # 100 iterações de treino

# # ----------------------------
# # 4. Atribuição de clusters
# # ----------------------------


# def get_cluster_coordinates(vector):
#     return som.winner(vector)


# df['cluster'] = [get_cluster_coordinates(x) for x in X]

# print(df[['texto_comentario', 'texto_limpo', 'cluster']])

# # ----------------------------
# # 5. Visualização do SOM
# # ----------------------------
# # Criar heatmap simples: número de comentários por cluster
# heatmap = np.zeros((som_size, som_size))
# for coord in df['cluster']:
#     heatmap[coord] += 1

# plt.figure(figsize=(6, 6))
# plt.imshow(heatmap, cmap='coolwarm')
# plt.colorbar()
# plt.title("Heatmap SOM - número de comentários por cluster")
# plt.show()
