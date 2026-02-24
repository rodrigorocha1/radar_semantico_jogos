import matplotlib.pyplot as plt
import numpy as np
import spacy
import tensorflow as tf

from src.modelo.modelo_som import SOM

# Carrega modelo médio português
nlp = spacy.load("pt_core_news_lg")

comentarios = [
    "Esse vídeo é incrível, adorei a edição!",
    "Não gostei, o áudio está ruim",
    "Muito educativo, aprendi bastante",
    "Haha, muito engraçado",
    "Perdi meu tempo assistindo, sem conteúdo útil",
    "Excelente explicação, obrigado!",
    "O canal está melhorando cada vez mais",
    "Não gostei do estilo do vídeo",
    "Amei os gráficos e animações",
    "Conteúdo muito pobre, não recomendo",
] * 15  # 100 comentários

# Gerar embeddings para cada comentário (vetor médio do documento)
embeddings = np.array(
    [nlp(text).vector for text in comentarios], dtype=np.float32)
print("Shape embeddings:", embeddings.shape)


batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices(
    embeddings).shuffle(len(embeddings)).batch(batch_size)


som = SOM(
    linhas=10,
    colunas=10,
    dimensao=embeddings.shape[1],
    taxa_aprendizado=0.5,
    metrica="euclidiana",
    log_dir="logs/som_portugues"
)

som.treinar(
    dataset=dataset,
    epocas=15,
    calcular_erro=True,
    dados_completos=embeddings
)


indices_bmu, coordenadas = som.mapear(embeddings)

rotulos = som.rotular_por_centroide(
    textos=comentarios,
    embeddings=embeddings,
    min_docs=3  # neurônios com menos de 3 comentários são ignorados
)
print(rotulos)
print(som.erro_quantizacao(embeddings))

# Exibir neurônios rotulados
for neuronio, label in rotulos.items():
    print(f"Neurônio {neuronio}: {label}")


u_matrix = som.calcular_u_matrix()
grid_labels = np.full((som.linhas, som.colunas), "", dtype=object)

for neuronio, label in rotulos.items():
    i, j = som.localizacoes[neuronio].numpy().astype(int)
    grid_labels[i, j] = label

plt.figure(figsize=(8, 8))
plt.imshow(u_matrix, cmap="viridis")
for i in range(som.linhas):
    for j in range(som.colunas):
        if grid_labels[i, j] != "":
            plt.text(j, i, grid_labels[i, j], ha='center',
                     va='center', color='white', fontsize=7)
plt.title("U-Matrix com rótulos de comentários em português")
plt.colorbar()
plt.savefig('u_matrix_portugues.png')