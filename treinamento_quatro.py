import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.modelo.modelo_som import SOM

# Reprodutibilidade
np.random.seed(42)
tf.random.set_seed(42)

# --- 3 clusters gaussianos ---
n_por_cluster = 300

cluster_1 = np.random.normal(loc=[0, 0], scale=0.5, size=(n_por_cluster, 2))
cluster_2 = np.random.normal(loc=[4, 4], scale=0.5, size=(n_por_cluster, 2))
cluster_3 = np.random.normal(loc=[0, 5], scale=0.5, size=(n_por_cluster, 2))

X = np.vstack([cluster_1, cluster_2, cluster_3]).astype(np.float32)

# Normalização (boa prática para SOM)
X = (X - X.mean(axis=0)) / X.std(axis=0)

print("Shape dados:", X.shape)


batch_size = 64

dataset = (
    tf.data.Dataset
    .from_tensor_slices(X)
    .shuffle(buffer_size=len(X))
    .batch(batch_size)
)


som = SOM(
    linhas=10,
    colunas=10,
    dimensao=2,
    taxa_aprendizado=0.5,
    metrica="euclidiana",
    log_dir="logs/som_exemplo"
)

som.treinar(
    dataset=dataset,
    epocas=20,
    calcular_erro=True,
    dados_completos=X
)


u_matrix = som.calcular_u_matrix()

plt.figure(figsize=(6, 6))
plt.imshow(u_matrix, cmap="viridis")
plt.colorbar()
plt.title("U-Matrix")
plt.savefig('u_matrix.png')


# Para efeito de demonstração, vamos criar "textos" artificiais
# Cada cluster terá um texto representativo

textos = (
    ["Cluster A"] * n_por_cluster +
    ["Cluster B"] * n_por_cluster +
    ["Cluster C"] * n_por_cluster
)

# embeddings são nossos dados X normalizados
embeddings = X


indices_bmu, coordenadas = som.mapear(embeddings)

print("Exemplo índices BMU:", indices_bmu.numpy()[:10])
print("Exemplo coordenadas BMU:", coordenadas.numpy()[:10])


rotulos = som.rotular_por_centroide(
    textos=textos,
    embeddings=embeddings,
    min_docs=50  # ignora neurônios com poucos dados
)

print("Neurônios rotulados:")
for neuronio, label in rotulos.items():
    print(f"Neurônio {neuronio}: {label}")


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
                     va='center', color='white', fontsize=8)
plt.title("U-Matrix com rótulos por centroide")
plt.colorbar()
plt.savefig('u_matrix_rotulos.png')


