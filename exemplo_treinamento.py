import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from src.modelo.modelo_som import SOM

# -------------------------------------------------
# 1️⃣ Gerar dados simples (3 clusters 2D)
# -------------------------------------------------

np.random.seed(42)

cluster1 = np.random.normal(loc=[2, 2], scale=0.5, size=(150, 2))
cluster2 = np.random.normal(loc=[-2, -2], scale=0.5, size=(150, 2))
cluster3 = np.random.normal(loc=[2, -2], scale=0.5, size=(150, 2))

X = np.vstack([cluster1, cluster2, cluster3]).astype(np.float32)

# Normalização
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# -------------------------------------------------
# 2️⃣ Dataset TensorFlow
# -------------------------------------------------

dataset = (
    tf.data.Dataset
    .from_tensor_slices(X)
    .shuffle(buffer_size=len(X))
    .batch(32)
)

# -------------------------------------------------
# 3️⃣ Criar SOM
# -------------------------------------------------

som = SOM(
    linhas=10,
    colunas=10,
    dimensao=2,
    taxa_aprendizado=0.5,
    metrica="euclidiana"
)


# -------------------------------------------------
# 4️⃣ Erro inicial
# -------------------------------------------------

erro_inicial = som.erro_quantizacao(X)
print(f"Erro inicial: {erro_inicial:.4f}")

# -------------------------------------------------
# 5️⃣ Treinar
# -------------------------------------------------

som.treinar(dataset, epocas=20)

# -------------------------------------------------
# 6️⃣ Erro final
# -------------------------------------------------

erro_final = som.erro_quantizacao(X)
print(f"Erro final: {erro_final:.4f}")

# -------------------------------------------------
# 7️⃣ Plot U-Matrix
# -------------------------------------------------

u_matrix = som.calcular_u_matrix()

plt.figure()
plt.imshow(u_matrix, cmap="coolwarm")
plt.colorbar()
plt.title("U-Matrix")
plt.savefig("som_u_matrix.png")

# -------------------------------------------------
# 8️⃣ Visualizar pesos no espaço 2D
# -------------------------------------------------

pesos = som.pesos.numpy()

plt.figure()
plt.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Dados")
plt.scatter(pesos[:, 0], pesos[:, 1], color="red", label="Neurônios")
plt.title("SOM - Organização dos neurônios")
plt.legend()
plt.savefig("som_organizacao.png")



# som keras
# som = SOM(20, 20, dimensao=4)

# total_steps = tf.constant(1000.0)

# for step, batch in enumerate(dataset):
#     taxa, sigma = som.train_step_som(
#         batch,
#         tf.constant(step, dtype=tf.float32),
#         total_steps
#     )

# tensorboard_callback = tf.keras.callbacks.TensorBoard(
#     log_dir="logs"
# )

# som.compile()  # mesmo sem optimizer

# som.fit(dataset, epochs=10, callbacks=[tensorboard_callback])
