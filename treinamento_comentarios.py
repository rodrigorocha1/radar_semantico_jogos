import ast
import math

import mlflow
import numpy as np
import pandas as pd
from langdetect import detect
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# from src.modelo.modelo_som import SOM
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb

a = 1


def is_english(text):

    try:
        if len(text) < 20:
            return False
        return detect(text) == 'en'
    except:
        return False


obddb = OperacoesBancoDuckDb()

# Tratamento Comentário

caminho_consulta = f's3://extracao/comentarios/prata/comentarios_limpos_2026_02_23_20_32_35.csv'


dataframe_comentarios = obddb.consultar_dados('1=1', caminho_consulta)

dataframe_comentarios['is_english'] = dataframe_comentarios['texto_comentario'].apply(
    is_english)

dataframe_comentarios = dataframe_comentarios.drop(
    dataframe_comentarios[dataframe_comentarios['is_english']].index)


print(dataframe_comentarios['is_english'].value_counts())

dataframe_comentarios = dataframe_comentarios.head(300)
# print(dataframe_comentarios.shape)


total_neuronios = 5 * math.sqrt(dataframe_comentarios.shape[0])
print(int(total_neuronios))  # 20 * 20
tamnho_batch = 64
dataframe_comentarios["embedings"] = dataframe_comentarios["embedings"].apply(
    ast.literal_eval
)


batches = dataframe_comentarios.shape[0] // tamnho_batch

print(batches)

epocas = 4000 // 102


print(dataframe_comentarios.columns)

scaler = StandardScaler()

embeddings_nomr = scaler.fit_transform(
    dataframe_comentarios['embedings'].tolist())

# dataset = (
#     tf.data.Dataset
#     .from_tensor_slices(embeddings_nomr.astype(np.float32))
#     .shuffle(buffer_size=len(embeddings_nomr))
#     .batch(64)
# )
# som = SOM(
#     linhas=20,
#     colunas=20,
#     dimensao=300,
#     taxa_aprendizado=0.5,
#     metrica="cosseno"
# )
# som.treinar(
#     dataset=dataset,
#     epocas=40,
#     calcular_erro=True,
#     dados_completos=embeddings_nomr
# )

# rotulos = som.rotular_por_centroide(
#     textos=dataframe_comentarios['texto_comentario'].tolist(),
#     embeddings=embeddings_nomr,
#     min_docs=3  # neurônios com menos de 3 comentários são ignorados
# )
# print(rotulos)


# indices, coordenadas = som.mapear(embeddings_nomr)
# print("Índices dos neurônios para cada comentário:", indices)
# print("Coordenadas dos neurônios para cada comentário:", coordenadas)
# # Exibir neurônios rotulados
# # for neuronio, label in rotulos.items():
# #     print(f"Neurônio {neuronio}: {label}")

# for neuronio, label in rotulos.items():
#     i, j = som.localizacoes[neuronio].numpy().astype(int)
#     print(f"({i}, {j}): ['{label}']")


# u_matrix = som.calcular_u_matrix()
# grid_labels = np.full((som.linhas, som.colunas), "", dtype=object)

# for neuronio, label in rotulos.items():
#     i, j = som.localizacoes[neuronio].numpy().astype(int)
#     grid_labels[i, j] = label


# som.plotar_decay(num_epocas=15, num_batches=sum(1 for _ in dataset))
# amostra_embeddings = embeddings_nomr[:5]

# som.registrar_modelo_mlflow(
#     nome_modelo="som_portugues",
#     exemplo_entrada=amostra_embeddings
# )

# indices_np = indices.numpy()
# coordenadas_np = coordenadas.numpy()

# fontes = dataframe_comentarios["site"].values

# som.plotar_heatmap_plataformas(
#     indices_np,
#     fontes
# )

# som.plotar_entropia_plataformas(
#     indices_np,
#     fontes
# )
# mlflow.end_run()
