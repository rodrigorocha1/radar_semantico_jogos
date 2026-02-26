import ast
import json
import math
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from langdetect import detect
from sklearn.preprocessing import StandardScaler

from src.modelo.modelo_som import SOM
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

dataframe_comentarios = dataframe_comentarios.dropna()


# dataframe_comentarios = dataframe_comentarios.head(100)
# print(dataframe_comentarios.shape)


total_neuronios = 5 * math.sqrt(dataframe_comentarios.shape[0])

tamnho_batch = 64
dataframe_comentarios["embedings"] = dataframe_comentarios["embedings"].apply(
    ast.literal_eval
)


batches = dataframe_comentarios.shape[0] // tamnho_batch


epocas = 4000 // 102


scaler = StandardScaler()

embeddings_nomr = scaler.fit_transform(
    dataframe_comentarios['embedings'].tolist())

dataset = (
    tf.data.Dataset
    .from_tensor_slices(embeddings_nomr.astype(np.float32))
    .shuffle(buffer_size=len(embeddings_nomr))
    .batch(64)
)
som = SOM(
    linhas=20,
    colunas=20,
    dimensao=300,
    taxa_aprendizado=0.5,
    metrica="cosseno"
)
som.treinar(
    dataset=dataset,
    epocas=40,
    calcular_erro=True,
    dados_completos=embeddings_nomr
)

rotulos = som.rotular_por_centroide(
    textos=dataframe_comentarios['texto_comentario'].tolist(),
    embeddings=embeddings_nomr,
    min_docs=3  # neurônios com menos de 3 comentários são ignorados
)


rotulos_formatados = {
    f"({som.localizacoes[neuronio].numpy()[0].astype(int)}, "
    f"{som.localizacoes[neuronio].numpy()[1].astype(int)}): ['{label}']": label
    for neuronio, label in rotulos.items()
}


mlflow.log_dict(rotulos_formatados, "rotulos/rotulos_rotulos_formatados.json")
mlflow.log_metric("qtd_neuronios_rotulados", len(rotulos_formatados))


rotulos = {str(k): v for k, v in rotulos.items()}
mlflow.log_dict(rotulos, "rotulos/rotulos_neuronios.json")

indices, coordenadas = som.mapear(embeddings_nomr)


u_matrix = som.calcular_u_matrix()


grid_labels = np.full((som.linhas, som.colunas), "", dtype=object)


som.plotar_decay(num_epocas=15, num_batches=sum(1 for _ in dataset))
amostra_embeddings = embeddings_nomr[:5]


som.gerar_summary_mlflow()
indices_np = indices.numpy()
coordenadas_np = coordenadas.numpy()

fontes = cast(np.ndarray, dataframe_comentarios["site"].to_numpy(dtype=str))

som.plotar_heatmap_plataformas(
    indices_np,
    fontes=fontes
)


resultado_bmu = som.localizar_neuronios_vencedores(
    embeddings_nomr,
    top_k=1
)

indices_bmu = resultado_bmu["indices"]
coordenadas_bmu = resultado_bmu["coordenadas"]
distancias_bmu = resultado_bmu["distancias"]

# Log coordenadas BMU
for amostra_idx, coords in enumerate(coordenadas_bmu):
    # Transformando as coordenadas em lista de tuplas [(i,j), (i,j), ...]
    coords_list = [(int(i), int(j)) for (i, j) in coords]

    # Salvando como string JSON no MLflow
    mlflow.log_param(
        f"coordenadas_bmu_amostra{amostra_idx}", json.dumps(coords_list))

# Log distâncias BMU
step_counter = 0
for amostra_idx, neuronios in enumerate(indices_bmu):
    for k_idx, val in enumerate(neuronios):
        mlflow.log_metric(
            f"indices_bmu_amostra{amostra_idx}_top{k_idx}",
            float(val),
            step=step_counter
        )
        step_counter += 1


df_importancia = som.calcular_importancia_neuronios(
    indices_bmu=indices_bmu.flatten(),
    embeddings=embeddings_nomr,
    fontes=fontes
)
mlflow.log_table(
    data=df_importancia,
    artifact_file="dataframes/importancia_neuronios.parquet"  # inclua a "pasta" no nome
)


som.registrar_ativacoes_bmu_mlflow(embeddings_nomr)
som.registrar_modelo_mlflow(
    nome_modelo="som_portugues",
    exemplo_entrada=amostra_embeddings
)

mlflow.end_run()
