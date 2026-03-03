import ast
import io
import json
import math
import os
from typing import Literal

import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
from langdetect import detect
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from src.modelo.som_v2 import SOMV2
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb


# ---------------- MLflow ----------------
def configurar_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri


configurar_mlflow("http://localhost:5000", experiment_name="RBM_Youtube_Cluster")


# ---------------- Funções auxiliares ----------------
def is_english(text):
    try:
        if len(text) < 20:
            return False
        return detect(text) == 'en'
    except:
        return False


def calcular_r_via_pca(base, normalizar=True):
    if normalizar:
        base = StandardScaler().fit_transform(base)
    pca = PCA(n_components=2)
    pca.fit(base)
    lambda1 = pca.explained_variance_[0]
    lambda2 = pca.explained_variance_[1]
    r = lambda1 / lambda2
    print(f"Lambda1: {lambda1:.6f}")
    print(f"Lambda2: {lambda2:.6f}")
    print(f"r (lambda1/lambda2): {r:.4f}")
    return r, lambda1, lambda2


def determinar_dimensao_ideal(base, variancia_alvo=0.95):
    pca = PCA()
    pca.fit(base)
    variancia_acumulada = np.cumsum(pca.explained_variance_ratio_)
    d_ideal = np.argmax(variancia_acumulada >= variancia_alvo) + 1
    print(f"Dimensão original: {base.shape[1]}")
    print(f"Dimensão ideal para {variancia_alvo * 100:.0f}% variância: {d_ideal}")
    return d_ideal


# ---------------- Carregar e preparar dados ----------------
obddb = OperacoesBancoDuckDb()
caminho_consulta = f's3://extracao/comentarios/prata/comentarios_limpos_2026_02_23_20_32_35.csv'
dataframe_comentarios = obddb.consultar_dados('1=1', caminho_consulta)

# Teste com primeiros 100 comentários
dataframe_comentarios = dataframe_comentarios.head(100)

dataframe_comentarios['is_english'] = dataframe_comentarios['texto_comentario'].apply(is_english)
dataframe_comentarios = dataframe_comentarios.drop(dataframe_comentarios[dataframe_comentarios['is_english']].index)
dataframe_comentarios = dataframe_comentarios.dropna()
dataframe_comentarios['embedings'] = dataframe_comentarios['embedings'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

embeddings_array = np.array(dataframe_comentarios['embedings'].tolist(), dtype=np.float32)
scaler = StandardScaler()
embeddings_nomr = scaler.fit_transform(embeddings_array)

total_neuronios = int(5 * math.sqrt(dataframe_comentarios.shape[0]))
raio, _, _ = calcular_r_via_pca(embeddings_nomr, normalizar=False)
linhas = int(math.sqrt(total_neuronios * raio))
colunas = int(math.sqrt(total_neuronios / raio))
dimensao_som = embeddings_nomr.shape[1]
dimensao = determinar_dimensao_ideal(base=embeddings_nomr)
metrica: Literal["cosseno"] = "cosseno"
sigma = 1
batch_size = int(math.sqrt(total_neuronios))
taxa_aprendizado = 0.5 / math.sqrt(batch_size)
epocas = max(50, int((500 * linhas * colunas * batch_size) / dataframe_comentarios.shape[0]))

som_v2 = SOMV2(
    linhas=linhas,
    colunas=colunas,
    dimensao=dimensao_som,
    taxa_aprendizado=taxa_aprendizado,
    metrica=metrica,
    sigma=sigma
)

dataset = (
    tf.data.Dataset
    .from_tensor_slices(embeddings_nomr.astype(np.float32))
    .shuffle(buffer_size=len(embeddings_nomr))
    .batch(batch_size)
)

som_v2.treinar(dataset, epocas=epocas)

mapa = som_v2.obter_mapa_ativacao(embeddings_nomr)
resposta = som_v2.obter_resposta_ativacao(embeddings_nomr)

with mlflow.start_run(run_name='SOM Comentários') as run:
    densidade = som_v2.obter_resposta_ativacao(embeddings_nomr)

    neuronio_vencedor = som_v2.obter_neuronio_vencedor(embeddings_nomr[1])
    indices_bmu, coordenadas_bmu = som_v2.mapear(embeddings_nomr)

    neuronios_vencedores_unicos = sorted(
        np.unique(indices_bmu.numpy()).tolist()
    )
    coordenadas = som_v2(tf.convert_to_tensor(embeddings_nomr, dtype=tf.float32))

    lista_neuronios_vencedores = [
        {
            "indice_embedding": i,
            "linha": int(coord[0].numpy()),
            "coluna": int(coord[1].numpy())
        }
        for i, coord in enumerate(coordenadas)
    ]

    json_buffer = io.StringIO()
    json.dump(lista_neuronios_vencedores, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/neuronio_vencedor.json')
    json_buffer.close()

    rotulos = som_v2.rotular_por_centroide(
        textos=dataframe_comentarios['texto_comentario'].tolist(),
        embeddings=embeddings_nomr,
        min_docs=3
    )

    coords = coordenadas_bmu.numpy()
    coords_list = [(int(i), int(j)) for (i, j) in coords]

    rotulos_formatados = {
        f"({som_v2.localizacoes[neuronio].numpy()[0].astype(int)}, "
        f"{som_v2.localizacoes[neuronio].numpy()[1].astype(int)}): "
        f"['{label}']": label
        for neuronio, label in rotulos.items()
    }
    json_buffer = io.StringIO()
    json.dump(rotulos_formatados, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/rotulos_formatados.json')
    json_buffer.close()



    mlflow.log_params(
        {
            "raio": float(raio),
            "linhas": int(som_v2.linhas),
            "colunas": int(som_v2.colunas),
            "dimensao": int(som_v2.dimensao),
            "taxa_aprendizado": float(som_v2.taxa_aprendizado),
            "sigma_inicial": float(som_v2.sigma),
            "metrica": str(som_v2.metrica),
            "epocas": int(epocas),
            "batch_size": int(batch_size),
            "total_amostras": int(embeddings_nomr.shape[0]),
            "total_neuronios": int(som_v2.total_neuronios),
            "shape_mapa_ativacao": str(mapa.shape),
            "sigma_final_configurado": float(sigma),
        }
    )

    pesos_np = som_v2.pesos.numpy()
    np.save("pesos_som.npy", pesos_np)
    mlflow.log_artifact("pesos_som.npy", artifact_path="matrizes")
    os.remove("pesos_som.npy")

    resposta_np = resposta.numpy()
    np.save("resposta_ativacao.npy", resposta_np)
    mlflow.log_artifact("resposta_ativacao.npy", artifact_path="matrizes")
    os.remove("resposta_ativacao.npy")

    u_matrix = som_v2.distance_map()

    mlflow.log_metrics({
        "u_matrix_mean": float(tf.reduce_mean(u_matrix).numpy()),
        "u_matrix_std": float(tf.math.reduce_std(u_matrix).numpy()),
        "u_matrix_max": float(tf.reduce_max(u_matrix).numpy()),
    })





    # Métricas
    qe = som_v2.calcular_erro_quantizacao(tf.convert_to_tensor(embeddings_nomr))
    te = som_v2.calcular_erro_topografico(tf.convert_to_tensor(embeddings_nomr))
    u_matrix = som_v2.distance_map()
    mlflow.log_metrics(
        {
            "quantization_error": float(qe.numpy()),
            "topographic_error": float(te.numpy()),
            "u_matrix_mean": float(tf.reduce_mean(u_matrix).numpy()),
            "u_matrix_max": float(tf.reduce_max(u_matrix).numpy()),
            "neuronios_ativos": int(tf.math.count_nonzero(resposta).numpy()),
            "densidade_media_por_neuronio": float(tf.reduce_mean(tf.cast(resposta, tf.float32)).numpy())
        }
    )

    # Neurônio mais populoso
    neur_vencedor = int(tf.argmax(tf.reshape(densidade, [-1])).numpy())
    coords_vencedor = tf.cast(som_v2.localizacoes[neur_vencedor], tf.int32)

    comentarios_cluster = []
    for idx, texto in enumerate(dataframe_comentarios['texto_comentario']):
        neuronio_idx = tf.cast(som_v2.obter_neuronio_vencedor(embeddings_nomr[idx]), tf.int32)
        if tf.reduce_all(neuronio_idx == coords_vencedor):
            comentarios_cluster.append(texto)

    # WordCloud
    indices_bmu = indices_bmu.numpy()

    clusters = {}

    for idx, neur in enumerate(indices_bmu):
        clusters.setdefault(int(neur), []).append(
            dataframe_comentarios['texto_comentario'].iloc[idx]
        )

    neuronios_vencedores_unicos = sorted(clusters.keys())

    print(f"Neurônios ativos: {neuronios_vencedores_unicos}")

    # ===============================
    # Plot grid controlado
    # ===============================

    from math import ceil

    num_cols = 4
    num_rows = ceil(len(neuronios_vencedores_unicos) / num_cols)

    plt.figure(figsize=(15, 4 * num_rows))

    for plot_idx, neur in enumerate(neuronios_vencedores_unicos):
        comentarios_cluster = clusters[neur]

        if not comentarios_cluster:
            continue

        texto_cluster = " ".join(comentarios_cluster)

        wordcloud = WordCloud(
            width=400,
            height=300,
            background_color="white"
        ).generate(texto_cluster)

        plt.subplot(num_rows, num_cols, plot_idx + 1)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Neurônio {neur}")

    plt.tight_layout()

    mlflow.log_figure(
        plt.gcf(),
        "fig/wordclouds_neuronios.png"
    )

    plt.close()



    densidade_np = densidade.numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        densidade_np, annot=True, fmt="d", cmap="viridis",
        xticklabels=[f'R{i}' for i in range(densidade_np.shape[1])],
        yticklabels=[f'A{i}' for i in range(densidade_np.shape[0])]
    )
    plt.xlabel('Região SOM')
    plt.ylabel('Amostra')
    plt.title('Densidade de Ativação por Região do SOM')
    mlflow.log_figure(plt.gcf(), 'fig/heatmap_densidade.png')
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(u_matrix.numpy(), cmap='coolwarm')
    plt.title('U-Matrix (Distâncias entre Neurônios)')
    plt.xlabel('Colunas SOM')
    plt.ylabel('Linhas SOM')

    mlflow.log_figure(plt.gcf(), 'fig/u_matrix.png')
    plt.close()

    neuronios_coords = np.array([som_v2.localizacoes[i].numpy() for i in range(som_v2.total_neuronios)])
    densidade_flat = densidade.numpy().flatten()
    plt.figure(figsize=(8, 6))
    plt.scatter(
        neuronios_coords[:, 1], neuronios_coords[:, 0],
        s=densidade_flat * 50 + 10,
        c=densidade_flat,
        cmap='viridis'
    )
    plt.colorbar(label='Número de Comentários')
    plt.gca().invert_yaxis()
    plt.title('Neurônios do SOM por densidade de comentários')
    plt.xlabel('Colunas SOM')
    plt.ylabel('Linhas SOM')

    mlflow.log_figure(plt.gcf(), 'fig/neuronios_densidade.png')
    plt.close()

    if hasattr(som_v2, 'historico_qe') and hasattr(som_v2, 'historico_te'):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(som_v2.historico_qe) + 1), som_v2.historico_qe, label='Quantization Error')
        plt.plot(range(1, len(som_v2.historico_te) + 1), som_v2.historico_te, label='Topographic Error')
        plt.xlabel('Época')
        plt.ylabel('Erro')
        plt.title('Evolução do QE e TE durante o treinamento')
        plt.legend()
        mlflow.log_figure(plt.gcf(), 'fig/qe_te_evolucao.png')
        plt.close()

    input_example = embeddings_nomr[:5].astype(np.float32)
    output_example = som_v2(tf.convert_to_tensor(input_example))
    signature = infer_signature(input_example, output_example.numpy())
    mlflow.tensorflow.log_model(
        model=som_v2,
        name="som_model",
        signature=signature,
        input_example=input_example
    )
