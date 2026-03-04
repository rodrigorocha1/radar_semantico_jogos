import ast
import io
import json
import math
import os
import re

import plotly.graph_objects as go
import plotly.io as pio

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
from typing import Literal
import random
import mlflow
import numpy as np
import seaborn as sns
import tensorflow as tf
from langdetect import detect
from matplotlib import pyplot as plt
from mlflow.models import infer_signature
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from wordcloud import WordCloud

from src.modelo.som_v2 import SOMV2
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()


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
# dataframe_comentarios = dataframe_comentarios.head(100)

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
epocas = 5
random_state = 42

som_v2 = SOMV2(
    linhas=linhas,
    colunas=colunas,
    dimensao=dimensao_som,
    taxa_aprendizado=taxa_aprendizado,
    metrica=metrica,
    sigma=sigma,
    random_state=42
)

dataset = (
    tf.data.Dataset
    .from_tensor_slices(embeddings_nomr.astype(np.float32))
    .shuffle(
        buffer_size=len(embeddings_nomr),
        seed=SEED,
        reshuffle_each_iteration=False  # CRÍTICO
    )
    .batch(batch_size, drop_remainder=False)
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
            "random_state": int(random_state),
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
    for key, label in rotulos_formatados.items():
        match = re.search(r'\((\d+),\s*(\d+)\)', key)
        if match:
            linha = int(match.group(1))
            coluna = int(match.group(2))

        else:
            print(f"Não foi possível extrair coordenadas de {key}")

        wc = WordCloud(
            width=400, height=400, background_color="white",
            colormap="plasma", max_font_size=80
        ).generate(texto)

        plt.figure(figsize=(6, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Neurônio ({linha}, {coluna})")

        mlflow.log_figure(plt.gcf(), f'fig/wordcloud_{linha}_{coluna}.png')
        plt.close()

    ######

    densidade_np = densidade.numpy()
    print('-' * 20)
    print(densidade)
    print(densidade_np)
    print('-' * 20)
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

    coords = coordenadas_bmu.numpy()

    associacao = []

    for idx, (i, j) in enumerate(coords):
        associacao.append({
            "comentario": dataframe_comentarios['texto_comentario'].iloc[idx],
            "linha": int(i),
            "coluna": int(j),
            "u_matrix_valor": float(u_matrix[int(i), int(j)])
        })

    threshold = np.mean(u_matrix)

    regioes_fronteira = []
    regioes_cluster = []

    for item in associacao:
        if item["u_matrix_valor"] > threshold:
            regioes_fronteira.append(item)
        else:
            regioes_cluster.append(item)

    plt.figure(figsize=(8, 6))
    sns.heatmap(u_matrix, cmap='coolwarm')

    for idx, (i, j) in enumerate(coords):
        plt.text(
            j + 0.5,
            i + 0.5,
            str(idx),
            ha='center',
            va='center',
            fontsize=6,
            color='black'
        )

    plt.title("U-Matrix com índices dos comentários")
    mlflow.log_figure(plt.gcf(), 'fig/u_matriz_indice_comentarios.png')
    plt.close()

    hit_map = som_v2.obter_resposta_ativacao(embeddings_nomr)

    plt.figure(figsize=(8, 6))
    sns.heatmap(hit_map, cmap='Blues', annot=True, fmt=".0f")
    plt.title("Hit Map (Densidade de Comentários)")
    mlflow.log_figure(plt.gcf(), 'fig/u_matriz_indice_comentarios.png')
    plt.close()

    hit_map = np.zeros_like(u_matrix)

    for (i, j) in coords:
        hit_map[int(i), int(j)] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(u_matrix, cmap='coolwarm', alpha=0.6)

    for i in range(hit_map.shape[0]):
        for j in range(hit_map.shape[1]):
            if hit_map[i, j] > 0:
                plt.text(
                    j + 0.5,
                    i + 0.5,
                    str(int(hit_map[i, j])),
                    ha='center',
                    va='center',
                    fontsize=8,
                    color='black',
                    weight='bold'
                )

    plt.title("U-Matrix com Densidade de Comentários")
    mlflow.log_figure(plt.gcf(), 'fig/u_matriz_hit_count.png')
    plt.close()

    # Pesos da SOM (shape: n_neuronios x n_features)
    range_k = range(2, 15)
    best_k = None
    best_score = -1
    # --------------------------------------------------
    pesos = som_v2.obter_pesos(flatten=True)

    for k in range_k:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = kmeans.fit_predict(pesos)

        # Garante que existem pelo menos 2 clusters reais
        if len(np.unique(labels)) > 1:
            score = silhouette_score(pesos, labels)

            if score > best_score:
                best_score = score
                best_k = k

    # Log MLflow
    mlflow.log_metric("best_k_kmeans", best_k)
    mlflow.log_metric("best_score_kmeans", best_score)

    print(f"Melhor k: {best_k}")
    print(f"Silhouette: {best_score:.4f}")

    # --------------------------------------------------
    # 3️⃣ Treinamento final
    # --------------------------------------------------
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(pesos)

    # --------------------------------------------------
    # 4️⃣ Corrigir reshape usando dimensões reais
    # --------------------------------------------------
    cluster_map = cluster_labels.reshape(
        som_v2.linhas,
        som_v2.colunas
    )

    # --------------------------------------------------
    # 5️⃣ Plot correto
    # --------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cluster_map,
        cmap="tab20",
        annot=True,
        fmt="d",
        cbar=True
    )
    plt.title(f"Clusters na SOM (k={best_k})")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), 'fig/cluster_som.png')
    plt.close()

    ### Calcular centroides por neurônio

    from collections import defaultdict
    import numpy as np

    clusters_indices = defaultdict(list)

    # Mapear índices por neurônio
    for idx, coord in enumerate(coords):
        i, j = int(coord[0]), int(coord[1])
        clusters_indices[(i, j)].append(idx)

    centroides = {}
    for chave, indices in clusters_indices.items():
        if len(indices) > 1:
            centroides[chave] = np.mean(embeddings_nomr[indices], axis=0)
    centroides_json = {str(k): v.tolist() for k, v in centroides.items()}
    json_buffer = io.StringIO()

    json.dump(centroides_json, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/centroides.json')
    json_buffer.close()

    # Clusterizar os centroides (Macrotemas)

    X = np.array(list(centroides.values()))
    ks = range(2, 11)
    inertias = []
    silhouettes = []

    best_score = -1
    best_k = None

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)

        # Silhouette Score correto
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            silhouettes.append(score)

            if score > best_score:
                best_score = score
                best_k = k

    mlflow.log_metric("silhouette_score_centroide", best_score)
    mlflow.log_metric("k_centroide", best_k)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ks, inertias, marker='o')
    plt.title("Método do Cotovelo")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inércia")

    # Plot do Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(ks, silhouettes, marker='o', color='orange')
    plt.title("Silhouette Score")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette Score")

    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), 'fig/cluster_centroide_macrotema.png')
    plt.close()

    X = np.array(list(centroides.values()))
    chaves = list(centroides.keys())

    # número de macrotemas
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    labels_macro = kmeans.fit_predict(X)

    macrotemas = {}

    for chave, label in zip(chaves, labels_macro):
        macrotemas.setdefault(label, []).append(chave)
    print(macrotemas)

    macrotemas_json = {str(k): v for k, v in macrotemas.items()}
    json_buffer = io.StringIO()
    json.dump(macrotemas_json, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/macrotemas.json')
    json_buffer.close()

    # Gráfico macrotemas
    # Dicionário final com macrotemas e coordenadas dos neurônios
    macrotemas_coords = {}

    for macro, neurons in macrotemas.items():
        coords = []
        for neuron in neurons:
            # neuron é a tupla (linha, coluna)
            coords.append({
                "linha": neuron[0],
                "coluna": neuron[1]
            })
        macrotemas_coords[macro] = coords

    # Mostrar resultado
    for k, v in macrotemas_coords.items():
        print(f"Macrotema {k}: {v}")

    # Salvar no MLflow
    macrotemas_coords_json = {str(k): v for k, v in macrotemas_coords.items()}
    json_buffer = io.StringIO()
    json.dump(macrotemas_coords_json, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/macrotemas_coords.json')
    json_buffer.close()


    # Gráfico macrotema


    # Criar matriz do SOM para plotagem (linhas x colunas)
    mapa_macrotema = np.full((som_v2.linhas, som_v2.colunas), -1, dtype=int)

    # Preencher matriz com o índice do macrotema de cada neurônio
    for macro, neurons in macrotemas_coords.items():
        for neuron in neurons:
            linha = neuron["linha"]
            coluna = neuron["coluna"]
            mapa_macrotema[linha, coluna] = macro

    # Plot
    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("tab20", n_colors=len(macrotemas_coords))
    sns.heatmap(
        mapa_macrotema,
        annot=True,
        fmt="d",
        cmap=cmap,
        cbar=True,
        linewidths=0.5,
        linecolor="gray"
    )
    plt.title("Mapa SOM com Macrotemas")
    plt.xlabel("Colunas SOM")
    plt.ylabel("Linhas SOM")
    plt.gca().invert_yaxis()  # inverter eixo y para coincidir com SOM
    plt.tight_layout()

    # Log no MLflow
    mlflow.log_figure(plt.gcf(), 'fig/som_macrotemas.png')
    plt.close()

    mlflow.tensorflow.log_model(
        model=som_v2,
        name="som_model",
        signature=signature,
        input_example=input_example
    )
