import ast
import io
import json
import os

import mlflow
import numpy as np
import tensorflow as tf
from langdetect import detect
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler

from src.modelo.som_v2 import SOMV2
from src.servicos.banco.operacoes_banco import OperacoesBancoDuckDb


def configurar_mlflow(tracking_uri: str, experiment_name: str):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri


configurar_mlflow("http://localhost:5000", experiment_name="RBM_Youtube_Cluster")


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
# Para teste, pegar apenas os primeiros 1000 comentários
dataframe_comentarios = dataframe_comentarios.head(100)

dataframe_comentarios['is_english'] = dataframe_comentarios['texto_comentario'].apply(
    is_english)

dataframe_comentarios = dataframe_comentarios.drop(
    dataframe_comentarios[dataframe_comentarios['is_english']].index)

dataframe_comentarios = dataframe_comentarios.dropna()

dataframe_comentarios['embedings'] = dataframe_comentarios['embedings'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

# Agora sim converte para numpy
embeddings_array = np.array(
    dataframe_comentarios['embedings'].tolist(),
    dtype=np.float32
)



scaler = StandardScaler()
embeddings_nomr = scaler.fit_transform(embeddings_array)
dataset = (
    tf.data.Dataset
    .from_tensor_slices(embeddings_nomr.astype(np.float32))
    .shuffle(buffer_size=len(embeddings_nomr))
    .batch(64)
)

som_v2 = SOMV2(
    linhas=20,
    colunas=20,
    dimensao=300,
    taxa_aprendizado=0.5,
    metrica="cosseno",
    sigma=1
)

som_v2.treinar(dataset, epocas=5)

mapa = som_v2.obter_mapa_ativacao(embeddings_nomr)

resposta = som_v2.obter_resposta_ativacao(embeddings_nomr)

with mlflow.start_run(run_name='SOM Comentários') as run:
    mlflow.log_params(
        {
            'pesos': som_v2.pesos.numpy(),
            'total_neuronios': som_v2.total_neuronios,
            'shape_mapa_ativacao': mapa.shape,
            'resposta': resposta,
            'media_das_distancias': som_v2.distance_map()

        }
    )




    neuronio_vencedor = som_v2.obter_neuronio_vencedor(embeddings_nomr[1])
    lista_neuronios_vencedor = [
        {
            f'comentário_{chave}': f'neurônio_vencedor_{som_v2.obter_neuronio_vencedor(embeddings_nomr[chave])}'
        }
        for chave, valor in enumerate(embeddings_nomr)
    ]

    json_buffer = io.StringIO()
    json.dump(lista_neuronios_vencedor, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/neuronio_vencedor.json')
    json_buffer.close()

    for chave, valor in enumerate(embeddings_nomr):
        neuronio_vencedor = som_v2.obter_neuronio_vencedor(embeddings_nomr[chave])
        print(f"Comentário {chave} - Neurônio Vencedor: {neuronio_vencedor}")
        print()

    rotulos = som_v2.rotular_por_centroide(
        textos=dataframe_comentarios['texto_comentario'].tolist(),
        embeddings=embeddings_nomr,
        min_docs=3  # neurônios com menos de 3 comentários são ignorados
    )

    indices_bmu, coords = som_v2.mapear(embeddings_nomr)

    coords = coords.numpy()

    coords_list = [(int(i), int(j)) for (i, j) in coords]

    print(coords_list[:10])

    rotulos_formatados = {
        f"({som_v2.localizacoes[neuronio].numpy()[0].astype(int)}, "
        f"{som_v2.localizacoes[neuronio].numpy()[1].astype(int)}): ['{label}']": label
        for neuronio, label in rotulos.items()
    }

    json_buffer = io.StringIO()
    json.dump(rotulos_formatados, json_buffer, indent=4, ensure_ascii=False)
    json_buffer.seek(0)
    mlflow.log_text(json_buffer.getvalue(), artifact_file='resultados/rotulos_formatados.json')
    json_buffer.close()



    #
    input_example = embeddings_nomr[:5].astype(np.float32)

    output_example = som_v2(
        tf.convert_to_tensor(input_example)
    )

    signature = infer_signature(
        input_example,
        output_example.numpy()
    )

    mlflow.tensorflow.log_model(
        model=som_v2,
        name="som_model",
        signature=signature,
        input_example=input_example
    )