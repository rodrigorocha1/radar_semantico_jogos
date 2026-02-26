

import ast
import json
from math import e

import numpy as np
import tensorflow as tf
from langdetect import detect
from sklearn.preprocessing import StandardScaler

from src.modelo.som_v2 import SOMV2
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
# Para teste, pegar apenas os primeiros 1000 comentários
# dataframe_comentarios = dataframe_comentarios.head(100)

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

print("Shape embeddings:", embeddings_array.shape)


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


print(f'{som_v2.pesos.shape}')
som_v2.treinar(dataset, epocas=5)

mapa = som_v2.obter_mapa_ativacao(embeddings_nomr)
print(mapa)
print("Shape mapa ativação:", mapa.shape)

resposta = som_v2.obter_resposta_ativacao(embeddings_nomr)
print(resposta)
print('U-Matrix:')
media_das_distancias = som_v2.distance_map()

print(media_das_distancias)
print('*' * 58)
neuronio_vencedor = som_v2.obter_neuronio_vencedor(embeddings_nomr[1])
print(neuronio_vencedor)

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

print(json.dumps(rotulos_formatados, indent=4, ensure_ascii=False))
