

from langdetect import detect

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


print(dataframe_comentarios.head())
