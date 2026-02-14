import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.config import Config as c


class CriacaoDataframeCompletoCorrente(Corrente):

    def __init__(self, servico_banco: IoperacoesBanco):
        super().__init__()
        self.__servico_banco = servico_banco
        self.__jogos_dict_invertido = c.CONFIG_JOGOS

    def __criar_dataframe_steam(self) -> pd.DataFrame:
        caminho_base = f's3://{c.MINIO_BUCKET_PLN}/steam/bronze/reviews_steam/**/*.json'
        dataframe = self.__servico_banco.consultar_dados(id_consulta='1=1', caminho_consulta=caminho_base)
        dataframe = dataframe[['recommendationid', 'codigo_steam', 'nome_jogo', 'review']]
        dataframe.rename(columns={'recommendationid': 'id_texto', 'review': 'texto_comentario'}, inplace=True)
        return dataframe

    def __criar_dataframe_youtube_comentarios(self) -> pd.DataFrame:
        caminho_base = f's3://{c.MINIO_BUCKET_PLN}/youtube/bronze/comentarios_youtube/**/*.json'
        dataframe = self.__servico_banco.consultar_dados(caminho_consulta=caminho_base, id_consulta="1=1")
        dataframe["codigo_steam"] = dataframe["nome_jogo"].map(self.__jogos_dict_invertido)
        dataframe["textDisplay"] = dataframe["snippet"].apply(
            lambda x: x["topLevelComment"]["snippet"]["textDisplay"]
        )
        dataframe = dataframe[['id', 'codigo_steam', 'nome_jogo', 'textDisplay']]
        dataframe.rename(columns={'id': 'id_texto', 'textDisplay': 'texto_comentario'}, inplace=True)
        return dataframe

    def __criar_dataframe_youtube_resposta_comentarios(self) -> pd.DataFrame:
        caminho_base = f's3://{c.MINIO_BUCKET_PLN}/youtube/bronze/resposta_comentarios_youtube/**/*.json'
        dataframe = self.__servico_banco.consultar_dados(id_consulta="1=1", caminho_consulta=caminho_base)
        dataframe["codigo_steam"] = dataframe["nome_jogo"].map(self.__jogos_dict_invertido)

        dataframe["textDisplay"] = dataframe["snippet"].apply(
            lambda x: x["textDisplay"]
        )
        dataframe = dataframe[['id', 'codigo_steam', 'nome_jogo', 'textDisplay']]
        dataframe.rename(columns={'id': 'id_texto', 'textDisplay': 'texto_comentario'}, inplace=True)

        return dataframe

    def executar_processo(self, contexto: Contexto) -> bool:
        dataframe_steam = self.__criar_dataframe_steam()
        dataframe_comentarios_youtube = self.__criar_dataframe_youtube_comentarios()
        dataframe_resposta_comentarios = self.__criar_dataframe_youtube_resposta_comentarios()
        dataframe_completo = pd.concat([dataframe_steam, dataframe_resposta_comentarios,dataframe_comentarios_youtube])
        print(dataframe_completo['nome_jogo'].unique())

        return True
