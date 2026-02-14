from datetime import datetime
from typing import Optional

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.configuracao_log import logger
from src.servicos.servico_s3.iservicos3 import Iservicos3


class GuardarDadosYoutubeS3Corrente(Corrente):

    def __init__(self, servico_s3: Iservicos3, servico_banco: IoperacoesBanco):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__caminho_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__caminho_arquivo = f'youtube/bronze/resposta_comentarios_youtube/'
        self.__servico_banco = servico_banco

    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        for dados in contexto.gerador_comentarios_youtube:
            id_comentario = dados['id']
            data_atualizacao_api = dados['snippet']['updatedAt']

            condicao = f"id = {id_comentario} AND snippet.topLevelComment.snippet.updatedAt = {data_atualizacao_api}"
            caminho_consulta = f"s3://extracao/youtube/bronze/resposta_comentarios_youtube{dados['nome_jogo']}/{dados['snippet']['channelId']}/{dados['videoId']}/*.json"

            dataframe = self.__servico_banco.consultar_dados(caminho_consulta=caminho_consulta, id_consulta=condicao)
            if dataframe.empty:
                caminho_completo = self.__caminho_arquivo + f"{dados['nome_jogo']}/{dados['snippet']['channelId']}/{dados['videoId']}" + f'data_{self.__caminho_data}' + '_reviews.json'
                self.__servico_s3.guardar_dados(dados, caminho_completo)
            else:
                logger.info(f'{id_comentario} não teve atualizacao')
        return True
