from datetime import datetime

import duckdb
import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.configuracao_log import logger
from src.servicos.servico_s3.iservicos3 import Iservicos3


class GuardarDadosYoutubeRespostaComentariosS3Corrente(Corrente):

    def __init__(self, servico_s3: Iservicos3, servico_banco: IoperacoesBanco):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__caminho_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__caminho_arquivo = f'youtube/bronze/resposta_comentarios_youtube/jogo_'
        self.__servico_banco = servico_banco

    def executar_processo(self, contexto: Contexto) -> bool:

        for dados in contexto.gerador_resposta_comentarios:
            logger.info(f'Guardando json do jogo {dados["nome_jogo"]}')

            id_comentario = dados['id']
            data_atualizacao_api = dados['snippet']['updatedAt']

            caminho_base = f"{self.__caminho_arquivo}{dados['nome_jogo']}/*/*/*.json"
            condicao = f"id = '{id_comentario}' AND snippet.updatedAt = '{data_atualizacao_api}'"
            caminho_consulta = f"s3://extracao/{caminho_base}"

            try:
                dataframe = self.__servico_banco.consultar_dados(caminho_consulta=caminho_consulta,
                                                                 id_consulta=condicao)
            except duckdb.IOException as e:
                dataframe = pd.DataFrame()

            if dataframe.empty:
                logger.info(f'Guardando resposta  comentários: {id_comentario} ')

                caminho_completo = f"{self.__caminho_arquivo}{dados['nome_jogo']}/canal_{dados['snippet']['channelId']}/video_{dados['id_video']}/nome_jogo_{dados['nome_jogo']}.json"
                self.__servico_s3.guardar_dados(dados, caminho_completo)
            else:
                logger.info(f'Resposta comentários: {id_comentario} não teve atualizacao')

        return True
