from datetime import datetime
from typing import Optional

import duckdb
import pandas as pd

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.banco.ioperacoes_banco import IoperacoesBanco
from src.servicos.config.configuracao_log import logger
from src.servicos.servico_s3.iservicos3 import Iservicos3


class GuardarDadosSteam3Corrente(Corrente):

    def __init__(self, servico_s3: Iservicos3, servico_banco: IoperacoesBanco):
        super().__init__()
        self.__servico_s3 = servico_s3
        self.__caminho_data = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.__caminho_arquivo = f'steam/bronze/reviews_steam/'
        self.__servico_banco = servico_banco


    def executar_processo(self, contexto: Contexto) -> bool:

        for dados in contexto.gerador_reviews_steam:

            logger.info(f'Guardando json do jogo {dados["codigo_steam"]}')
            steamid_api = dados['author']['steamid']
            timestamp_updated_api = dados['timestamp_updated']

            condicao = f"author.steamid = {steamid_api} AND timestamp_updated = {timestamp_updated_api}"
            caminho_consulta = "s3://extracao/steam/bronze/reviews_steam/" + f'jogo_{dados["codigo_steam"]}/' + '*.json'
            try:
                dataframe = self.__servico_banco.consultar_dados(caminho_consulta=caminho_consulta,id_consulta=condicao)
            except duckdb.IOException:
                dataframe = pd.DataFrame()
            if  dataframe.empty:
                caminho_completo = self.__caminho_arquivo + f'jogo_{dados["codigo_steam"]}/' + f'data_{self.__caminho_data}' + '_reviews.json'
                self.__servico_s3.guardar_dados(dados, caminho_completo)
            else:
                logger.info(f'{steamid_api} não teve atualizacao')
        return True
