import subprocess
import urllib.parse
from typing import Final, Generator, Dict

import requests

from src.servicos.config.config import Config
from src.servicos.config.configuracao_log import logger
from src.servicos.steam.iapi_steam import IAPISteam


# adicionar log

class SteamAPI(IAPISteam):
    def __init__(self):
        self.__URL_BASE: Final[str] = Config.STEAM_API_URL

    def checar_conexao(self) -> bool:
        """
            Método para checar a conexão
            :return: True se conexão é um sucesso falso caso contrário
            :rtype: bool
        """

        try:
            subprocess.run(
                ["curl", "-s", "-f", self.__URL_BASE],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info('Sucesso ao conectar na api da steam')
            return True
        except subprocess.CalledProcessError:
            logger.info('Falha ao conectar na api da steam')
            return False

    def obter_reviews_steam(self, codigo_jogo_steam: int, intervalo_dias) -> Generator[Dict, None, None]:
        """
        Método para obter as reviews da steam
        :param codigo_jogo_steam: código do jogo da steam
        :type codigo_jogo_steam: int
        :param intervalo_dias: intervalo de buscas
        :type intervalo_dias: int
        :return: Gerador com as reviews
        :rtype:  Generator[Dict, None, None]
        """
        parametros = {
            'json': 1,
            'filter': 'recent',
            'language': 'portuguese',
            'cursor': '*',
            'review_type': 'all',
            'purchase_type': 'all',
            'num_per_page': '100',
            'day_range': intervalo_dias
        }
        while True:
            if parametros['cursor'] is None:
                break
            url_api = f'{self.__URL_BASE}{codigo_jogo_steam}'
            req = requests.get(url=url_api, params=parametros, timeout=10)
            req = req.json()
            yield from req['reviews']
            cursor = req.get('cursor')
            if not cursor:
                break
            cursor = cursor.strip('"')
            parametros['cursor'] = urllib.parse.quote(cursor)
