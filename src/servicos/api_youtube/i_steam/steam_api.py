import subprocess
from typing import List, Final

from src.servicos.api_youtube.i_steam.iapi_steam import IAPISteam
from src.servicos.config.config import Config


# adicionar log

class SteamAPI(IAPISteam):
    def __init__(self):
        self.__url_base: Final[str] = Config.STEAM_API_URL

    def checar_conexao(self) -> bool:
        """
            Método para checar a conexão
            :return: True se conexão é um sucesso falso caso contrário
            :rtype: bool
        """

        try:
            subprocess.run(
                ["curl", "-s", "-f", self.__url_base],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


    def obter_comentarios(self, id_jogo: int) -> List:
        """
            Método para obter o comentários de um jogo
            :param id_jogo: id do jogo
            :type id_jogo: int
            :return: A lista de comentários
            :rtype: List
        """
        pass

