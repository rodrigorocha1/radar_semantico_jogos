from typing import List, Final

from src.servicos.api_youtube.i_steam.iapi_steam import IAPISteam
from src.servicos.config.config import Config


class SteamAPI(IAPISteam):
    def __init__(self):
        self.__url_base: Final[str] = Config.STEAM_API_URL

    def checar_conexao(self) -> bool:
        pass

    def obter_comentarios(self, id_jogo: int) -> List:
        pass
