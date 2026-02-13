from typing import List

from src.servicos.api_youtube.i_steam.iapi_steam import IAPISteam


class SteamAPI(IAPISteam):
    def checar_conexao(self) -> bool:
        pass

    def obter_comentarios(self, id_jogo: int) -> List:
        pass