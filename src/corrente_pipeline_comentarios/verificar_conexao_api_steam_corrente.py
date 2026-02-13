from typing import Optional

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.steam.iapi_steam import IAPISteam


class VerificarConexaoApiSteamCorrente(Corrente):

    def __init__(self, steam_api: IAPISteam):
        super().__init__()
        self.__steam_api = steam_api

    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        return self.__steam_api.checar_conexao()
