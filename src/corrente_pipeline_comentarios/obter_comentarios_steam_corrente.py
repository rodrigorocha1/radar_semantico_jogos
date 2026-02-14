from itertools import chain
from typing import Optional, List

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.steam.iapi_steam import IAPISteam


class ObterComentariosSteamCorrente(Corrente):
    def __init__(self, api_steam: IAPISteam, lista_jogos: List[int]):
        self.__api_steam = api_steam
        self.__lista_jogos = lista_jogos
        super().__init__()

    def executar_processo(self, contexto: Contexto) -> bool:
        if contexto is None:
            return False
        if contexto.gerador_reviews_steam is None:
            gerador_reviews = chain.from_iterable(
                map(
                    lambda review: {**review, 'codigo_steam': codigo_jogo},
                    self.__api_steam.obter_reviews_steam(
                        codigo_jogo_steam=codigo_jogo,
                        intervalo_dias=365
                    )
                )
                for codigo_jogo in self.__lista_jogos
            )
            contexto.gerador_reviews_steam = gerador_reviews
            return True
        return False
