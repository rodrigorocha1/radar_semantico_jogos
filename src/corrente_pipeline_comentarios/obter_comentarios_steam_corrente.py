from itertools import chain
from typing import List, Tuple

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.steam.iapi_steam import IAPISteam


class ObterComentariosSteamCorrente(Corrente):
    def __init__(self, api_steam: IAPISteam, lista_jogos: List[Tuple[int, str]]):
        self.__api_steam = api_steam
        self.__lista_jogos = lista_jogos
        super().__init__()

    def executar_processo(self, contexto: Contexto) -> bool:
        gerador_reviews = chain.from_iterable(
            map(
                lambda review: {**review, 'codigo_steam': codigo_jogo[0], 'nome_jogo': codigo_jogo[1]},
                self.__api_steam.obter_reviews_steam(
                    codigo_jogo_steam=codigo_jogo[0],
                    intervalo_dias=365
                )
            )
            for codigo_jogo in self.__lista_jogos
        )
        contexto.gerador_reviews_steam = gerador_reviews
        return True
