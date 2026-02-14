from itertools import chain
from typing import Optional, List, Tuple

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.steam.iapi_steam import IAPISteam


class ObterComentariosYoutubeCorrente(Corrente):
    def __init__(self, api_youtube: IApiYoutube, lista_jogos:List[Tuple[str, ...]]):
        self.__api_youtube = api_youtube
        self.__lista_jogos = lista_jogos
        super().__init__()

    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        if contexto is None:
            return False
        if contexto.gerador_comentarios_youtube is None:
            gerador_reviews = chain.from_iterable(
                map(
                    lambda review: {**review, 'nome_jogo': nome_jogo[1]},
                    self.__api_youtube.obter_comentarios_youtube(
                        id_video=nome_jogo[0]
                    )
                )
                for nome_jogo in self.__lista_jogos
            )
            contexto.gerador_comentarios_youtube = gerador_reviews
            return True
        return False
