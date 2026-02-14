from itertools import chain
from typing import Optional, List, Tuple

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterRespostaComentariosYoutubeCorrente(Corrente):
    def __init__(self, api_youtube: IApiYoutube, lista_jogos: List[Tuple[str, ...]]):
        self.__api_youtube = api_youtube
        self.__lista_jogos = lista_jogos
        super().__init__()

    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        if contexto is None:
            return False
        if contexto.gerador_comentarios_youtube is not None:
            gerador_reviews = chain.from_iterable(
                map(
                    lambda review: {**review, 'nome_jogo': comentario['nome_jogo'],
                                    'id_video': comentario.get('id_video')},
                    self.__api_youtube.obter_resposta_comentarios(
                        id_comentario=comentario['id']
                    )
                )
                for comentario in contexto.gerador_comentarios_youtube
            )
            contexto.gerador_resposta_comentarios_youtube = gerador_reviews
            return True
        return False
