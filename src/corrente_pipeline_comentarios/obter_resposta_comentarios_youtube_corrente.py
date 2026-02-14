from itertools import chain

from src.contexto.contexto import Contexto
from src.corrente_pipeline_comentarios.corrente import Corrente
from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class ObterRespostaComentariosYoutubeCorrente(Corrente):
    def __init__(self, api_youtube: IApiYoutube):
        self.__api_youtube = api_youtube
        super().__init__()

    def executar_processo(self, contexto: Contexto) -> bool:
        gerador_reviews = chain.from_iterable(
            map(
                lambda review: {**review, 'nome_jogo': id_comentario[0],
                                'id_video': id_comentario[2], },
                self.__api_youtube.obter_resposta_comentarios(
                    id_comentario=id_comentario[3],

                )
            )
            for id_comentario in contexto.lista_id_comentarios
        )
        contexto.gerador_resposta_comentarios = gerador_reviews



        return True
