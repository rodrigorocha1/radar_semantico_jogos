from typing import Generator, Dict

from src.servicos.api_youtube.iapi_youtube import IApiYoutube


class YoutubeAPI(IApiYoutube):
    def obter_comentarios_youtube(self, id_video: str) -> Generator[Dict, None, None]:
        pass

    def obter_resposta_comentarios(self, id_comentario: str) -> Generator[Dict, None, None]:
        pass