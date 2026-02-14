from typing import Generator, Dict

from googleapiclient.discovery import build

from src.servicos.api_youtube.iapi_youtube import IApiYoutube
from src.servicos.config.config import Config


class YoutubeAPI(IApiYoutube):

    def __init__(self):
        self.__youtube = build('youtube', 'v3', developerKey=Config.URL_API_YOUTUBE)

    def obter_comentarios_youtube(self, id_video: str) -> Generator[Dict, None, None]:
        """
            Método para obter comentários de um vídeo do youtube
            :param id_video: id do vídeo
            :type id_video: str
            :return: Gerador dos comentários
            :rtype: Generator[Dict, None, None]
        """
        next_page_token = None
        while True:
            request = self.__youtube.commentThreads().list(
                part="snippet",
                videoId=id_video,
                maxResults=100,
                pageToken=next_page_token,
                textFormat="plainText"
            )
            response = request.execute()
            yield from response["items"]
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break

    def obter_resposta_comentarios(self, id_comentario: str) -> Generator[Dict, None, None]:
        """
        Recupera todas as respostas de um comentário no YouTube.

        Args:
            id_comentario (str): ID do comentário principal.

        Yields:
            Dict: Cada resposta do comentário.
        """
        next_page_token = None

        while True:
            request = self.__youtube.comments().list(
                part="snippet",
                parentId=id_comentario,
                maxResults=100,
                textFormat="plainText"  # plainText ou html
            )
            if next_page_token:
                request = request.pageToken(next_page_token)
            response = request.execute()
            yield from response.get("items", [])
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
