from abc import ABC, abstractmethod
from typing import Dict, Generator


class IApiYoutube(ABC):

    @abstractmethod
    def obter_comentarios_youtube(self, id_video: str) -> Generator[Dict, None, None]:
        """
        Método para obter comentários de um vídeo do youtube
        :param id_video: id do vídeo
        :type id_video: str
        :return: Gerador dos comentários
        :rtype: Generator[Dict, None, None]
        """
        pass

    @abstractmethod
    def obter_resposta_comentarios(self, id_comentario: str) -> Generator[Dict, None, None]:
        """
        Método para obter a resposta do comentários
        :param id_comentario: id do comentários
        :type id_comentario: str
        :return: Gerador da resposta do comentários
        :rtype: str
        """