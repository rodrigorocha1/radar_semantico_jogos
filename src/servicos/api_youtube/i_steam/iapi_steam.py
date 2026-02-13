from abc import ABC, abstractmethod
from typing import List


class IAPISteam(ABC):
    @abstractmethod
    def checar_conexao(self) -> bool:
        """
        Método para checar a conexão
        :return: True se conexão é um sucesso falso caso contrário
        :rtype: bool
        """
        pass

    @abstractmethod
    def obter_comentarios(self, id_jogo: int) -> List:
        """
        Método para obter o comentários de um jogo
        :param id_jogo: id do jogo
        :type id_jogo: int
        :return: A lista de comentários
        :rtype: List
        """
        pass
