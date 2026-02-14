from abc import ABC, abstractmethod
from typing import Dict


class Iservicos3(ABC):

    @abstractmethod
    def guardar_dados(self, dados: Dict, caminho_arquivo: str):
        """
        Método para guardar dados

        :param dados: requisição da api
        :type dados: Dict
        :param caminho_arquivo: caminho do arquivo
        :type caminho_arquivo:  str
        :return:
        :rtype:
        """
        pass