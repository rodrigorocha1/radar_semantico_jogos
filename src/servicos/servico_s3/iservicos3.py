from abc import ABC
from typing import Dict


class Iservicos3(ABC):

    def guardar_dados(self, dados: Dict):
        """
        Método para Guardar json no s3 minio
        :param dados: json
        :type dados: Dict
        :return: None
        :rtype: None
        """
