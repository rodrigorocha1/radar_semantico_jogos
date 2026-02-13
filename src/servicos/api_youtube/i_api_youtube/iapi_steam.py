from abc import ABC, abstractmethod
from typing import List


class IAPISteam(ABC):
    @abstractmethod
    def checar_conexao(self, id_video: str) -> bool:
        pass

    @abstractmethod
    def obter_dados(self, id_jogo: int) -> List:
        pass
