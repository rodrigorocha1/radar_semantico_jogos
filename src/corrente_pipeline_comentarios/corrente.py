from abc import ABC, abstractmethod
from typing import Optional


class Corrente(ABC):
    def __init__(self) -> None:
        self._proxima_corrente: Optional["Corrente"] = None

    def set_proxima_corrente(self, corrente: "Corrente"):
        self._proxima_corrente = corrente
        return corrente

    def corrente(self, contexto):
        if self.executar_processo(contexto):
            print('Sucesso')
        else:
            print('Falha')

    @abstractmethod
    def executar_processo(self, contexto) -> bool:
        pass
