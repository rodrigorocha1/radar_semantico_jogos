from abc import ABC, abstractmethod
from typing import Optional
from src.servicos.config.configuracao_log import logger
from src.contexto.contexto import Contexto


class Corrente(ABC):
    def __init__(self) -> None:
        self._proxima_corrente: Optional["Corrente"] = None

    def set_proxima_corrente(self, corrente: "Corrente"):
        self._proxima_corrente = corrente
        return corrente

    def corrente(self, contexto: Optional[Contexto] = None):
        logger.info(f'Executando {self.__class__.__name__}')
        if self.executar_processo(contexto):
            logger.info(f'Sucesso ao executar {self.__class__.__name__}')
        else:
            logger.info(f'Falha ao executar {self.__class__.__name__}')

    @abstractmethod
    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        pass
