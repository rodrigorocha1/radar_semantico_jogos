from abc import ABC, abstractmethod
from typing import Optional
from src.servicos.config.configuracao_log import logger
from src.contexto.contexto import Contexto


class Corrente(ABC):
    def __init__(self) -> None:
        self._proxima_corrente: Optional["Corrente"] = None

    def set_proxima_corrente(self, corrente: "Corrente") -> "Corrente":
        """
        Define a próxima corrente da pipeline.
        Retorna a corrente passada para permitir encadeamento.
        """
        self._proxima_corrente = corrente
        return corrente

    def corrente(self, contexto: Optional[Contexto] = None):
        """
        Executa o processo desta corrente.
        Se houver sucesso, passa para a próxima corrente (se existir).
        """
        logger.info(f'Executando {self.__class__.__name__}')
        try:
            sucesso = self.executar_processo(contexto)
        except Exception as e:
            logger.error(f'Erro ao executar {self.__class__.__name__}: {e}')
            sucesso = False

        if sucesso:
            logger.info(f'Sucesso ao executar {self.__class__.__name__}')
            # Verificação segura antes de chamar a próxima corrente
            if self._proxima_corrente:
                self._proxima_corrente.corrente(contexto)
        else:
            logger.warning(f'Falha ao executar {self.__class__.__name__}')

    @abstractmethod
    def executar_processo(self, contexto: Optional[Contexto] = None) -> bool:
        """
        Deve ser implementado em cada corrente concreta.
        Retorna True se o processo foi bem-sucedido, False caso contrário.
        """
        pass
