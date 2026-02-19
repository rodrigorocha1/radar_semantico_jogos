from typing import Generic, TypeVar, Optional

from src.servicos.estrategia_tratamento.itratatamento import ITratamento

T = TypeVar("T")
U = TypeVar("U")


class ProcessadorTexto(Generic[T, U ]):

    def __init__(self):
        self._estrategia: Optional[ITratamento[T, U]] = None

    @property
    def estrategia(self) -> Optional[ITratamento[T, U]]:
        return self._estrategia

    @estrategia.setter
    def estrategia(self, nova_estrategia: ITratamento[T, U]) -> None:
        self._estrategia = nova_estrategia

    def processar(self, comentario: T) -> U:
        if self._estrategia is None:
            raise ValueError("Estratégia de tratamento não foi definida.")
        return self._estrategia.executar_tratamento(comentario)
