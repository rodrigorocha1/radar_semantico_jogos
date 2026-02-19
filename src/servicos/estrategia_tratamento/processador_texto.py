from typing import Generic, TypeVar, Optional

from src.servicos.estrategia_tratamento.itratatamento import ITratamento

T = TypeVar("T")


class ProcessadorTexto(Generic[T]):

    def __init__(self):
        self._estrategia = None

    @property
    def estrategia(self) -> Optional[ITratamento[T]]:
        return self._estrategia

    @estrategia.setter
    def estrategia(self, nova_estrategia: ITratamento[T]) -> None:
        self._estrategia = nova_estrategia

    def processar(self, comentario: str) -> T:
        return self._estrategia.executar_tratamento(comentario)
