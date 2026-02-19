# itratatamento.py

from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class ITratamento(Protocol[T_co]):
    def executar_tratamento(self, comentario: str) -> T_co:
        ...
