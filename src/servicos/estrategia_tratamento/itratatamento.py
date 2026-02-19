from typing import Protocol, TypeVar

E = TypeVar("E", contravariant=True)
T = TypeVar("T", covariant=True)

class ITratamento(Protocol[E, T]):
    def executar_tratamento(self, comentario: E) -> T:
        ...
