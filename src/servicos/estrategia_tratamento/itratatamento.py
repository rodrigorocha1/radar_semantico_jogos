from typing import Protocol, TypeVar

E = TypeVar("E", contravariant=True)
T = TypeVar("T", covariant=True)


class ITratamento(Protocol[T]):
    @property
    def comentario(self) -> str:
        ...

    @comentario.setter
    def comentario(self, novo_comentario: str) -> None:
        ...

    def executar_tratamento(self) -> T:
        ...