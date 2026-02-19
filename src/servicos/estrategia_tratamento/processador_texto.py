from typing import Generic, TypeVar, Optional, Union, List
from src.servicos.estrategia_tratamento.itratatamento import ITratamento

T = TypeVar("T")

class ProcessadorTexto(Generic[T]):

    def __init__(self):
        self._estrategia: Optional[ITratamento[T]] = None

    @property
    def estrategia(self) -> Optional[ITratamento[T]]:
        return self._estrategia

    @estrategia.setter
    def estrategia(self, nova_estrategia: ITratamento[T]) -> None:
        self._estrategia = nova_estrategia

    def processar(self, dados: Union[str, List[str]]) -> Union[T, List[T]]:
        if self._estrategia is None:
            raise ValueError("Estratégia de tratamento não foi definida.")

        if isinstance(dados, str):
            self._estrategia.comentario = dados
            return self._estrategia.executar_tratamento()
        elif isinstance(dados, list):
            resultados: List[T] = []
            for item in dados:
                self._estrategia.comentario = item
                resultados.append(self._estrategia.executar_tratamento())
            return resultados
        else:
            raise TypeError("Dados devem ser str ou List[str]")