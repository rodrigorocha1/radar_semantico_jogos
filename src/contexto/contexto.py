from dataclasses import dataclass, field
from typing import Iterable, Any, List, Tuple


@dataclass
class Contexto:
    gerador_reviews_steam: Iterable[Any] = field(default_factory=list)
    gerador_comentarios_youtube: Iterable[Any] = field(default_factory=list)
    lista_id_comentarios: List[Tuple[str, ...]] = field(default_factory=list)
    gerador_resposta_comentarios : Iterable[Any] = field(default_factory=list)