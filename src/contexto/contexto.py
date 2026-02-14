from dataclasses import dataclass, field
from typing import Iterable, Any


@dataclass
class Contexto:
    gerador_reviews_steam: Iterable[Any] = field(default_factory=list)
    gerador_comentarios_youtube: Iterable[Any] = field(default_factory=list)
    gerador_resposta_comentarios_youtube: Iterable[Any] = field(default_factory=list)
