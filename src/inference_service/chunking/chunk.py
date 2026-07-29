from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: int
    order: int
    text: str
    length: int
