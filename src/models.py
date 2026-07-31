from dataclasses import dataclass, field


@dataclass
class TextChunk:
    page_content: str
    metadata: dict = field(default_factory=dict)
