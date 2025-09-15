from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class CrawledData:
    id: str
    source_type: str
    source_name: str
    title: str
    content: str
    url: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    full_content: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
