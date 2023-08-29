from dataclasses import dataclass
from typing import List, Optional
from gams import GamsEngineConfiguration


@dataclass
class EngineConfig:
    host: str
    username: Optional[str] = None
    password: Optional[str] = None
    jwt: Optional[str] = None
    namespace: str = "global"
    extra_model_files: Optional[List[str]] = None
    engine_options: Optional[dict] = None
    remove_results: bool = True

    def get_engine_config(self):
        return GamsEngineConfiguration(
            self.host,
            self.username,
            self.password,
            self.jwt,
            self.namespace,
        )
