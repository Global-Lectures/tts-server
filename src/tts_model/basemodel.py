from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BaseServiceModel(ABC):
    target_lang: str
    support_language: dict[str, str]

    def __post_init__(self):
        self.target_lang = self.support_language[self.target_lang]

    def run(
        self,
        text: str,
        speaker_wav: str
    ):
        self.init_model()
        result = self.process(
            text=text,
            speaker_wav=speaker_wav
        )
        self.clear_memory()
        return result

    @abstractmethod
    def init_model(self):
        ...

    @abstractmethod
    def process(
        self,
        text: str,
        speaker_wav: str
    ):
        ...

    @abstractmethod
    def clear_memory(self):
        ...
