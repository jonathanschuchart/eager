from abc import ABC, abstractmethod


class Embedding(ABC):
    @abstractmethod
    def fit(self, entities):
        pass

    @abstractmethod
    def embed(self, entities):
        pass
