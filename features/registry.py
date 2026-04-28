from features.base import FeatureExtractor


class FeatureRegistry:
    def __init__(self):
        self._items: dict[str, FeatureExtractor] = {}

    def register(self, extractor: FeatureExtractor) -> None:
        self._items[extractor.name] = extractor

    def get(self, name: str) -> FeatureExtractor:
        if name not in self._items:
            raise ValueError(f"Unknown feature extractor: {name}")
        return self._items[name]

    def list_features(self) -> list[str]:
        return sorted(self._items.keys())
