from ts_featurelab.features.base import FeatureExtractor


class FeatureRegistry:
    """Mutable registry that maps feature names to extractor instances."""

    def __init__(self):
        """Initialize an empty feature registry."""
        self._items: dict[str, FeatureExtractor] = {}

    def register(self, extractor: FeatureExtractor) -> None:
        """Register or replace a feature extractor by its ``name``.

        Args:
            extractor: Feature extractor instance to make available.
        """
        self._items[extractor.name] = extractor

    def get(self, name: str) -> FeatureExtractor:
        """Return a registered extractor by name.

        Args:
            name: Feature extractor name.

        Returns:
            Registered extractor instance.

        Raises:
            ValueError: If no extractor is registered for ``name``.
        """
        if name not in self._items:
            raise ValueError(f"Unknown feature extractor: {name}")
        return self._items[name]

    def list_features(self) -> list[str]:
        """List registered feature names.

        Returns:
            Sorted list of feature names.
        """
        return sorted(self._items.keys())
