import math

from ts_featurelab.features.base import FeatureResult, SingleColumnFeatureExtractor
from ts_featurelab.features.context import FeatureContext
from ts_featurelab.features.featurespec import FeatureSpec

try:
    import pywt
except Exception as exc:  # pragma: no cover - depends on local binary env
    pywt = None
    _PYWT_IMPORT_ERROR = exc
else:
    _PYWT_IMPORT_ERROR = None


class WaveletFeature(SingleColumnFeatureExtractor):
    """Compute wavelet energy and dispersion features for one input series.

    Params:
        wavelet: PyWavelets wavelet name. Defaults to ``"db4"``.
        level: Wavelet decomposition level. Defaults to ``4``.
        min_points: Minimum number of non-null values required. Defaults to
            ``64``.
        resample: Optional Polars duration string used before extraction.
        agg: Aggregation applied during resampling. Defaults to ``"mean"``.
    """

    name = "wavelet"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> FeatureResult:
        """Extract wavelet-derived scalar features for the configured column.

        Args:
            context: Window-scoped feature context.
            spec: Feature specification with ``column`` and optional params.

        Returns:
            Scalar result with one or more columns prefixed by ``spec.alias``.
            Returns an empty result when there are fewer than ``min_points``.

        Raises:
            ImportError: If PyWavelets is unavailable or failed to import.
        """
        if pywt is None:
            raise ImportError(
                "WaveletFeature requires a working PyWavelets installation"
            ) from _PYWT_IMPORT_ERROR

        col = self.require_column(spec)
        prefix = spec.alias
        level = int(spec.params.get("level", 4))
        wavelet = spec.params.get("wavelet", "db4")
        min_points = int(spec.params.get("min_points", 64))

        series = context.get_aggregated_series(
            col,
            every=spec.params.get("resample"),
            agg=spec.params.get("agg", "mean"),
        ).drop_nulls()
        x = series.to_numpy().copy()

        if len(x) < min_points:
            return FeatureResult()

        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
        out: dict[str, float] = {}

        for idx, coeff in enumerate(coeffs):
            band = f"a{level}" if idx == 0 else f"d{level - idx + 1}"
            energy = float((coeff ** 2).sum())
            std = float(coeff.std()) if len(coeff) else 0.0
            out[f"{prefix}_wl_energy_{band}"] = energy
            out[f"{prefix}_wl_std_{band}"] = std

        if len(coeffs) > 1:
            detail_energy = sum(float((coeff ** 2).sum()) for coeff in coeffs[1:])
            first_detail_energy = float((coeffs[-1] ** 2).sum())
            ratio = 0.0 if math.isclose(detail_energy, 0.0) else first_detail_energy / detail_energy
            out[f"{prefix}_wl_ratio_d1"] = float(ratio)

        return FeatureResult.from_scalars(out)
