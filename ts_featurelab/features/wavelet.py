import math

from ts_featurelab.features.base import SingleColumnFeatureExtractor
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
    name = "wavelet"

    def extract(self, context: FeatureContext, spec: FeatureSpec) -> dict:
        if pywt is None:
            raise ImportError(
                "WaveletFeature requires a working PyWavelets installation"
            ) from _PYWT_IMPORT_ERROR

        col = self.require_column(spec)
        prefix = spec.alias or col
        level = int(spec.params.get("level", 4))
        wavelet = spec.params.get("wavelet", "db4")
        min_points = int(spec.params.get("min_points", 64))

        series = context.raw()[col].drop_nulls()
        x = series.to_numpy().copy()

        if len(x) < min_points:
            return {}

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

        return out
