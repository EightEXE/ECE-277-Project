import copy
from typing import Dict

import numpy as np

COLOR_NAMES = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]

HUE_BANDS = {
    "red": (0.0, 60.0),
    "orange": (30.0, 50.0),
    "yellow": (60.0, 50.0),
    "green": (125.0, 80.0),
    "aqua": (180.0, 50.0),
    "blue": (240.0, 70.0),
    "purple": (280.0, 50.0),
    "magenta": (320.0, 70.0),
}


def default_color_params() -> dict:
    """Factory for a fresh, no-op color parameter tree."""
    return {
        "white_balance": {
            "enabled": False,
            "temperature": 0.0,
            "tint": 0.0,
            "preset": "As Shot",
            # multipliers derived from picker; kept as list for JSON friendliness
            "multipliers": [1.0, 1.0, 1.0],
        },
        "hsl": {
            "enabled": True,  # defaults to on but neutral sliders = no-op
            "hue": {c: 0.0 for c in COLOR_NAMES},
            "sat": {c: 0.0 for c in COLOR_NAMES},
            "lum": {c: 0.0 for c in COLOR_NAMES},
        },
        "recolor": {
            "enabled": False,
            "target_hue": 0.0,  # degrees
            "strength": 0.0,    # 0-1
            "saturation": 0.0,  # -1..1
            "preserve_luminance": True,
            "mode": "Tint",
        },
        "black_white": {
            "enabled": False,
            "mix": {c: 0.0 for c in COLOR_NAMES},
            "contrast_boost": 0.0,
        },
        "selective_color": {
            "enabled": False,
            "range": "Reds",
            "mode": "Relative",
            "cyan": 0.0,
            "magenta": 0.0,
            "yellow": 0.0,
            "black": 0.0,
        },
        "color_balance": {
            "enabled": False,
            "preserve_luminance": True,
            "shadows": {"cr": 0.0, "mg": 0.0, "yb": 0.0},
            "midtones": {"cr": 0.0, "mg": 0.0, "yb": 0.0},
            "highlights": {"cr": 0.0, "mg": 0.0, "yb": 0.0},
        },
    }


def _merge_dict(target: dict, src: dict):
    """Recursive dict merge with copying for safety."""
    for key, val in (src or {}).items():
        if isinstance(val, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], val)
        else:
            target[key] = copy.deepcopy(val)


def merge_color_params(user_params: dict | None, flat_params: dict | None = None) -> dict:
    """
    Merge provided color params with defaults.
    flat_params supports legacy flat hsl_* keys.
    """
    merged = default_color_params()
    if user_params:
        _merge_dict(merged, user_params)

    # Legacy HSL values (flat params)
    if flat_params:
        hsl_section = merged.setdefault("hsl", {})
        hsl_section.setdefault("hue", {c: 0.0 for c in COLOR_NAMES})
        hsl_section.setdefault("sat", {c: 0.0 for c in COLOR_NAMES})
        hsl_section.setdefault("lum", {c: 0.0 for c in COLOR_NAMES})

        for c in COLOR_NAMES:
            for kind, key_prefix in [
                ("hue", "hsl_hue_"),
                ("sat", "hsl_sat_"),
                ("lum", "hsl_lum_"),
            ]:
                flat_key = f"{key_prefix}{c}"
                if flat_key in flat_params:
                    hsl_section[kind][c] = float(flat_params.get(flat_key, 0))
                    if abs(hsl_section[kind][c]) > 1e-6:
                        hsl_section["enabled"] = True

    return merged


def _angle_diff_norm(a: np.ndarray, b: float) -> np.ndarray:
    """Shortest angular distance on [0,1) hue wheel (where 1 == 360°)."""
    return np.abs(((a - b + 0.5) % 1.0) - 0.5)


def rgb_to_hsv(rgb: np.ndarray):
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    delta = maxc - minc

    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # hue in [0,1)
    h[mask & (maxc == r)] = ((g - b)[mask & (maxc == r)] / delta[mask & (maxc == r)] % 6.0) / 6.0
    h[mask & (maxc == g)] = (((b - r)[mask & (maxc == g)] / delta[mask & (maxc == g)]) + 2.0) / 6.0
    h[mask & (maxc == b)] = (((r - g)[mask & (maxc == b)] / delta[mask & (maxc == b)]) + 4.0) / 6.0
    h = np.mod(h, 1.0)

    s = np.zeros_like(maxc)
    denom = np.where(maxc > 1e-6, maxc, 1.0)
    s[mask] = delta[mask] / denom[mask]
    v = maxc
    return h, s, v


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h6 = h * 6.0
    i = np.floor(h6).astype(np.int32)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    rgb = np.zeros(h.shape + (3,), dtype=np.float32)
    i_mod = np.mod(i, 6)

    mask0 = i_mod == 0
    rgb[mask0] = np.stack((v[mask0], t[mask0], p[mask0]), axis=-1)

    mask1 = i_mod == 1
    rgb[mask1] = np.stack((q[mask1], v[mask1], p[mask1]), axis=-1)

    mask2 = i_mod == 2
    rgb[mask2] = np.stack((p[mask2], v[mask2], t[mask2]), axis=-1)

    mask3 = i_mod == 3
    rgb[mask3] = np.stack((p[mask3], q[mask3], v[mask3]), axis=-1)

    mask4 = i_mod == 4
    rgb[mask4] = np.stack((t[mask4], p[mask4], v[mask4]), axis=-1)

    mask5 = i_mod == 5
    rgb[mask5] = np.stack((v[mask5], p[mask5], q[mask5]), axis=-1)

    return rgb


def apply_color_white_balance(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    temp = float(params.get("temperature", 0.0))
    tint = float(params.get("tint", 0.0))
    multipliers = params.get("multipliers", [1.0, 1.0, 1.0]) or [1.0, 1.0, 1.0]

    # Temperature: warm cool scale on R/B
    temp_scale = np.clip(temp / 100.0, -1.0, 1.0)
    r_gain = 1.0 + temp_scale * 0.15
    b_gain = 1.0 - temp_scale * 0.15

    # Tint: magenta/green bias on G
    tint_scale = np.clip(tint / 100.0, -1.0, 1.0)
    g_gain = 1.0 - tint_scale * 0.12
    r_gain *= 1.0 + tint_scale * 0.06
    b_gain *= 1.0 + tint_scale * 0.06

    gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    gains *= np.array(multipliers, dtype=np.float32)

    balanced = image * gains[None, None, :]
    max_gain = np.max(gains)
    if max_gain > 1.2:
        balanced = balanced / max_gain
    return np.clip(balanced, 0.0, 1.0)


def apply_hsl(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    hue_shifted = image
    h, s, v = rgb_to_hsv(hue_shifted)
    any_change = False

    for color_name in COLOR_NAMES:
        center_deg, width_deg = HUE_BANDS.get(color_name, (0.0, 60.0))
        center = center_deg / 360.0
        width = max(width_deg / 360.0, 1e-3)

        w = np.exp(-0.5 * (_angle_diff_norm(h, center) / (width * 0.5)) ** 2)

        hue_delta = float(params.get("hue", {}).get(color_name, 0.0))
        sat_delta = float(params.get("sat", {}).get(color_name, 0.0))
        lum_delta = float(params.get("lum", {}).get(color_name, 0.0))

        if abs(hue_delta) > 1e-3:
            any_change = True
            h = np.mod(h + (hue_delta / 100.0) * (60.0 / 360.0) * w, 1.0)
        if abs(sat_delta) > 1e-3:
            any_change = True
            s = np.clip(s * (1.0 + (sat_delta / 100.0) * w), 0.0, 2.5)
        if abs(lum_delta) > 1e-3:
            any_change = True
            v = np.clip(v * (1.0 + (lum_delta / 100.0) * 0.8 * w), 0.0, 2.0)

    if not any_change:
        return image

    return np.clip(hsv_to_rgb(h, s, v), 0.0, 1.0)


def apply_recolor(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    target_hue = (float(params.get("target_hue", 0.0)) % 360.0) / 360.0
    strength = np.clip(float(params.get("strength", 0.0)) / 100.0, 0.0, 1.0)
    sat_delta = float(params.get("saturation", 0.0)) / 100.0
    preserve_lum = bool(params.get("preserve_luminance", True))
    mode = (params.get("mode") or "Tint").title()

    h, s, v = rgb_to_hsv(image)
    base_v = v.copy()

    if strength > 0:
        if mode == "Colorize":
            h = np.mod(target_hue + (h - target_hue) * (1.0 - strength), 1.0)
            s = np.clip((1.0 - strength) * s + strength * 0.9, 0.0, 1.0)
        else:  # Tint
            h = np.mod(h * (1.0 - strength) + target_hue * strength, 1.0)

    if abs(sat_delta) > 1e-3:
        s = np.clip(s * (1.0 + sat_delta), 0.0, 2.5)

    if preserve_lum:
        v = base_v

    return np.clip(hsv_to_rgb(h, s, v), 0.0, 1.0)


def apply_black_white(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    h, _, _ = rgb_to_hsv(image)
    gray = (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2])

    mix = params.get("mix", {}) or {}
    for color_name in COLOR_NAMES:
        slider = float(mix.get(color_name, 0.0))
        if abs(slider) < 1e-3:
            continue
        center_deg, width_deg = HUE_BANDS.get(color_name, (0.0, 60.0))
        center = center_deg / 360.0
        width = max(width_deg / 360.0, 1e-3)
        weight = np.exp(-0.5 * (_angle_diff_norm(h, center) / (width * 0.5)) ** 2)
        gray += weight * (slider / 100.0) * gray

    contrast_boost = float(params.get("contrast_boost", 0.0)) / 100.0
    if abs(contrast_boost) > 1e-3:
        gray = np.clip((gray - 0.5) * (1.0 + contrast_boost) + 0.5, 0.0, 1.0)

    gray3 = np.clip(gray[..., None], 0.0, 1.0)
    return np.repeat(gray3, 3, axis=2)


def _range_weight(h: np.ndarray, L: np.ndarray, range_name: str):
    """Weight per Photoshop-like range."""
    if range_name in {"Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas"}:
        centers = {
            "Reds": HUE_BANDS["red"],
            "Yellows": HUE_BANDS["yellow"],
            "Greens": HUE_BANDS["green"],
            "Cyans": HUE_BANDS["aqua"],
            "Blues": HUE_BANDS["blue"],
            "Magentas": HUE_BANDS["magenta"],
        }
        center_deg, width_deg = centers.get(range_name, (0.0, 60.0))
        center = center_deg / 360.0
        width = max(width_deg / 360.0, 1e-3)
        hue_w = np.exp(-0.5 * (_angle_diff_norm(h, center) / (width * 0.5)) ** 2)
        return hue_w

    # Luminance-based groups
    if range_name == "Whites":
        return np.clip((L - 0.65) / 0.25, 0.0, 1.0)
    if range_name == "Blacks":
        return np.clip((0.35 - L) / 0.35, 0.0, 1.0)
    # Neutrals: midtones emphasis
    return np.exp(-((L - 0.5) ** 2) / (2 * 0.12 ** 2))


def apply_selective_color(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    h, s, v = rgb_to_hsv(image)
    L = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])

    rng = params.get("range", "Reds")
    weight = _range_weight(h, L, rng)
    if np.max(weight) < 1e-4:
        return image

    mode = (params.get("mode") or "Relative").title()
    cyan = float(params.get("cyan", 0.0)) / 100.0
    magenta = float(params.get("magenta", 0.0)) / 100.0
    yellow = float(params.get("yellow", 0.0)) / 100.0
    black = float(params.get("black", 0.0)) / 100.0

    C = 1.0 - image[..., 0]
    M = 1.0 - image[..., 1]
    Y = 1.0 - image[..., 2]
    K = np.minimum(np.minimum(C, M), Y)

    if mode == "Absolute":
        C = np.clip(C + cyan * 0.35, 0.0, 1.5)
        M = np.clip(M + magenta * 0.35, 0.0, 1.5)
        Y = np.clip(Y + yellow * 0.35, 0.0, 1.5)
        K = np.clip(K + black * 0.35, 0.0, 1.5)
    else:
        C = np.clip(C * (1.0 + cyan), 0.0, 1.5)
        M = np.clip(M * (1.0 + magenta), 0.0, 1.5)
        Y = np.clip(Y * (1.0 + yellow), 0.0, 1.5)
        K = np.clip(K * (1.0 + black), 0.0, 1.5)

    adj_rgb = np.clip(1.0 - np.stack([C, M, Y], axis=-1), 0.0, 1.0)
    adj_rgb *= (1.0 - K[..., None]) + 1e-6

    weight3 = weight[..., None]
    return np.clip(image * (1.0 - weight3) + adj_rgb * weight3, 0.0, 1.0)


def _tone_masks_from_L(L: np.ndarray):
    shadows = 1.0 - np.clip((L - 0.30) / 0.25, 0.0, 1.0)
    highlights = np.clip((L - 0.55) / 0.25, 0.0, 1.0)
    mids = np.clip(1.0 - shadows - highlights, 0.0, 1.0)
    total = shadows + mids + highlights + 1e-6
    return shadows / total, mids / total, highlights / total


def apply_color_balance(image: np.ndarray, params: dict) -> np.ndarray:
    if not params or not params.get("enabled", False):
        return image

    preserve = bool(params.get("preserve_luminance", True))
    L = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2])
    sh_mask, mid_mask, hi_mask = _tone_masks_from_L(L)

    def bias_from_section(section: Dict[str, float]):
        cr = float(section.get("cr", 0.0)) / 100.0
        mg = float(section.get("mg", 0.0)) / 100.0
        yb = float(section.get("yb", 0.0)) / 100.0
        return np.array([cr, mg, yb], dtype=np.float32) * 0.25

    offsets = {
        "shadows": bias_from_section(params.get("shadows", {})),
        "midtones": bias_from_section(params.get("midtones", {})),
        "highlights": bias_from_section(params.get("highlights", {})),
    }

    out = image.copy()
    masks = {"shadows": sh_mask, "midtones": mid_mask, "highlights": hi_mask}
    for name, mask in masks.items():
        weight = mask[..., None]
        bias = offsets[name]
        if preserve:
            bias = bias - np.mean(bias)
        out = np.clip(out + bias * weight, 0.0, 1.5)
    return np.clip(out, 0.0, 1.0)


def is_section_default(section_state: dict, defaults: dict) -> bool:
    """Shallow-ish comparison with tolerance for floats."""
    if not isinstance(section_state, dict):
        return False

    for key, default_val in defaults.items():
        if key not in section_state:
            return False
        val = section_state[key]
        if isinstance(default_val, dict):
            if not is_section_default(val, default_val):
                return False
        elif isinstance(default_val, (int, float)):
            if abs(float(val) - float(default_val)) > 1e-6:
                return False
        else:
            if val != default_val:
                return False
    return True

