"""
Outpaint to Target — pads an image for outpainting so the result lands on an
exact resolution or aspect ratio, without calculating the padding by hand.

Drop this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
Output is the same as core "Pad Image for Outpainting": IMAGE with 0.5-grey
padding, plus a MASK (1 = area to generate).
"""

import math
import torch
import torch.nn.functional as F

RATIOS = ["1:1", "4:5", "3:4", "2:3", "3:5", "9:16", "1:2", "9:21",
          "custom (target_width:target_height)"]
ANCHORS = ["center", "top", "bottom", "left", "right",
           "top-left", "top-right", "bottom-left", "bottom-right"]


def _round_up(v, m):
    return int(math.ceil(v / m) * m) if m > 1 else int(math.ceil(v))


def _round_near(v, m):
    return max(m, int(round(v / m) * m)) if m > 1 else max(1, int(round(v)))


def _place(anchor, free_w, free_h):
    """Return (left, top) offset of the original inside the canvas."""
    if "left" in anchor:
        left = 0
    elif "right" in anchor:
        left = free_w
    else:
        left = free_w // 2
    if "top" in anchor:
        top = 0
    elif "bottom" in anchor:
        top = free_h
    else:
        top = free_h // 2
    return left, top


def _resize(image, w, h):
    # image: [B,H,W,C] 0..1
    if image.shape[1] == h and image.shape[2] == w:
        return image
    x = image.movedim(-1, 1)
    x = F.interpolate(x, size=(h, w), mode="bicubic", antialias=True, align_corners=False)
    return x.movedim(1, -1).clamp(0.0, 1.0)


class OutpaintToTarget:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["target_resolution", "aspect_ratio", "manual_pixels"], {
                    "tooltip": "target_resolution: exact WxH (image is scaled to fit, rest is outpainted)\n"
                               "aspect_ratio: keep original pixels, extend to the chosen ratio\n"
                               "manual_pixels: classic left/top/right/bottom padding"}),
                "target_width": ("INT", {"default": 1280, "min": 16, "max": 16384, "step": 8}),
                "target_height": ("INT", {"default": 720, "min": 16, "max": 16384, "step": 8}),
                "aspect_ratio": (RATIOS, {"default": "9:16"}),
                "orientation": (["landscape", "portrait"], {
                    "tooltip": "For aspect_ratio presets. 3:4 portrait = 3 wide x 4 tall, landscape = 4 wide x 3 tall."}),
                "anchor": (ANCHORS, {"default": "center",
                    "tooltip": "Where the original image sits. 'top' = extend downward, 'left' = extend to the right, etc."}),
                "scale_mode": (["fit (scale to fill target)", "shrink only (never upscale)"], {
                    "tooltip": "target_resolution only. 'shrink only' keeps small images at native size, so more gets outpainted."}),
                "manual_left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "manual_top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "manual_right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "manual_bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 128}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "width", "height")
    FUNCTION = "pad"
    CATEGORY = "image"

    def pad(self, image, mode, target_width, target_height, aspect_ratio, orientation,
            anchor, scale_mode, manual_left, manual_top, manual_right, manual_bottom,
            divisible_by, feathering):
        d = max(1, divisible_by)
        _, h, w, _ = image.shape

        if mode == "target_resolution":
            W, H = _round_near(target_width, d), _round_near(target_height, d)
            s = min(W / w, H / h)
            if scale_mode.startswith("shrink"):
                s = min(s, 1.0)
            nw, nh = min(W, max(1, round(w * s))), min(H, max(1, round(h * s)))
            image = _resize(image, nw, nh)
            left, top = _place(anchor, W - nw, H - nh)

        elif mode == "aspect_ratio":
            if aspect_ratio.startswith("custom"):
                rw, rh = target_width, target_height
            else:
                a, b = (int(x) for x in aspect_ratio.split(":"))
                lo, hi = min(a, b), max(a, b)
                rw, rh = (hi, lo) if orientation == "landscape" else (lo, hi)
            k = max(w / rw, h / rh)
            W, H = _round_up(rw * k, d), _round_up(rh * k, d)
            W, H = max(W, _round_up(w, d)), max(H, _round_up(h, d))
            nw, nh = w, h
            left, top = _place(anchor, W - nw, H - nh)

        else:  # manual_pixels
            nw, nh = w, h
            W, H = nw + manual_left + manual_right, nh + manual_top + manual_bottom
            # round the canvas up, adding the extra on the padded sides
            eW, eH = _round_up(W, d) - W, _round_up(H, d) - H
            W, H = W + eW, H + eH
            left = manual_left + (eW if manual_left and not manual_right else 0)
            top = manual_top + (eH if manual_top and not manual_bottom else 0)

        right, bottom = W - nw - left, H - nh - top

        out = torch.full((image.shape[0], H, W, image.shape[3]), 0.5,
                         dtype=image.dtype, device=image.device)
        out[:, top:top + nh, left:left + nw, :] = image

        mask = torch.ones((H, W), dtype=torch.float32, device=image.device)
        inner = torch.zeros((nh, nw), dtype=torch.float32, device=image.device)
        if feathering > 0 and feathering * 2 < nh and feathering * 2 < nw:
            i = torch.arange(nh, device=image.device).float().unsqueeze(1).expand(nh, nw)
            j = torch.arange(nw, device=image.device).float().unsqueeze(0).expand(nh, nw)
            big = float(max(nh, nw))
            dt = i if top else torch.full_like(i, big)
            db = (nh - i) if bottom else torch.full_like(i, big)
            dl = j if left else torch.full_like(j, big)
            dr = (nw - j) if right else torch.full_like(j, big)
            dist = torch.minimum(torch.minimum(dt, db), torch.minimum(dl, dr))
            v = ((feathering - dist) / feathering).clamp(min=0)
            inner = v * v
        mask[top:top + nh, left:left + nw] = inner

        print(f"[OutpaintToTarget] {w}x{h} -> placed {nw}x{nh} at ({left},{top}) "
              f"on {W}x{H} canvas | pad L{left} T{top} R{right} B{bottom}")
        return (out, mask.unsqueeze(0), W, H)


def _gauss_blur(m, sigma):
    """m: [H,W] float. Separable gaussian blur."""
    if sigma <= 0:
        return m
    r = int(math.ceil(sigma * 3))
    x = torch.arange(-r, r + 1, device=m.device, dtype=m.dtype)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = k / k.sum()
    t = m[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    return t[0, 0]


class OutpaintPasteOriginal:
    """Paste untouched original pixels back over the generated image.

    With protect_mask connected (e.g. the face painted in Load Image's mask
    editor) only that area is pasted, with a soft edge, so the rest of the
    image stays coherent with the extension. Without it, the whole original
    area is pasted and only the outpaint seam is blended.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated": ("IMAGE",),
                "original_padded": ("IMAGE", {"tooltip": "IMAGE output of 'Outpaint to Target' (or Pad Image for Outpainting)"}),
                "mask": ("MASK", {"tooltip": "MASK output of the same pad node (1 = generated area)"}),
                "blend_px": ("INT", {"default": 48, "min": 0, "max": 1024, "step": 1,
                    "tooltip": "Softness of the pasted edge, in output pixels. "
                               "With protect_mask: feather around the painted area. "
                               "Without: seam width along the outpaint edge."}),
                "grow_px": ("INT", {"default": 16, "min": 0, "max": 512, "step": 1,
                    "tooltip": "protect_mask only: expand the painted area before feathering, "
                               "so the feather falls outside the face rather than on it."}),
                "color_match": ("BOOLEAN", {"default": True,
                    "tooltip": "Shift the generated image's colours to match the original before pasting."}),
            },
            "optional": {
                "protect_mask": ("MASK", {"tooltip": "Area to keep 100% original, in ORIGINAL image coordinates "
                                          "(paint it on the Load Image node). Leave empty to paste the whole original area."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "paste_alpha")
    FUNCTION = "paste"
    CATEGORY = "image"

    def paste(self, generated, original_padded, mask, blend_px, grow_px, color_match, protect_mask=None):
        G = generated
        B, H, W, C = G.shape
        dev = G.device
        S = _resize(original_padded.to(dev, G.dtype), W, H)[..., :C]
        if S.shape[0] != B:
            S = S[:1].expand(B, -1, -1, -1)

        m = mask.to(dev).float()
        if m.dim() == 2:
            m = m.unsqueeze(0)
        m = F.interpolate(m[:1].unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False)[0, 0]

        keep = m < 0.5
        rows = torch.nonzero(keep.any(dim=1)).flatten()
        cols = torch.nonzero(keep.any(dim=0)).flatten()
        if rows.numel() == 0 or cols.numel() == 0:
            print("[OutpaintPasteOriginal] no original area found in mask, returning generated image")
            return (G, torch.zeros((1, H, W), device=dev))
        y0, y1 = int(rows[0]), int(rows[-1]) + 1
        x0, x1 = int(cols[0]), int(cols[-1]) + 1
        h, w = y1 - y0, x1 - x0

        if color_match:
            g = G[:, y0:y1, x0:x1, :].reshape(B, -1, C)
            s = S[:, y0:y1, x0:x1, :].reshape(B, -1, C)
            gm, gs = g.mean(1), g.std(1).clamp(min=1e-4)
            sm, ss = s.mean(1), s.std(1)
            ratio = (ss / gs).clamp(0.8, 1.25)
            G = ((G - gm[:, None, None, :]) * ratio[:, None, None, :] + sm[:, None, None, :]).clamp(0.0, 1.0)

        pm = None
        if protect_mask is not None:
            pm = protect_mask.to(dev).float()
            if pm.dim() == 2:
                pm = pm.unsqueeze(0)
            pm = pm[:1]
            if pm.max() < 0.5:
                print("[OutpaintPasteOriginal] protect_mask is empty (nothing painted) - pasting whole original area")
                pm = None

        # seam alpha: 1 inside the original rect, fading to 0 toward padded edges only
        big = float(max(H, W))
        i = torch.arange(h, device=dev).float().unsqueeze(1).expand(h, w)
        j = torch.arange(w, device=dev).float().unsqueeze(0).expand(h, w)
        dt = i if y0 > 0 else torch.full_like(i, big)
        db = (h - 1 - i) if y1 < H else torch.full_like(i, big)
        dl = j if x0 > 0 else torch.full_like(j, big)
        dr = (w - 1 - j) if x1 < W else torch.full_like(j, big)
        dist = torch.minimum(torch.minimum(dt, db), torch.minimum(dl, dr))
        if blend_px > 0:
            a = (dist / blend_px).clamp(0, 1)
            a = a * a * (3 - 2 * a)  # smoothstep
        else:
            a = torch.ones_like(dist)
        seam = torch.zeros((H, W), device=dev, dtype=torch.float32)
        seam[y0:y1, x0:x1] = a

        if pm is not None:
            # map from original-image coords into the original's rectangle on the output canvas
            p = F.interpolate(pm.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[0, 0]
            alpha = torch.zeros((H, W), device=dev, dtype=torch.float32)
            alpha[y0:y1, x0:x1] = (p > 0.5).float()
            if grow_px > 0:
                alpha = F.max_pool2d(alpha[None, None], 2 * grow_px + 1, stride=1, padding=grow_px)[0, 0]
            alpha = _gauss_blur(alpha, blend_px / 2.0).clamp(0, 1) * seam  # never paste into the padding
            alpha = torch.where(alpha > 0.999, torch.ones_like(alpha), alpha)
            area = "protect_mask"
        else:
            alpha = seam
            area = "whole original"

        a4 = alpha.to(G.dtype)[None, :, :, None]
        out = S * a4 + G * (1 - a4)
        print(f"[OutpaintPasteOriginal] {area} | original rect ({x0},{y0})-({x1},{y1}) on {W}x{H} | "
              f"blend {blend_px}px grow {grow_px}px color_match={color_match}")
        return (out, alpha.unsqueeze(0))


NODE_CLASS_MAPPINGS = {"OutpaintToTarget": OutpaintToTarget,
                       "OutpaintPasteOriginal": OutpaintPasteOriginal}
NODE_DISPLAY_NAME_MAPPINGS = {"OutpaintToTarget": "Outpaint to Target (resolution / ratio)",
                              "OutpaintPasteOriginal": "Outpaint Paste Original Back (zero face change)"}
