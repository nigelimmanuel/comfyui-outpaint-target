# Outpaint to Target

Two ComfyUI nodes for outpainting:

**Outpaint to Target (resolution / ratio)** pads an image for outpainting without hand-calculating the padding. It outputs IMAGE + MASK like core *Pad Image for Outpainting*, plus the final width and height.

| mode | what it does |
|---|---|
| `target_resolution` | Exact final size (e.g. 1280×720). The image is scaled to fit and the rest is outpainted. |
| `aspect_ratio` | Extends to a ratio (1:1, 4:5, 3:4, 2:3, 3:5, 9:16, 1:2, 9:21 or custom) with portrait/landscape. Original pixels are not resized. |
| `manual_pixels` | Classic left/top/right/bottom padding. |

`anchor` sets where the original sits (`top` = extend downward, `center` = extend both sides, …). Sizes snap to `divisible_by` (default 16).

**Outpaint Paste Original Back** composites the untouched original over the generated image after VAE Decode, so the original area (faces included) has zero alteration. A soft seam of `blend_px` blends only along the outpaint edge, and `color_match` shifts the generated colours to match the original.

```
Load Image → Outpaint to Target ─IMAGE/MASK─┐
                     └→ (encode, sample, VAE Decode) → Paste Original Back → Save
```

## Install

ComfyUI Manager → search **Outpaint to Target**, or:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_GITHUB_USERNAME/comfyui-outpaint-target
```

Restart ComfyUI. No extra dependencies (uses torch only).
