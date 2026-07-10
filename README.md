# ComfyUI Ideogram 4 Scheduler (Optimized Grid)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An optimized, pure PyTorch implementation of the Ideogram 4 logit-normal timestep scheduler for ComfyUI. 

While the official ComfyUI implementation correctly scales the resolution with the $512 \times 512$ base training constant, this node employs a mathematically cleaner grid allocation method that improves generation stability and structural coherence on non-square and high-resolution aspect ratios.

---

## Technical Details: What makes it different?

The core difference between the built-in ComfyUI scheduler and this version lies in **how the sampling grid is constructed** and how the boundaries of the logit-normal distribution are handled.

### 1. Endpoint Boundary Handling (Avoiding Clamping)
* **The Built-in Scheduler:** 
  Generates quantiles $u \in [0, 1]$ *including* the extreme endpoints $0.0$ and $1.0$. Because these endpoints map to $-\infty$ and $+\infty$ in the inverse normal CDF ($\text{ndtri}$), the official code relies on hard-coded signal-to-noise ratio limits (`_LOGSNR_MIN = -15.0` and `_LOGSNR_MAX = 18.0`) to manually clamp the outputs.
* **This Optimized Scheduler:** 
  Generates quantiles $q \in (0, 1)$ by slicing a larger interval `[1:-1]`. By naturally excluding the boundary endpoints, the logits never reach infinity. The sampling curve is mathematically self-contained and avoids artificial clipping.

### 2. Unstable Extreme Noise Cutoff (First Sigma)
* **The Built-in Scheduler:** Always begins generation at a clamped maximum noise level ($\sigma \approx 0.9999$).
* **This Optimized Scheduler:** Due to the boundary exclusion, the maximum starting quantile at 20 steps is $20/21 \approx 0.952$. This scales the initial sigma to a slightly lower range ($\approx 0.94 - 0.97$ depending on resolution).

*By skipping the initial fraction of absolute, chaotic noise (where the model does not build geometry but is prone to generating chromatic artifacts), the sampler bypasses early phase instability, reducing overall graininess.*

### 3. Step Density in the "Sweet Spot"
Because the boundary values are omitted, all requested sampling steps are packed tighter within the highly active $[0.05, 0.95]$ noise range. This mid-range is the "sweet spot" where the model makes critical decisions regarding composition, typography, and human anatomy. A higher concentration of steps in this region yields cleaner details and better structural alignment.

---

## Mathematical Formulation

The scheduler computes the noise levels natively on the GPU/CPU using PyTorch:

$$\mu_{\text{eff}} = \mu + 0.5 \ln\left(\frac{\text{width} \times \text{height}}{512^2}\right)$$

$$q_i = \frac{i}{\text{steps} + 1} \quad \text{for } i \in [1, \text{steps}]$$

$$z_i = \mu_{\text{eff}} + \sigma \cdot \sqrt{2} \cdot \text{erfinv}(2q_i - 1)$$

$$t_i = \frac{1}{1 + e^{-z_i}}$$

The final tensor is reversed and padded with a terminal $0.0$ to comply with the ComfyUI pipeline.

---

## Features

- **No SciPy/NumPy overhead:** Native PyTorch tensor operations.
- **Improved High-Resolution Performance:** Reduces anatomy distortions and over-sharpened textures on non-square ratios.
- **Drop-in Compatibility:** Direct replacement for the standard scheduler node, outputting standard `SIGMAS`.

---

## Installation

1. Navigate to your ComfyUI `custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/Alexdnk/comfyui-ideogram4-scheduler-fixed.git
   ```
3. Restart ComfyUI.
