# ComfyUI Ideogram 4 Resolution-Aware Scheduler (Fixed)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A lightweight custom node for ComfyUI that implements the official resolution-aware timestep scheduler for Ideogram 4, written in pure PyTorch.

---

## The Problem & Theoretical Justification

Flow-matching models like Ideogram 4 utilize a logit-normal noise schedule. The dynamics of flow matching are highly sensitive to the spatial dimensions of the target image relative to the resolution used during the model's base training phase ($512 \times 512$ pixels).

When generating at higher resolutions (e.g., $1024 \times 1024$ or above), the model needs to allocate more denoising steps to the initial high-noise stages (near $t = 1$) to successfully establish the global structure of the image. 

Without adjusting the schedule's mean parameter ($\mu$), the denoising path progresses to the low-noise stages too quickly. This often results in:
* **Structural incoherence:** Multiple heads, limbs, or distorted global anatomy.
* **Severe graininess / noise artifacts:** The model attempts to resolve high-frequency details before the base shapes are solidified.

To counter this, the official Diffusers implementation of Ideogram 4 dynamically shifts the mean of the logit-normal distribution based on the target image resolution. This custom node replicates that precise behavior.

---

## Mathematical Formulation

The scheduler performs the following mathematical operations to generate the final noise levels (sigmas):

### 1. Resolution-Aware Mean Shift
First, the effective mean $\mu_{\text{eff}}$ is computed by scaling the baseline $\mu$ relative to the $512 \times 512$ training resolution:

$$\mu_{\text{eff}} = \mu + 0.5 \ln\left(\frac{\text{width} \times \text{height}}{512^2}\right)$$

*This logarithmic scaling shifts the sampling distribution towards higher noise levels as the pixel count increases.*

### 2. Quantile Generation
We generate uniform quantiles $q$ representing the sampling steps, excluding the boundary values $0$ and $1$ to prevent infinite logits during normal distribution mapping:

$$q_i = \frac{i}{\text{steps} + 1} \quad \text{for } i \in [1, \text{steps}]$$

### 3. Inverse Normal CDF Mapping
The quantiles are mapped to a normal distribution using the inverse complementary error function ($\text{erfcinv}$):

$$z_i = \mu_{\text{eff}} + \sigma \cdot \sqrt{2} \cdot \text{erfcinv}(2(1-q_i))$$

#### PyTorch Optimization:
To eliminate CPU-GPU context-switching overhead and drop external dependencies like NumPy or SciPy, we utilize the mathematical identity:
$$\text{erfcinv}(y) = \text{erfinv}(1 - y)$$

Substituting $y = 2(1-q_i)$ yields:
$$1 - 2(1-q_i) = 2q_i - 1$$

Thus, the formula is computed natively and efficiently on the GPU/CPU inside PyTorch as:

$$z_i = \mu_{\text{eff}} + \sigma \cdot \sqrt{2} \cdot \text{erfinv}(2q_i - 1)$$

### 4. Sigmoid Projection
Finally, the latent coordinates $z_i$ are mapped back to the $[0, 1]$ timestep interval via the sigmoid function:

$$t_i = \frac{1}{1 + e^{-z_i}}$$

The resulting tensor is sorted in descending order ($t_{\text{start}} \to t_{\text{end}}$) and appended with $0.0$ to denote the end of the diffusion process according to ComfyUI standards.

---

## Features

- **Pure PyTorch Implementation:** Zero dependency on SciPy or NumPy. Runs entirely within the PyTorch ecosystem for optimal performance.
- **Improved Generation Quality:** Eliminates structural inconsistencies and muddy textures on non-square and high-resolution aspect ratios.
- **Drop-in Compatibility:** Exposes the exact same interface and output type (`SIGMAS`) as standard ComfyUI schedulers.

---

## Installation

1. Navigate to your ComfyUI installation directory, then go to `custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Clone this repository:
   ```bash
   git clone https://github.com/Alexdnk/comfyui-ideogram4-scheduler-fixed.git
   ```
3. Restart ComfyUI.

---

## Usage

Replace the default scheduler node in your Ideogram 4 workflow with **Ideogram 4 Scheduler (Fixed)**.

### Input Parameters
* **`steps`**: The number of sampling steps.
* **`width` / `height`**: Target resolution. (Note: Ideogram 4 latents require multiples of 16).
* **`mu`**: Baseline mean of the logit-normal schedule (Default: `0.0`).
* **`std`**: Standard deviation of the logit-normal schedule (Default: `1.75`).
