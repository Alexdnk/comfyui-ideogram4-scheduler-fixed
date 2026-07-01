import torch
import numpy as np
from scipy import special

class Ideogram4SchedulerFixed:
    """
    Ideogram 4 Scheduler with the correct resolution-aware mu shift formula.

    Fully replicates the interface and output of the standard Ideogram4Scheduler node:
    inputs steps, width, height, mu, std → output SIGMAS (in ComfyUI format).

    Automatically applies the resolution-aware mu shift
    used in the official diffusers pipeline for Ideogram 4:

        mu_eff = mu + 0.5 * ln((width * height) / (512 * 512))

    Where 512×512 is the base training resolution.

    Algorithm:
        1. Resolution-aware mu shift: mu_eff = mu + 0.5 * ln(ratio)
        2. Quantile generation: q = linspace(0, 1, steps+2)[1:-1]
        3. Inverse normal CDF: z = mu_eff + std * sqrt(2) * erfcinv(2*(1-q))
        4. Sigmoid: t = 1 / (1 + exp(-z))
        5. Reverse order (ComfyUI convention: from high to low)
        6. Append 0 at the end (end of denoising)
        7. Output as torch.Tensor float32
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 16}),  # Ideogram 4 requires multiples of 16
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 16}),
                "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.75, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling"

    def get_sigmas(self, steps, width, height, mu, std):
        # --- Resolution-aware mu shift ---
        resolution_ratio = (width * height) / (512.0 * 512.0)
        mu_eff = mu + 0.5 * np.log(resolution_ratio)

        # --- Logit-normal timestep sampling ---
        q = np.linspace(0, 1, steps + 2)[1:-1]
        z = mu_eff + std * np.sqrt(2) * special.erfcinv(2 * (1 - q))
        t = 1.0 / (1.0 + np.exp(-z))

        # ComfyUI convention: descending, 0 at the end
        sigmas = np.concatenate([t[::-1], [0.0]])
        sigmas_tensor = torch.from_numpy(sigmas.astype(np.float32))

        return (sigmas_tensor,)


NODE_CLASS_MAPPINGS = {
    "Ideogram4SchedulerFixed": Ideogram4SchedulerFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4SchedulerFixed": "Ideogram 4 Scheduler (Fixed)",
}
