import torch

class Ideogram4SchedulerFixed:
    """
    A ComfyUI custom node implementing the resolution-aware logit-normal
    timestep scheduler for Ideogram 4.

    Background:
    Flow-matching models trained with logit-normal schedules require adjusting 
    the schedule mean (mu) based on the target image resolution. 
    Larger images require more denoising attention near the initial high-noise 
    states (t=1). Without this shift, generating at resolutions higher than 
    the base training size (512x512) often results in structural incoherence 
    or severe noise graininess.

    Mathematical Formulation:
    
    1. Effective Mean (mu_eff) Shift:
       mu_eff = mu + 0.5 * ln( (width * height) / (512 * 512) )
       where 512x512 is the base training resolution of the model.

    2. Inverse Logit-Normal CDF Sampling (Quantile Function):
       To sample 'steps' timesteps from the logit-normal distribution:
       q = linspace(0, 1, steps + 2)[1:-1]  # Exclude boundary points (0, 1) to avoid infinite logits
       z = mu_eff + std * sqrt(2) * erfcinv(2 * (1 - q))
       
       In pure PyTorch, erfcinv(y) is computed via erfinv(1 - y):
       erfcinv(2 * (1 - q)) = erfinv(1 - 2 * (1 - q)) = erfinv(2 * q - 1)
       z = mu_eff + std * sqrt(2) * erfinv(2 * q - 1)

    3. Sigmoid Mapping:
       t = 1 / (1 + exp(-z))

    4. ComfyUI Sigmas Convention:
       ComfyUI expects the sequence of noise levels (sigmas) to be in 
       descending order (from maximum noise to 0) and terminate with an 
       explicit 0.0 value.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                # Ideogram 4 requires dimensions to be multiples of 16
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.001}),
                "std": ("FLOAT", {"default": 1.75, "min": 0.01, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/custom_sampling"

    def get_sigmas(self, steps, width, height, mu, std):
        # Calculate ratio relative to the base training resolution of 512x512
        resolution_ratio = (width * height) / (512.0 * 512.0)
        
        # Apply the resolution-aware shift to the schedule mean
        mu_eff = mu + 0.5 * torch.log(torch.tensor(resolution_ratio, dtype=torch.float32)).item()

        # Generate quantiles excluding boundary values 0 and 1 to prevent logit overflow
        q = torch.linspace(0, 1, steps + 2, dtype=torch.float32)[1:-1]
        
        # Map quantiles to normal distribution using PyTorch's inverse CDF.
        # Since erfcinv(2 * (1 - q)) is equivalent to erfinv(2 * q - 1), we can avoid scipy dependencies.
        z = mu_eff + std * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * q - 1)
        
        # Apply sigmoid to project values back to timesteps interval [0, 1]
        t = 1.0 / (1.0 + torch.exp(-z))

        # Reverse the order to align with ComfyUI's descending convention, and append terminal 0.0
        sigmas = torch.cat([torch.flip(t, dims=[0]), torch.tensor([0.0])])

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "Ideogram4SchedulerFixed": Ideogram4SchedulerFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4SchedulerFixed": "Ideogram 4 Scheduler (Fixed)",
}
