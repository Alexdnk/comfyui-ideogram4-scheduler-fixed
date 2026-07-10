import torch

class Ideogram4SchedulerFixed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
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
        # Scale the mean parameter based on target aspect ratio relative to 512x512
        resolution_ratio = (width * height) / (512.0 * 512.0)
        mu_eff = mu + 0.5 * torch.log(torch.tensor(resolution_ratio, dtype=torch.float32)).item()

        # Generate step quantiles, excluding boundary values (0, 1) to prevent infinite logits
        q = torch.linspace(0, 1, steps + 2, dtype=torch.float32)[1:-1]
        
        # Map quantiles to normal distribution using the inverse CDF
        # erfcinv(2 * (1 - q)) is computed via native erfinv(2 * q - 1) to avoid SciPy dependency
        z = mu_eff + std * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * q - 1)
        
        # Sigmoid projection back to the [0, 1] interval
        t = 1.0 / (1.0 + torch.exp(-z))

        # Reordering to descending sigmas and appending the terminal 0.0 step
        sigmas = torch.cat([torch.flip(t, dims=[0]), torch.tensor([0.0])])

        return (sigmas,)


NODE_CLASS_MAPPINGS = {
    "Ideogram4SchedulerFixed": Ideogram4SchedulerFixed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ideogram4SchedulerFixed": "Ideogram 4 Scheduler (Fixed)",
}
