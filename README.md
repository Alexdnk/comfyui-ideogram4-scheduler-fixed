# comfyui-ideogram4-scheduler-fixed
Resolution-aware mu shift scheduler for Ideogram 4 models. Automatically adjusts noise schedule based on image resolution (official formula: μ_eff = μ + 0.5·ln(pixels / 262144)). Drop‑in replacement for the default ComfyUI Ideogram4Scheduler – just use it with the same parameters.
