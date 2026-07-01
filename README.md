# ComfyUI Ideogram4 Resolution Scheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A custom node for ComfyUI that implements the **official resolution-aware mu shift** for Ideogram 4 noise schedules.

## Features

- **Resolution‑aware** – automatically adjusts `mu` according to image size:  
  `μ_eff = μ + 0.5 * ln( (width * height) / (512 * 512) )`
- **Drop‑in replacement** – same interface as the standard `Ideogram4Scheduler` node (inputs: `steps`, `width`, `height`, `mu`, `std`; output: `SIGMAS`)

## Installation

1. Navigate to your ComfyUI `custom_nodes/` directory.
2. Clone this repository:
   ```bash
   git clone https://github.com/Alexdnk/comfyui-ideogram4-scheduler-fixed.git

3. Restart ComfyUI.

## Usage

   Add the node `Ideogram 4 Scheduler (Fixed)` to your workflow.


## Technical Details

   The scheduler generates logit‑normal timesteps:

   Uniform quantiles q from 0 to 1 (excluding endpoints)

   Transform: z = μ_eff + std · √2 · erfc⁻¹(2(1−q))

   Sigmoid: t = 1 / (1 + exp(−z))

   Return reversed order (high noise → low noise) with 0 at the end (ComfyUI convention)

   This matches the official Ideogram 4 implementation in Diffusers.
