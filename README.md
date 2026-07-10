# ComfyUI Ideogram 4 Scheduler

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



---
# ComfyUI Ideogram 4 Scheduler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Легковесная кастомная нода для ComfyUI, реализующая оптимизированный планировщик временных шагов (logit-normal scheduler) для модели Ideogram 4 на чистом PyTorch.

Хотя встроенный планировщик ComfyUI корректно масштабирует разрешение относительно базового значения $512 \times 512$, данная нода использует более чистый с точки зрения математики метод построения сетки шагов. Это повышает стабильность генерации, снижает уровень шума (зернистости) и улучшает анатомическую связность на нестандартных и высоких разрешениях.

---

## В чем отличие от встроенного планировщика?

Ключевая разница между стандартной нодой ComfyUI и этой версией заключается в **способе построения сетки квантилей** и обработке граничных значений лог-нормального распределения.

### 1. Обработка крайних точек (Без принудительного зажима)
* **Встроенный планировщик:** 
  Генерирует квантили $u \in [0, 1]$, *включая* крайние точки $0.0$ и $1.0$. Поскольку эти точки при переводе в логиты через обратную функцию распределения ($\text{ndtri}$) уходят в $-\infty$ и $+\infty$, оригинальный код вынужден использовать искусственные жесткие ограничения отношения сигнал/шум (`_LOGSNR_MIN = -15.0` и `_LOGSNR_MAX = 18.0`) для зажима значений.
* **Этот оптимизированный планировщик:** 
  Генерирует квантили $q \in (0, 1)$ путем отсечения крайних точек через срез `[1:-1]`. Благодаря естественному исключению границ логиты никогда не достигают бесконечности. Кривая сэмплирования выстраивается плавно и не требует принудительного зажима (clamping).

### 2. Пропуск нестабильного начального шума (Первый шаг)
* **Встроенный планировщик:** Всегда начинает процесс генерации с зажатого максимального шума ($\sigma \approx 0.9999$).
* **Этот оптимизированный планировщик:** Из-за исключения крайней точки максимальный начальный квантиль при 20 шагах составляет $20/21 \approx 0.952$. Это масштабирует начальную сигму до чуть более низкого диапазона ($\approx 0.94 - 0.97$ в зависимости от разрешения).

*Пропуск самого первого этапа абсолютного, хаотичного шума (где модель еще не строит геометрию объектов, но склонна вносить паразитные цветовые оттенки) позволяет обойти нестабильную фазу генерации, заметно снижая общую зернистость кадра.*

### 3. Плотность шагов в «рабочей зоне»
Поскольку крайние точки исключены из вычислений, все заданные шаги сэмплирования плотнее распределяются внутри наиболее активного диапазона шума $[0.05, 0.95]$. Именно в этом диапазоне модель принимает ключевые решения относительно композиции, расположения текста и анатомии. Повышенная плотность шагов в этой зоне обеспечивает лучшую прорисовку деталей и букв.

---

## Математическое описание

Вычисления производятся нативно на GPU/CPU средствами PyTorch:

$$\mu_{\text{eff}} = \mu + 0.5 \ln\left(\frac{\text{width} \times \text{height}}{512^2}\right)$$

$$q_i = \frac{i}{\text{steps} + 1} \quad \text{для } i \in [1, \text{steps}]$$

$$z_i = \mu_{\text{eff}} + \sigma \cdot \sqrt{2} \cdot \text{erfinv}(2q_i - 1)$$

$$t_i = \frac{1}{1 + e^{-z_i}}$$

Полученный тензор разворачивается в обратном порядке и дополняется финальным шагом $0.0$ для совместимости со стандартами ComfyUI.

---

## Особенности

- **Чистый PyTorch:** Полный отказ от использования NumPy и SciPy. Работа идет напрямую с тензорами, что исключает накладные расходы на копирование данных между CPU и GPU.
- **Повышение качества на высоких разрешениях:** Помогает исправить искажения анатомии (лишние конечности, дублирование лиц) и избыточную зернистость на неквадратных кадрах.
- **Полная совместимость:** Нода подключается аналогично стандартным планировщикам и выдает стандартный тип данных `SIGMAS`.

---

## Установка

1. Перейдите в папку вашего установленного ComfyUI, в директорию `custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes/
   ```
2. Склонируйте этот репозиторий:
   ```bash
   git clone https://github.com/Alexdnk/comfyui-ideogram4-scheduler-fixed.git
   ```
3. Перезапустите ComfyUI.

---

## Использование

Добавьте ноду **Ideogram 4 Scheduler (Fixed)** в ваш рабочий процесс (workflow) вместо стандартного планировщика.

### Параметры на входе:
* **`steps`**: Количество шагов сэмплирования.
* **`width` / `height`**: Целевое разрешение изображения (Ideogram 4 требует кратности 16).
* **`mu`**: Базовое среднее значение распределения временных шагов (по умолчанию: `0.0`).
* **`std`**: Стандартное отклонение распределения временных шагов (по умолчанию: `1.75`).
