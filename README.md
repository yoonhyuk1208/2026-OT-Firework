# New Year 2026 — OT Fireworks (ModernGL)

A GPU-driven particle piece that starts as **“2025”**, blows up like fireworks, and then **re-forms into “2026”**.
Everything runs in real time using **ModernGL + Transform Feedback** (no compute shader required).

---

## Preview

- Default state: particles settle into **2025**
- Fireworks sequence: **explode for 10s → morph into 2026 for 5s**
- Runs on OpenGL **3.3** (works on most integrated GPUs)

---

## Features

- **GPU particle simulation** with *transform feedback* (ping-pong buffers)
- **Text-to-particles**: sample pixels from rasterized text to generate point clouds
- **Sliced Optimal Transport–style matching (rank-based)** to map “2025” points to “2026” points
- **Procedural motion fields** (curl noise + swirl) for “alive” movement
- **One-shot impulse explosion** + **hold mode** for extended fireworks
- **Trail accumulation** using FBO ping-pong + multiplicative fade
- Adjustable runtime controls (exposure, trails, phase, preview)

---

## Tech Stack

- **Python 3.11+**
- **moderngl** (OpenGL wrapper)
- **moderngl-window** (window/context + input handling)
- **numpy** (point generation, mapping, randomized initialization)
- **Pillow (PIL)** (rasterize text → sample pixels)
- (Optional) **zoneinfo / tzdata** for Korea (Asia/Seoul) time zone support

---

## How It Works

### 1) “2025 / 2026” as Point Clouds
1. Rasterize the string (`"2025"`, `"2026"`) into a grayscale image using PIL.
2. Collect all “ink” pixels (where intensity is above a threshold).
3. Randomly sample `N` pixels and normalize into clip-like coordinates `[-1, 1]`.
4. The result is two point clouds:
   - `home`: positions forming **2025**
   - `target`: positions forming **2026**

This keeps the shape crisp while giving enough randomness so the text doesn’t look like a rigid grid.

---

### 2) Point Matching: Sliced OT (Rank Aggregation)
To morph cleanly, each particle needs a consistent destination in the 2026 cloud.

Instead of solving full optimal transport, this project uses a fast approximation:
- Project both clouds onto multiple random directions `θ`
- For each projection:
  - Sort particles by `dot(x, θ)` (rank)
- Accumulate ranks across projections
- Match particles by overall rank

This produces a stable 1:1 mapping that tends to preserve structure, reduces crossings, and avoids “scrambling” during morph.

---

### 3) GPU Simulation via Transform Feedback
Particles are updated entirely on the GPU with **Transform Feedback** (OpenGL 3.3 compatible):

- Each particle stores:
  - position `pos`
  - velocity `vel`
  - `home` (2025 anchor)
  - `target` (mapped 2026 destination)
  - random seed

**Ping-pong buffers**:
- Buffer A → update shader → write into Buffer B
- Next frame swaps roles.

This avoids CPU-GPU round trips and scales well with large particle counts.

---

### 4) Forces and Motion Model

Particles are driven by a few layered behaviors:

#### (A) Spring-to-target morph
A spring pulls the particle towards:
- `home` when `phase = 0`
- `target` when `phase = 1`

`morph_target = mix(home, target, ease(phase))`

Spring constant increases as phase progresses so the final “2026” locks in sharply.

#### (B) Curl + swirl “life”
A lightweight procedural field (curl-like) plus tangential swirl adds motion.
To prevent the initial “2025” from looking too chaotic:
- noise is reduced when particles are already close to their target (“idle damping”)
- noise is also scaled down during the re-forming phase so the shape stabilizes

#### (C) One-shot explosion impulse
Fireworks are not “constant acceleration”.
Instead:
- on a single frame, each particle receives a velocity kick (radial + slight tangential)
- then it coasts under damping

This produces a much more natural ballistic expansion.

#### (D) Explosion hold
During the explosion window, the spring term is heavily weakened so particles don’t instantly snap back into text.

---

### 5) Trails via FBO Ping-Pong
A second ping-pong system accumulates trails:

1. Render last frame’s trail texture into a new FBO with `fade < 1`
2. Additively draw particles on top
3. Present the composed texture to screen

This yields long exposure-style streaks without storing historical particle positions.

Different fade values are used depending on stage:
- explode: longer trails
- morph: shorter trails for clarity
- final: stable glow

---

## Timeline

- **Explosion duration**: `10s`
- **Morph duration**: `5s`
- Total sequence: **15 seconds**

Stages:
1. `phase = 0` (2025)
2. impulse burst (one frame)
3. hold explosion (10s)
4. morph into 2026 (5s)
5. lock in (phase=1)

---

## Controls

| Key | Action |
|---|---|
| `G` | Start preview sequence (reset + explode→morph) |
| `F` | Toggle preview mode |
| `R` | Regenerate clouds + reset |
| `SPACE` | Manual boom flash |
| `M` | Toggle manual phase override |
| `UP/DOWN` | Adjust manual phase (when enabled) |
| `=` / `-` | Exposure up/down |
| `,` / `.` | Trail keep (fade) up/down |
| `ESC` | Quit |

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install moderngl moderngl-window numpy pillow
