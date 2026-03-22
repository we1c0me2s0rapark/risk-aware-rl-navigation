# 🚗 Risk-Aware RL Navigation in CARLA

> A risk-sensitive autonomous driving agent built with distributional reinforcement learning, CVaR optimisation, and multimodal perception.

[![Python](https://img.shields.io/badge/Python-3.10.20-blue?style=flat-square)](https://python.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.15-orange?style=flat-square)](https://carla.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat-square)]()

---

## Project Status

Track live progress, milestones, and the backlog on the **[GitHub Project Board](https://github.com/users/we1c0me2s0rapark/projects/1/views/4)**

---

## Overview

This project develops a **risk-aware autonomous driving agent** in the [CARLA](https://carla.org) simulator. Rather than optimising purely for reward, the agent explicitly models and minimises **tail risk** - catastrophic, low-probability events like collisions.

### Key Components

| Component | Description |
|---|---|
| **Algorithm** | Soft Actor-Critic (SAC), modified for distributional & risk-aware learning |
| **Critic** | Distributional Q-network - outputs a return *distribution* Z(s, a), not a scalar |
| **Risk Objective** | CVaR optimisation - maximises expected return in the worst α% of outcomes |
| **Memory** | LSTM for temporal modelling and partial observability |
| **Perception** | Multimodal - RGB camera + LiDAR (BEV) |

---

## Architecture

```
       ┌──────────┐      ┌───────────┐
Camera │          │      │           │
LiDAR  │   CNNs   │────▶│  Fusion   │
       └──────────┘      └─────┬─────┘
                               │
                               ▼
                              LSTM
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
                Actor                    Critic
              π(a | s)           Distributional Z(s, a)
                                    (CVaR optimised)
```

- **Perception:** Camera + LiDAR → CNN encoders → fused latent representation
- **Memory:** LSTM over the fused encoding (sequence length ≈ 10–20 steps)
- **Policy:** Actor head π(a | s)
- **Value:** Distributional critic Z(s, a) with quantile regression (32–64 quantiles)
- **Objective:** CVaR_α(Z), typically α = 0.1

---

## Development Phases

The project is built incrementally. **Do not introduce risk modelling before achieving baseline stability.**

### Phase 1 - Baseline RL Environment & Agent

**Goal:** Establish a stable end-to-end RL pipeline.

- Gym-compatible `CarlaEnv` with synchronous simulation mode
- RGB camera (84×84)
- Basic reward: `+speed`, `-collision (terminal)`
- SAC agent with CNN encoder + MLP actor/critic heads

**Deliverable:** Agent demonstrates stable forward driving.

---

### Phase 2 - Multimodal Perception & Stability

**Goal:** Improve state representation and driving robustness.

- LiDAR sensor (`ray_cast`) → Bird's Eye View (BEV) grid (64×64)
- Sensor fusion: `Camera CNN + LiDAR CNN → concatenation → latent`
- Refined reward shaping:
  - `+speed`
  - `-collision`
  - `-lane deviation` *(optional)*
  - `-steering instability` (|Δsteer|)

**Deliverable:** Smoother driving with improved obstacle awareness.

---

### Phase 3 - Distributional Critic & CVaR Optimisation

**Goal:** Model and optimise tail risk - catastrophic, low-probability events.

**Distributional Critic**

Replace the scalar Q-network with a distribution over returns:
```
Q(s, a) → scalar          # Phase 1/2
Z(s, a) → distribution   # Phase 3+
```
Implemented via quantile regression (32–64 quantiles).

**CVaR Optimisation**

```
CVaRα(Z) = E[Z | Z ≤ quantile_α(Z)]
```

Training maximises CVaR_α(Z) (typically α = 0.1), penalising worst-case outcomes.

**Risk Signal Engineering**

| Signal | Value |
|---|---|
| Collision penalty (hard) | −10 to −100 |
| Proximity penalty (soft) | `∝ exp(−distance_to_obstacle)` |

**Deliverable:** Conservative, risk-aware behaviour with reduced collision frequency.

---

### Phase 4 - Temporal Modelling via LSTM

**Goal:** Handle partial observability and temporal risk accumulation.

**Extended Architecture**
```
(Camera CNN + LiDAR CNN) → Fusion → LSTM → Actor & Critic
```

Optionally include previous action: `[o_t, a_{t-1}]`

**Benefits**
- Infers previously observed hazards
- Handles occluded or partially visible objects
- Captures speed build-up and delayed risk
- Enables contextual decisions (e.g., anticipating turns)

**Training Notes**
- Truncated BPTT with sequence length ≈ 10–20
- Hidden state reset at episode boundaries
- Sequences stored in replay buffer

**Deliverable:** Improved anticipation and reduced delayed/unexpected collisions.

---

## Evaluation Metrics

### Performance
- Average episodic return
- Distance travelled per episode
- Route completion rate

### Safety & Risk
- Collision rate
- Near-miss frequency (LiDAR proximity-based)
- CVaR of return distribution

---

## Known Implementation Risks

- **Sensor desynchronisation** between camera and LiDAR streams
- **Training instability** from combining distributional and recurrent learning
- **Sparse catastrophic events** slowing convergence of the risk signal
- **CARLA performance constraints** limiting simulation throughput

---

## Getting Started

> Setup instructions coming soon. The project targets CARLA 0.9.x and Python 3.8+.

```bash
# Clone the repository
git clone https://github.com/we1c0me2s0rapark/risk-aware-rl-navigation.git
cd risk-aware-rl-navigation

# Install dependencies
pip install -r requirements.txt

# Launch CARLA server (separately), then run
python train.py --phase 1
```

---

## Recommended Development Order

```
Phase 1  →  Phase 2  →   Phase 3  →   Phase 4
Baseline    Multimodal   Risk-Aware   Temporal
  SAC       Perception   CVaR/Dist    LSTM
```

---

## Final Goal

A risk-aware autonomous driving agent that:

- ✅ Minimises catastrophic failures through explicit tail-risk optimisation
- ✅ Leverages multimodal perception (camera + LiDAR)
- ✅ Reasons over time via LSTM memory
- ✅ Explicitly optimises for safety under uncertainty using CVaR

---

*Built with CARLA, PyTorch, and a healthy respect for rare but catastrophic events.*
