# Circuit Parameter Sampling Documentation and Reference

---

## Table of Contents

- [Environment Parameters](#environment-parameters) (7 fixed context parameters)
- [Sizing Parameters](#sizing-parameters) (design variables)
- [Sobol Dimension Mapping](#sobol-dimension-mapping)
- [E-Series Standards](#e-series-standards)
- [Sampling Reference](#sampling-reference)

---

## Environment Parameters

Environment parameters define operating conditions for each design point. They are treated as **fixed context** during optimization (TuRBO-based algorithms) rather than tunable variables.

### 1. is_hp — Library/Speed Grade

| Property | Value |
|----------|-------|
| **Type** | Categorical |
| **Choices** | 0 (LSTP), 1 (HP) |
| **Sampling** | `idx = floor(u × 2)` where u ∈ [0,1] |
| **Inverse Mapping** | Bin center: 0.25 (LSTP), 0.75 (HP) |
| **Context** | Library selection affects power, speed, leakage |

---

### 2. process_corner — PVT Corner (n_state, p_state)

| Property | Value |
|----------|-------|
| **Type** | Categorical (tuple pairs) |
| **Choices** | 5 corners: FF, TT, SS, FS, SF |
| **Sampling** | `idx = floor(u × 5)` where u ∈ [0,1] |
| **Inverse Mapping** | Bin center: 0.1, 0.3, 0.5, 0.7, 0.9 |
| **Context** | Process variation → I_dsat, V_t shifts |

**Corner Definitions:**

| Corner | n_state | p_state | Meaning |
|--------|---------|---------|---------|
| FF | 1 | 1 | Fast NMOS, Fast PMOS |
| TT | 0 | 0 | Typical NMOS, Typical PMOS |
| SS | -1 | -1 | Slow NMOS, Slow PMOS |
| FS | 1 | -1 | Fast NMOS, Slow PMOS (mixed) |
| SF | -1 | 1 | Slow NMOS, Fast PMOS (mixed) |

---

### 3. fet_num — Technology Node

| Property | Value |
|----------|-------|
| **Type** | Categorical |
| **Choices** | 7, 10, 14, 16, 20 (nm) |
| **Sampling** | `idx = floor(u × 5)`, clamped to [0, 4] |
| **Inverse Mapping** | `idx / 4` |
| **Context** | Sets L bounds, nominal V_DD, baseline performance |

**Per-Node Configuration:**

| Node | lmin | lmax | vdd_nom |
|------|------|------|---------|
| 7nm  | 10nm | 30nm | 0.70V   |
| 10nm | 10nm | 30nm | 0.75V   |
| 14nm | 10nm | 30nm | 0.80V   |
| 16nm | 10nm | 30nm | 0.80V   |
| 20nm | 10nm | 24nm | 0.90V   |

---

### 4. vdd — Supply Voltage

| Property | Value |
|----------|-------|
| **Type** | Continuous |
| **Grid Step** | 10mV (0.01V) |
| **Range Formula** | `[0.9 × vdd_nom, 1.1 × vdd_nom]` |
| **Sampling** | Linear grid → `idx = floor(u × len(grid))` |
| **Inverse Mapping** | Nearest grid point normalized: `u = idx / len(grid)` |
| **Context** | Primary knob: power, speed, noise margin |

**VDD Ranges by Node:**

| Node | Min | Max | # Grid Points |
|------|-----|-----|---------------|
| 7nm  | 0.63V | 0.77V | ~15 |
| 10nm | 0.675V | 0.825V | ~16 |
| 14nm | 0.72V | 0.88V | ~17 |
| 16nm | 0.72V | 0.88V | ~17 |
| 20nm | 0.81V | 0.99V | ~19 |

**Sampling Algorithm:**
```
grid = arange( round(lb/0.01)×0.01, round(ub/0.01)×0.01 + δ, 0.01 )
value = grid[min(floor(u × len(grid)), len(grid)-1)]
```

---

### 5. vcm — Input Common-Mode Voltage

| Property | Value |
|----------|-------|
| **Type** | Continuous, topology-aware |
| **Grid Step** | 10mV (0.01V) |
| **Sampling** | Depends on topology-specified fractions of sampled VDD |
| **Inverse Mapping** | Reads vdd from row; recalculates bounds dynamically |
| **Context** | Input biasing → linearity, gain |

**Topology-Aware Fractions:**

| Topology | Lower Frac | Upper Frac | Example (VDD=0.8V) |
|----------|-----------|-----------|-------------------|
| NMOS | 0.65 | 0.85 | 0.52V – 0.68V |
| PMOS | 0.15 | 0.35 | 0.12V – 0.28V |
| Default | 0.10 | 0.90 | 0.08V – 0.72V |

---

### 6. tempc — Temperature

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Range** | –40°C to 125°C |
| **Span** | 165°C |
| **Sampling** | `t = round(-40 + u × 165)`, clamped |
| **Inverse Mapping** | `u = (val + 40) / 165` |
| **Context** | Affects leakage, speed, V_t drift |

---

### 7. cload_val — Load Capacitance

| Property | Value |
|----------|-------|
| **Type** | Continuous (E12 series) |
| **Range** | 10 fF – 5 pF |
| **Series** | E12 (12 bases/decade) |
| **# Values** | ~47 grid points |
| **Sampling** | E-series grid → `idx = floor(u × len(grid))` |
| **Inverse Mapping** | `nearest_grid_u(grid, val)` |
| **Context** | Settling time, slew rate, bandwidth, power |

---

## Sizing Parameters

Sizing parameters are circuit design variables optimized by TuRBO and similar algorithms. All bounds are defined in `TECH_CONSTANTS['shared']['circuit_params']`.

### GROUP A: Channel Length — nA[1–8]

**MOSFETs Channel Lengths**

| Property | Value |
|----------|-------|
| **Type** | Continuous (1nm quantization) |
| **Range** | `[lmin, lmax]` per technology (see node table) |
| **Grid Step** | 1nm (1e–9 m) |
| **Sampling** | `grid = arange(lmin, lmax+δ, 1e-9)` → pick by u |
| **Inverse Mapping** | `argmin_idx(\|grid - val\|) / (len(grid)-1)` |
| **Purpose** | I_ds, g_m, output impedance |

**Example: 7nm node**
- Grid: [10nm, 11nm, 12nm, ..., 30nm]
- Count: 21 values

---

### GROUP B: Fin Count — nB[1–8]

**FinFET Device Width**

| Property | Value |
|----------|-------|
| **Type** | Integer |
| **Range** | 4 – 128 fins |
| **Sampling** | `val = round(4 + u × 124)`, clamp to [4, 128] |
| **Inverse Mapping** | `(val - 4) / 124` |
| **Purpose** | I_ds ∝ N_fin |

---

### GROUP C: Bias Voltages — vbiasn[0–2], vbiasp[0–2]

**NMOS Tail Current Bias**

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|------------|
| vbiasn0 | 0.45 × vdd_nom | 0.70 × vdd_nom |
| vbiasn1 | 0.45 × vdd_nom | 0.70 × vdd_nom |
| vbiasn2 | 0.65 × vdd_nom | 0.85 × vdd_nom |

**PMOS Tail Current Bias**

| Parameter | Lower Bound | Upper Bound |
|-----------|-------------|------------|
| vbiasp0 | 0.40 × vdd_nom | 0.85 × vdd_nom |
| vbiasp1 | 0.40 × vdd_nom | 0.85 × vdd_nom |
| vbiasp2 | 0.15 × vdd_nom | 0.50 × vdd_nom |

**Sampling Details:**

| Property | Details |
|----------|---------|
| **Type** | Continuous (10mV grid) |
| **Expression Eval** | Bounds are expressions in vdd_nom; evaluated: `eval("0.45 * vdd_nominal", vdd_nom)` |
| **Grid Step** | 10mV; normalized: `lb = round(lb/0.01) × 0.01` |
| **Sampling** | Linear grid → pick by u |
| **Inverse Mapping** | Nearest grid point → `nearest_grid_u(grid, val)` |

---

### GROUP D: Internal Capacitance — nC[1–4]

**Load Capacitances**

| Property | Value |
|----------|-------|
| **Type** | Continuous (E12 series) |
| **Range** | 100 fF – 5 pF |
| **Series** | E12 (12 bases/decade) |
| **# Values** | ~47 points |
| **Sampling** | E-series grid → pick by u |
| **Inverse Mapping** | `nearest_grid_u(grid, val)` |
| **Purpose** | Frequency compensation, bandwidth, phase margin |

---

### GROUP E: Internal Resistance — nR[1–4]

**Series Stabilization Resistances**

| Property | Value |
|----------|-------|
| **Type** | Continuous (E24 series) |
| **Range** | 500 Ω – 500 kΩ |
| **Series** | E24 (24 bases/decade) |
| **# Values** | ~70 points |
| **Sampling** | E-series grid → pick by u |
| **Inverse Mapping** | `nearest_grid_u(grid, val)` |
| **Purpose** | Stabilization, compensation, impedance match |

---

## Sobol Dimension Mapping

The Sobol sequence generates uniform samples **u ∈ [0,1]^d** where **d = 7 + # sizing params**.

### Standard Configuration: 7 Fixed Environment Parameters

| Dimension | Parameter | Role |
|-----------|-----------|------|
| 0 | `is_hp` | Library selection |
| 1 | `process_corner` | PVT corner |
| 2 | `fet_num` | Technology node |
| 3 | `vdd` | Supply voltage |
| 4 | `vcm` | Input common-mode |
| 5 | `tempc` | Temperature |
| 6 | `cload_val` | Load capacitance |
| **7+** | **Sizing variables** | **Design parameters** |

### Example: 5 Sizing Variables

For `sizing_params=['nA1', 'nB1', 'vbiasn0', 'nC1', 'nR1']`:

```
Total Dimension: 12

u_samples shape: (n_samples, 12)

Columns:
  [0–6]   Fixed parameters (is_hp, process_corner, ..., cload_val)
  [7–11]  Sizing (nA1, nB1, vbiasn0, nC1, nR1)
```

---

## E-Series Standards

### E12 Series (12 values per decade)

Used for: Load capacitances, internal capacitances

```
[1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]
```

**Example in range [100 fF, 5 pF]:**
```
100fF, 120fF, 150fF, 180fF, 220fF, ..., 3.3pF, 3.9pF, 4.7pF, 5.6pF
(Note: truncated at 5pF upper bound)
```

### E24 Series (24 values per decade)

Used for: Internal resistances

```
[1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1]
```

**Example in range [500 Ω, 500 kΩ]:**
```
500Ω, 550Ω, 620Ω, 680Ω, ..., 68kΩ, 75kΩ, 82kΩ, 91kΩ
(truncated at 500kΩ)
```

### Grid Generation

```python
def e_series_grid(lb, ub, series='E24'):
    """Generate E-series values between lb and ub."""
    bases = E24 or E12
    values = []
    
    min_exp = floor(log₁₀(lb))
    max_exp = ceil(log₁₀(ub))
    
    for exp in range(min_exp - 1, max_exp + 2):
        for base in bases:
            val = base × 10^exp
            if lb ≤ val ≤ ub:
                values.append(val)
    
    return sorted(set(values))
```

---

## Sampling Reference

### Forward Sampling: [0,1]^d → Physical Values

**Generic Algorithm for Each Parameter:**

```
Input: u ∈ [0, 1]

IF parameter is CATEGORICAL:
    idx = floor(u × num_choices)
    value = choices[idx]

ELSE IF parameter is INTEGER:
    value = round(lb + u × (ub - lb))
    clamp to [lb, ub]

ELSE IF parameter is CONTINUOUS (linear grid):
    grid = arange(lb, ub + ε, step)
    idx = floor(u × len(grid))
    clamp idx to [0, len(grid)-1]
    value = grid[idx]

ELSE IF parameter is CONTINUOUS (E-series):
    grid = e_series_grid(lb, ub, series)
    idx = floor(u × len(grid))
    clamp idx to [0, len(grid)-1]
    value = grid[idx]
```

### Inverse Mapping: Physical Values → [0,1]^d

Used by TuRBO's "Sight" initialization mode to map existing data back to Sobol space.

```
Input: physical value

IF parameter is CATEGORICAL:
    idx = choices.index(value)
    u = (idx + 0.5) / num_choices  # bin center

ELSE IF parameter is INTEGER:
    u = (value - lb) / (ub - lb)
    clamp to [0, 1]

ELSE IF parameter is CONTINUOUS (grid):
    grid = generate_grid(...)
    idx = argmin(|grid - value|)
    u = idx / max(1, len(grid) - 1)

ELSE IF parameter is CONTINUOUS (E-series):
    grid = e_series_grid(lb, ub, series)
    idx = argmin(|grid - value|)
    u = idx / max(1, len(grid) - 1)
```

---

## Implementation Locations

| Component | File | Lines |
|-----------|------|-------|
| TECH_CONSTANTS | `algorithms/sobol/generator.py` | 20–57 |
| Forward sampling | `SobolSizingGenerator.generate()` | 318–615 |
| Inverse mapping | `SobolSizingGenerator.inverse_map()` | 625–810 |
| E-series grid | `e_series_grid()` | 80–101 |
| Linear grid | `discrete_linear_grid()` | 104–119 |
| Grid picker | `pick_from_grid_by_u()` | 122–145 |
| Grid inverter | `nearest_grid_u()` | 148–170 |
| VCM topology | `_vcm_bounds_for_topology()` | 61–76 |

---

## Design Rationale

### 10mV Voltage Step (VDD, VCM, Vbias)

✓ **Optimal Sobol balance**: 10–20 grid points per dimension  
✓ **Engineering precision**: ~1–2% of nominal voltage  
✓ **Numerical stability**: Avoids inverse mapping singularities  
✓ **Industry standard**: Matches PDK simulation conventions  

### 1nm Channel Length Step

✓ **Process realism**: Matches advanced node (7–20nm) PDK resolution  
✓ **Design relevance**: 1nm ≈ 3–5% of lmin  
✓ **No sub-nm effects**: Physical variation negligible below 1nm  

### E-Series for Passive Components

✓ **Physical reality**: Manufacturing tolerances in E-series  
✓ **Design intent**: Practical, realizable circuits  
✓ **Industry norm**: Standard for analog, RF, power electronics

---

## Revisions

| Date | Author | Notes |
|------|--------|-------|
| 2026-03-31 | natelgrw | Comprehensive markdown documentation with tables |
