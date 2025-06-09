# depin control tool

optimal control framework for decentralized physical infrastructure networks

## system dynamics

the protocol evolves according to the following state update equations:

### state variables
- **S(t)**: token supply
- **R(t)**: usd reserve 
- **P(t)**: token price
- **C(t)**: network capacity
- **D(t)**: service demand
- **Td(t)**: token demand

### control variables
- **u₁(t)**: mint rate (tokens/timestep)
- **u₂(t)**: burn share ∈ [0,1]

### dynamics

**supply evolution:**
```
S(t+1) = S(t) + u₁(t) - B(t)
```

**reserve evolution:**
```
R(t+1) = R(t) + Rev(t) + B(t) × P(t) × (1 - u₂(t))
```

**price formation:**
```
P(t+1) = (1-λ) × V₀ × [Td(t)/S(t+1)] × [1 + β × U(t+1)] × [1 + γ × M(t)]
         + λ × P(t) × [1 + M(t)]
```

**capacity evolution:**
```
C(t+1) = C(t) × [1 + κ × max(0, π(t) - 1) - δ]
where π(t) = [u₁(t) × P(t) / C(t)] / [c₀ × (1 + α × C(t))]
```

**service demand with network effects:**
```
D(t+1) = D(t) × [1 + ε₁ × (C(t)/C(0) - 1) + ε₂ × (P(t)/P(0) - 1) + σ_d × ω_d(t)]
```

**burned tokens:**
```
B(t) = min(D(t), C(t)) × p_s × u₂(t) / P(t)
```

**utilization:**
```
U(t) = min(1, D(t)/C(t))
```

where ω_d(t), M(t) are gaussian noise terms.

## cost function

quadratic penalty optimization:
```
J = Σₜ γᵗ [w₁(ġ_p(t) - ġ_p*)² + w₂ max(0, U* - U(t))² + w₃(ġ_c(t) - ġ_c*)²]
```

where:
- ġ_p(t) = (P(t) - P(t-1))/P(t-1): price growth rate
- ġ_c(t) = (C(t) - C(t-1))/C(t-1): capacity growth rate
- (w₁, w₂, w₃): user-defined objective weights
- (ġ_p*, U*, ġ_c*): target values

## parameter values

**network effects:**
- ε₁ = 0.02 (service network effect)
- ε₂ = 0.015 (service value effect)

**price formation:**
- V₀ = 1.0 (base utility value)
- β = 0.2 (utilization price boost)
- λ = 0.3 (speculation weight)
- γ = 0.5 (market sensitivity)

**capacity dynamics:**
- κ = 0.008 (response speed)
- δ = 0.005 (depreciation rate) 
- c₀ = 0.012 (cost ratio)
- α = 0.0001 (cost scale factor)

**service economics:**
- p_s = 1.0 (service price)

**volatility:**
- σ_d = 0.05 (service demand volatility)

**cost function defaults:**
- γ = 0.95 (discount factor)
- (ġ_p*, U*, ġ_c*) = (0.018, 0.8, 0.025)

## usage

### streamlit interface
```bash
uv run python run_app.py
```

### programmatic
```python
from optimal_control import OptimalController
from cost_function import CostWeights, CostTargets

weights = CostWeights(price_growth=1.5, utilization_min=2.0, capacity_growth=1.0)
targets = CostTargets(price_growth=0.018, utilization_min=0.8, capacity_growth=0.025)

controller = OptimalController(weights, targets)
controller.train_optimal_policy()

optimal_policy = controller.get_optimal_policy(initial_state)
```

## implementation

- **system_dynamics.py**: state evolution equations
- **cost_function.py**: quadratic penalty optimization
- **value_function.py**: fitted value iteration
- **optimal_control.py**: policy optimization
- **streamlit_app.py**: interactive interface
- **depin_utils.py**: simulation utilities
