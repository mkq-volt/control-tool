# depin control tool

optimal control framework for depin protocol optimization

## overview

this tool enables protocol designers to simulate, evaluate, and optimize controller designs using formal optimal control methods. it provides a complete framework for analyzing token emission policies and incentive structures in decentralized infrastructure networks.

## 🎮 streamlit interface

interactive web interface for designing and optimizing depin controllers.

### quick start
```bash
# install dependencies
uv add streamlit plotly pandas

# run the interface
uv run python run_app.py
```

then open http://localhost:8501 in your browser.

### workflow
1. **controller selection**: choose between inflation policy or service pricing optimization
2. **system configuration**: input initial protocol state with auto-validation and calculated metrics
3. **policy design**: configure your proposed control policy
4. **objectives**: set cost function targets and weights with mathematical explanations
5. **optimization**: train value function and compare optimal vs proposed policies

### features
- guided wizard interface with progress tracking
- real-time input validation and stability suggestions
- simplified inputs with auto-calculated token demand (50% of supply)
- interactive policy configuration with live examples
- comprehensive mathematical explanations of cost functions
- monte carlo simulation results with confidence intervals
- trajectory visualizations for all state variables

## current implementation

### ✅ network-aware system dynamics
- **system_dynamics.py**: protocol mechanics with coupled service demand and capacity growth
- **depin_utils.py**: utilities for simulation, validation, and stability analysis
- network effects: service demand grows with capacity and token value
- balanced capacity economics with natural depreciation
- significantly reduced utilization penalty dominance (from 83% to ~53%)

### ✅ comprehensive cost function 
- **cost_function.py**: user-configurable optimization objectives
- three core objectives: price growth, utilization minimum, capacity growth
- quadratic penalty functions with scale normalization
- robust multi-run evaluation with confidence intervals and significance testing

### ✅ robust value function approximation
- **value_function.py**: fitted value iteration with comprehensive state coverage
- neural network architecture: 6 → 96 → 64 → 32 → 1 with tanh activation
- utilization-aware state sampling focusing on problematic ranges (0.5-0.9)
- multi-method bellman optimization: l-bfgs-b + differential evolution + grid search fallback
- built-in validation against random alternative policies

### ✅ optimal control
- **optimal_control.py**: complete workflow for policy training and comparison
- trains optimal policies using fitted value iteration with robust optimization
- compares user policies against optimal with statistical significance testing
- comprehensive evaluation with cost breakdowns and confidence intervals

### ✅ streamlit interface
- **streamlit_app.py**: interactive web interface for controller design
- guided 5-step workflow from controller selection to optimization results
- real-time validation with stability suggestions
- mathematical explanations and interactive visualizations
- monte carlo comparison with confidence interval plots

## usage

### web interface (recommended)
```bash
uv run python run_app.py
```

### programmatic usage

#### basic cost evaluation
```python
from cost_function import CostFunction, CostWeights, CostTargets

weights = CostWeights(price_growth=1.5, utilization_min=2.0, capacity_growth=1.0)
targets = CostTargets(price_growth=0.015, utilization_min=0.75, capacity_growth=0.01)
cost_fn = CostFunction(weights, targets)

evaluation = cost_fn.evaluate_policy_robust(initial_state, policy, timesteps=50, num_runs=10)
```

#### optimal control workflow
```python
from optimal_control import OptimalController, TrainingConfig

# define objectives
controller = OptimalController(weights, targets)

# train optimal policy with comprehensive coverage
training_config = TrainingConfig(
    num_trajectories=200, 
    fvi_iterations=6,
    utilization_aware_sampling=True,
    state_space_coverage='comprehensive',
    use_global_optimization=True
)
controller.train_optimal_policy(training_config)

# compare policies with robust evaluation
user_policies = {
    'conservative': {'mint_rate': 10_000, 'burn_share': 0.8},
    'aggressive': {'mint_rate': 100_000, 'burn_share': 0.2}
}
comparison = controller.compare_policies(initial_state, user_policies, num_runs=20)
controller.print_comparison_summary(comparison)
```

## architecture

focused on clean separation of concerns with comprehensive coverage:
- **dynamics**: network-aware protocol mechanics with coupled demand-capacity growth
- **cost**: user-defined objectives with robust multi-run evaluation
- **control**: fitted value iteration with comprehensive state coverage and multi-method optimization
- **evaluation**: statistical comparison with confidence intervals and significance testing
- **interface**: guided streamlit workflow with educational explanations

## performance

- **training time**: 10-15 seconds for full optimal policy training (150 trajectories, 4 iterations)
- **evaluation**: robust multi-run policy comparison with statistical significance
- **scalability**: works across 100x protocol size differences with utilization-aware sampling
- **accuracy**: utilization penalty dominance reduced from 83% to ~53%
- **coverage**: comprehensive state space sampling with focus on problematic utilization ranges
- **interface**: responsive web ui with real-time validation and interactive visualizations

## key improvements

- **reduced utilization penalty dominance**: network effects couple service demand to capacity growth
- **comprehensive state coverage**: utilization-aware sampling focuses on critical ranges (0.5-0.9)
- **robust optimization**: multi-method bellman solving (l-bfgs-b + differential evolution + grid search)
- **statistical rigor**: confidence intervals and significance testing for policy comparisons
- **validation framework**: systematic verification of learned optimal policies
- **user interface**: guided wizard with educational explanations and interactive visualizations

## status

production-ready implementation with all prd phases complete and significant enhancements:
- ✅ network-aware system dynamics with balanced utilization
- ✅ comprehensive controller design with robust optimization
- ✅ user-configurable cost functions with statistical evaluation
- ✅ optimal policy approximation with extensive state coverage
- ✅ robust simulation & evaluation with confidence intervals
- ✅ interactive streamlit interface with guided workflow
