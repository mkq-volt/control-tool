"""
depin control tool utilities

main functions for working with depin protocol dynamics
"""

from system_dynamics import SystemState, SystemParams
from typing import Dict, List, Any
import numpy as np


def create_system(initial_state: Dict[str, float], custom_params: Dict[str, float] = None) -> SystemState:
    """create a new system with optional custom parameters"""
    if custom_params:
        params = SystemParams(**custom_params)
    else:
        params = SystemParams()
    
    return SystemState(initial_state, params)


def get_stable_controls(initial_state: Dict[str, float]) -> Dict[str, float]:
    """get suggested stable control parameters for given initial state"""
    return SystemParams.suggest_stable_controls(initial_state)


def validate_inputs(initial_state: Dict[str, float], mint_rate: float) -> Dict[str, Any]:
    """validate system inputs and get suggestions for stability"""
    params = SystemParams()
    return params.validate_and_scale_inputs(initial_state, mint_rate)


def simulate_policy(initial_state: Dict[str, float], control_policy: Dict[str, float], 
                   timesteps: int = 50, custom_params: Dict[str, float] = None) -> Dict[str, List]:
    """simulate a control policy and return results"""
    system = create_system(initial_state, custom_params)
    
    results = {key: [val] for key, val in system.get_current_state().items()}
    results['burn_tokens'] = []
    results['service_revenue_usd'] = []
    
    for t in range(timesteps):
        step_result = system.step(control_policy)
        
        # record all metrics
        for key in results:
            if key in step_result:
                results[key].append(step_result[key])
            elif key in system.get_current_state():
                results[key].append(system.get_current_state()[key])
    
    return results


def quick_stability_check(initial_state: Dict[str, float], control_policy: Dict[str, float],
                         timesteps: int = 20) -> Dict[str, float]:
    """run quick simulation and return stability metrics"""
    results = simulate_policy(initial_state, control_policy, timesteps)
    
    # analyze final half for stability
    final_half = len(results['price']) // 2
    price_final = results['price'][final_half:]
    capacity_final = results['capacity'][final_half:]
    util_final = results['utilization'][final_half:]
    
    # calculate metrics
    price_cv = np.std(price_final) / np.mean(price_final) if np.mean(price_final) > 0 else float('inf')
    capacity_trend = np.polyfit(range(len(capacity_final)), capacity_final, 1)[0]
    util_mean = np.mean(util_final)
    
    price_range = max(results['price']) / min(results['price']) if min(results['price']) > 0 else float('inf')
    
    # stability score (0-4, higher = more stable)
    score = 0
    score += 1 if price_cv < 0.1 else 0
    score += 1 if abs(capacity_trend) < 5 else 0
    score += 1 if 0.6 <= util_mean <= 0.9 else 0
    score += 1 if price_range < 5 else 0
    
    return {
        'stability_score': score,
        'price_cv': price_cv,
        'capacity_trend': capacity_trend,
        'util_mean': util_mean,
        'price_range': price_range,
        'final_price': results['price'][-1],
        'final_capacity': results['capacity'][-1],
        'final_utilization': results['utilization'][-1]
    }


# example usage
if __name__ == "__main__":
    # example initial state
    state = {
        'supply': 10_000_000,
        'usd_reserve': 100_000,
        'price': 1.0,
        'capacity': 1000,
        'token_demand': 8_000_000,
        'service_demand': 800,
        'market_factor': 0.0
    }
    
    # get stable controls
    controls = get_stable_controls(state)
    print(f"suggested controls: {controls}")
    
    # validate
    validation = validate_inputs(state, controls['mint_rate'])
    print(f"validation: {validation['stability_score']} warnings")
    
    # quick check
    stability = quick_stability_check(state, controls)
    print(f"stability score: {stability['stability_score']}/4")
    
    print("✅ depin utilities ready") 