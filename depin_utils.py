"""
depin control tool utilities
"""

from system_dynamics import SystemState, SystemParams, ReactiveSystemState, ReactiveCapacityParams
from typing import Dict, List, Any
import numpy as np


def create_system(initial_state: Dict[str, float], custom_params: Dict[str, float] = None, 
                 use_reactive: bool = True) -> SystemState:
    if use_reactive:
        if custom_params:
            params = ReactiveCapacityParams(**custom_params)
        else:
            params = ReactiveCapacityParams()
        return ReactiveSystemState(initial_state, params)
    else:
        if custom_params:
            params = SystemParams(**custom_params)
        else:
            params = SystemParams()
        return SystemState(initial_state, params)


def get_stable_controls(initial_state: Dict[str, float], use_reactive: bool = True) -> Dict[str, float]:
    if use_reactive:
        supply = initial_state['supply']
        capacity = initial_state['capacity']
        utilization = initial_state['service_demand'] / max(capacity, 1)
        
        # mint rate: 0.1-0.4% of supply for reactive system
        if utilization > 0.85:
            suggested_mint = supply * 0.004
        elif utilization > 0.65:
            suggested_mint = supply * 0.002
        else:
            suggested_mint = supply * 0.001
        
        suggested_burn = 0.5
        
        return {
            'mint_rate': suggested_mint,
            'burn_share': suggested_burn
        }
    else:
        return SystemParams.suggest_stable_controls(initial_state)


def validate_inputs(initial_state: Dict[str, float], mint_rate: float, 
                   use_reactive: bool = True) -> Dict[str, Any]:
    if use_reactive:
        params = ReactiveCapacityParams()
    else:
        params = SystemParams()
    return params.validate_and_scale_inputs(initial_state, mint_rate)


def simulate_policy(initial_state: Dict[str, float], control_policy: Dict[str, float], 
                   timesteps: int = 50, custom_params: Dict[str, float] = None,
                   use_reactive: bool = True) -> Dict[str, List]:
    system = create_system(initial_state, custom_params, use_reactive)
    
    results = {key: [val] for key, val in system.get_current_state().items()}
    results['burn_tokens'] = []
    results['service_revenue_usd'] = []
    
    for t in range(timesteps):
        step_result = system.step(control_policy)
        
        for key in results:
            if key in step_result:
                results[key].append(step_result[key])
            elif key in system.get_current_state():
                results[key].append(system.get_current_state()[key])
    
    return results


def quick_stability_check(initial_state: Dict[str, float], control_policy: Dict[str, float],
                         timesteps: int = 20, use_reactive: bool = True) -> Dict[str, float]:
    results = simulate_policy(initial_state, control_policy, timesteps, use_reactive=use_reactive)
    
    # analyze final half for stability
    final_half = len(results['price']) // 2
    price_final = results['price'][final_half:]
    capacity_final = results['capacity'][final_half:]
    util_final = results['utilization'][final_half:]
    
    price_cv = np.std(price_final) / np.mean(price_final) if np.mean(price_final) > 0 else float('inf')
    capacity_trend = np.polyfit(range(len(capacity_final)), capacity_final, 1)[0]
    util_mean = np.mean(util_final)
    
    price_range = max(results['price']) / min(results['price']) if min(results['price']) > 0 else float('inf')
    
    # stability score (0-4)
    score = 0
    score += 1 if price_cv < 0.1 else 0
    score += 1 if abs(capacity_trend) < 10 else 0
    score += 1 if 0.5 <= util_mean <= 0.95 else 0
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


def calculate_control_ranges(initial_state: Dict[str, float], use_reactive: bool = True) -> Dict[str, Dict[str, float]]:
    """calculate reasonable control ranges based on initial system state"""
    
    supply = initial_state.get('supply', 50_000_000)
    capacity = initial_state.get('capacity', 5000)
    utilization = initial_state.get('service_demand', 0) / max(capacity, 1)
    
    if use_reactive:
        min_mint_pct = 0.0001
        max_mint_pct = 0.008
        
        if utilization > 0.85:
            default_mint_pct = 0.004
        elif utilization > 0.65:
            default_mint_pct = 0.002
        else:
            default_mint_pct = 0.001
    else:
        min_mint_pct = 0.0001
        max_mint_pct = 0.002
        default_mint_pct = 0.0005
    
    mint_range = {
        'min': max(1, int(supply * min_mint_pct)),
        'max': int(supply * max_mint_pct),
        'default': int(supply * default_mint_pct),
        'step': max(1, int(supply * 0.00005))
    }
    
    burn_range = {
        'min': 0.0,
        'max': 1.0,
        'default': 0.5,
        'step': 0.01
    }
    
    base_price_range = {
        'min': 0.1,
        'max': 10.0,
        'default': 1.0,
        'step': 0.1
    }
    
    elasticity_range = {
        'min': 0.1,
        'max': 3.0,
        'default': 1.0,
        'step': 0.1
    }
    
    return {
        'mint_rate': mint_range,
        'burn_share': burn_range,
        'base_service_price': base_price_range,
        'price_elasticity': elasticity_range,
        'system_info': {
            'supply': supply,
            'utilization': utilization
        }
    }


def suggest_stable_initial_state(user_input: Dict[str, float]) -> Dict[str, float]:
    """suggest improvements to user's initial state for better stability"""
    
    # use user's values as base
    suggested = user_input.copy()
    
    supply = user_input.get('supply', 10_000_000)
    capacity = user_input.get('capacity', 1000)
    
    # suggest reasonable relationships between variables
    
    # usd reserve: 1-5% of supply value at $1
    if 'usd_reserve' not in suggested or suggested['usd_reserve'] <= 0:
        suggested['usd_reserve'] = supply * 0.02
        
    # price: start at reasonable level
    if 'price' not in suggested or suggested['price'] <= 0:
        suggested['price'] = 1.0
    
    # token demand: 70-90% of supply for healthy economy
    if 'token_demand' not in suggested or suggested['token_demand'] <= 0:
        suggested['token_demand'] = supply * 0.8
        
    # service demand: 60-85% utilization
    if 'service_demand' not in suggested:
        suggested['service_demand'] = capacity * 0.75
    else:
        # adjust if utilization is extreme
        utilization = suggested['service_demand'] / max(capacity, 1)
        if utilization > 0.95:
            suggested['service_demand'] = capacity * 0.85
        elif utilization < 0.3:
            suggested['service_demand'] = capacity * 0.6
            
    # market factor: start neutral
    if 'market_factor' not in suggested:
        suggested['market_factor'] = 0.0
    
    return suggested


def validate_and_suggest_controls(initial_state: Dict[str, float], 
                                 proposed_controls: Dict[str, float],
                                 use_reactive: bool = True) -> Dict[str, Any]:
    """validate proposed controls and suggest improvements"""
    
    mint_rate = proposed_controls.get('mint_rate', 0)
    burn_share = proposed_controls.get('burn_share', 0.5)
    
    # get validation from system params
    validation = validate_inputs(initial_state, mint_rate, use_reactive)
    
    # collect all issues
    issues = validation.get('warnings', [])
    suggestions = validation.get('suggestions', {})
    
    # add control-specific validations
    if burn_share < 0.1 or burn_share > 0.9:
        issues.append(f"burn share ({burn_share:.1%}) should be between 10% and 90%")
        suggestions['burn_share'] = 0.5
    
    # suggest stable controls if current ones are problematic
    if validation.get('stability_score', 0) > 2:
        stable_controls = get_stable_controls(initial_state, use_reactive)
        suggestions.update(stable_controls)
        issues.append("policy may cause instability - consider suggested improvements")
    
    # determine if validation passed
    is_valid = len(issues) == 0
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'suggestions': suggestions,
        'recommended_controls': suggestions if suggestions else proposed_controls
    }


# test the utilities
if __name__ == "__main__":
    test_state = {
        'supply': 50_000_000,
        'usd_reserve': 500_000,
        'price': 1.0,
        'capacity': 5000,
        'token_demand': 40_000_000,
        'service_demand': 4000,
        'market_factor': 0.0
    }
    
    controls = get_stable_controls(test_state)
    stability = quick_stability_check(test_state, controls)
    
    print(f"✅ depin utilities validation: stability_score={stability['stability_score']}/4")
    print(f"   suggested controls: mint_rate={controls['mint_rate']:,.0f}, burn_share={controls['burn_share']:.2f}") 