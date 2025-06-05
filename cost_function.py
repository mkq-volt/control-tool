"""
cost function implementation for depin protocol optimization

allows users to specify weights for different objectives and computes
both instantaneous and cumulative discounted costs
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class CostWeights:
    """user-configurable cost weights"""
    price_growth: float = 1.0        # weight for price growth deviation
    utilization_min: float = 1.0     # weight for utilization below minimum
    capacity_growth: float = 1.0     # weight for capacity growth deviation


@dataclass 
class CostTargets:
    """user-configurable cost targets"""
    price_growth: float = 0.02       # target price growth rate per timestep (2%)
    utilization_min: float = 0.7     # minimum acceptable utilization rate
    capacity_growth: float = 0.01    # target capacity growth rate per timestep (1%)


class CostFunction:
    """computes protocol costs based on user-defined weights and targets"""
    
    def __init__(self, weights: CostWeights, targets: CostTargets, 
                 discount_factor: float = 0.95):
        self.weights = weights
        self.targets = targets
        self.gamma = discount_factor
        
        # validate inputs
        if not 0 < discount_factor <= 1:
            raise ValueError("discount_factor must be in (0, 1]")
        
        # bound targets to reasonable ranges
        self.targets.price_growth = np.clip(targets.price_growth, -0.1, 0.1)  # ±10%
        self.targets.utilization_min = np.clip(targets.utilization_min, 0.3, 0.95)
        self.targets.capacity_growth = np.clip(targets.capacity_growth, -0.05, 0.1)  # -5% to 10%
    
    def compute_instant_cost(self, current_state: Dict[str, float], 
                           previous_state: Dict[str, float], 
                           control_actions: Dict[str, float]) -> Dict[str, float]:
        """compute instantaneous cost for single timestep"""
        
        # calculate utilization if not provided
        if 'utilization' not in current_state:
            if current_state.get('capacity', 0) > 0:
                current_state['utilization'] = min(1.0, current_state.get('service_demand', 0) / current_state['capacity'])
            else:
                current_state['utilization'] = 0.0
        
        # price growth penalty (normalized by previous price)
        if previous_state['price'] > 0:
            actual_price_growth = (current_state['price'] - previous_state['price']) / previous_state['price']
        else:
            actual_price_growth = 0.0
        
        price_penalty = (actual_price_growth - self.targets.price_growth) ** 2
        
        # utilization penalty (soft constraint for minimum)
        util_shortfall = max(0, self.targets.utilization_min - current_state['utilization'])
        util_penalty = util_shortfall ** 2
        
        # capacity growth penalty (normalized by previous capacity)
        if previous_state['capacity'] > 0:
            actual_capacity_growth = (current_state['capacity'] - previous_state['capacity']) / previous_state['capacity']
        else:
            actual_capacity_growth = 0.0
            
        capacity_penalty = (actual_capacity_growth - self.targets.capacity_growth) ** 2
        
        # weighted total cost
        weighted_costs = {
            'price_growth': self.weights.price_growth * price_penalty,
            'utilization_min': self.weights.utilization_min * util_penalty,
            'capacity_growth': self.weights.capacity_growth * capacity_penalty
        }
        
        total_cost = sum(weighted_costs.values())
        
        return {
            'total': total_cost,
            'breakdown': weighted_costs,
            'raw_penalties': {
                'price_growth': price_penalty,
                'utilization_min': util_penalty, 
                'capacity_growth': capacity_penalty
            },
            'actual_values': {
                'price_growth': actual_price_growth,
                'utilization': current_state['utilization'],
                'capacity_growth': actual_capacity_growth
            }
        }
    
    def compute_trajectory_cost(self, trajectory: Dict[str, List[float]], 
                              control_sequence: List[Dict[str, float]]) -> Dict[str, Any]:
        """compute cumulative discounted cost over entire trajectory"""
        
        # trajectory has T+1 states (initial + T steps), control_sequence has T actions
        T = len(control_sequence)
        
        # use only core state variables that should have consistent length
        core_state_vars = ['supply', 'usd_reserve', 'price', 'capacity', 'utilization']
        
        # verify trajectory has correct length for core variables
        for var in core_state_vars:
            if var not in trajectory:
                raise ValueError(f"trajectory missing required variable: {var}")
            if len(trajectory[var]) != T + 1:
                raise ValueError(f"trajectory[{var}] length ({len(trajectory[var])}) must be {T + 1}")
        
        discounted_costs = []
        cumulative_breakdown = {
            'price_growth': 0.0,
            'utilization_min': 0.0,
            'capacity_growth': 0.0
        }
        
        # compute cost for each timestep
        for t in range(T):
            # extract core state variables only
            current_state = {var: trajectory[var][t + 1] for var in core_state_vars}
            previous_state = {var: trajectory[var][t] for var in core_state_vars}
            
            # instantaneous cost
            instant_cost = self.compute_instant_cost(current_state, previous_state, control_sequence[t])
            
            # apply discount factor
            discount = self.gamma ** t
            discounted_cost = instant_cost['total'] * discount
            discounted_costs.append(discounted_cost)
            
            # accumulate breakdown with discounting
            for component, value in instant_cost['breakdown'].items():
                cumulative_breakdown[component] += value * discount
        
        total_discounted_cost = sum(discounted_costs)
        
        # compute average instantaneous cost (undiscounted equivalent)
        if T > 0 and self.gamma < 1:
            discount_sum = (1 - self.gamma ** T) / (1 - self.gamma)
            average_instantaneous = total_discounted_cost / discount_sum if discount_sum > 0 else 0
        else:
            average_instantaneous = total_discounted_cost / T if T > 0 else 0
        
        return {
            'total_discounted_cost': total_discounted_cost,
            'average_instantaneous_cost': average_instantaneous,
            'cost_breakdown': cumulative_breakdown,
            'timestep_costs': discounted_costs,
            'horizon': T,
            'discount_factor': self.gamma
        }
    
    def evaluate_policy(self, initial_state: Dict[str, float], 
                       control_policy: Dict[str, float], 
                       timesteps: int = 50) -> Dict[str, Any]:
        """evaluate a fixed control policy over a simulation horizon"""
        from depin_utils import simulate_policy
        
        # run simulation
        trajectory = simulate_policy(initial_state, control_policy, timesteps)
        
        # create control sequence (repeated policy)
        control_sequence = [control_policy] * timesteps
        
        # compute trajectory cost
        cost_result = self.compute_trajectory_cost(trajectory, control_sequence)
        
        return {
            **cost_result,
            'trajectory': trajectory,
            'control_policy': control_policy,
            'final_state': {key: vals[-1] for key, vals in trajectory.items()}
        }
    
    def evaluate_policy_robust(self, initial_state: Dict[str, float],
                              control_policy: Dict[str, float],
                              timesteps: int = 50,
                              num_runs: int = 10,
                              random_seed: int = None) -> Dict[str, Any]:
        """evaluate policy with multiple runs for robust statistics"""
        from depin_utils import simulate_policy
        import numpy as np
        
        # store results from multiple runs
        costs = []
        final_states = []
        cost_breakdowns = []
        
        # save original random state
        original_state = np.random.get_state()
        
        try:
            for run in range(num_runs):
                # set deterministic seed for this run if provided
                if random_seed is not None:
                    np.random.seed(random_seed + run)
                
                # run simulation
                trajectory = simulate_policy(initial_state, control_policy, timesteps)
                control_sequence = [control_policy] * timesteps
                
                # compute costs
                cost_result = self.compute_trajectory_cost(trajectory, control_sequence)
                
                costs.append(cost_result['total_discounted_cost'])
                final_states.append({key: vals[-1] for key, vals in trajectory.items()})
                cost_breakdowns.append(cost_result['cost_breakdown'])
        
        finally:
            # restore original random state
            np.random.set_state(original_state)
        
        # compute statistics
        costs = np.array(costs)
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        min_cost = np.min(costs)
        max_cost = np.max(costs)
        
        # average cost breakdown
        avg_breakdown = {}
        for component in cost_breakdowns[0].keys():
            component_values = [breakdown[component] for breakdown in cost_breakdowns]
            avg_breakdown[component] = np.mean(component_values)
        
        # average final state
        avg_final_state = {}
        for key in final_states[0].keys():
            values = [state[key] for state in final_states]
            avg_final_state[key] = np.mean(values)
        
        return {
            'total_discounted_cost': mean_cost,
            'cost_std': std_cost,
            'cost_min': min_cost,
            'cost_max': max_cost,
            'cost_breakdown': avg_breakdown,
            'final_state': avg_final_state,
            'num_runs': num_runs,
            'all_costs': costs.tolist(),
            'control_policy': control_policy,
            'confidence_95': (mean_cost - 1.96 * std_cost / np.sqrt(num_runs),
                             mean_cost + 1.96 * std_cost / np.sqrt(num_runs))
        }
    
    def get_cost_summary(self, trajectory_cost: Dict[str, Any]) -> str:
        """generate human-readable cost summary"""
        total = trajectory_cost['total_discounted_cost']
        avg = trajectory_cost['average_instantaneous_cost']
        breakdown = trajectory_cost['cost_breakdown']
        
        summary = f"total discounted cost: {total:.3f}\n"
        summary += f"average instantaneous: {avg:.3f}\n"
        summary += "cost breakdown:\n"
        
        for component, value in breakdown.items():
            percentage = (value / total * 100) if total > 0 else 0
            summary += f"  {component}: {value:.3f} ({percentage:.1f}%)\n"
        
        return summary


# validation test
if __name__ == "__main__":
    # simple validation test
    weights = CostWeights(price_growth=1.5, utilization_min=2.0, capacity_growth=0.5)
    targets = CostTargets(price_growth=0.01, utilization_min=0.75, capacity_growth=0.005)
    cost_fn = CostFunction(weights, targets)
    
    # test states
    prev_state = {'price': 1.0, 'utilization': 0.8, 'capacity': 1000, 'supply': 10_000_000, 'usd_reserve': 150_000}
    curr_state = {'price': 1.02, 'utilization': 0.65, 'capacity': 1005, 'supply': 10_005_000, 'usd_reserve': 155_000}
    controls = {'mint_rate': 5000, 'burn_share': 0.4}
    
    instant_cost = cost_fn.compute_instant_cost(curr_state, prev_state, controls)
    print(f"✅ cost function validation: total={instant_cost['total']:.3f}") 