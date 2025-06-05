"""
optimal control for depin protocols

complete workflow for training optimal controllers and comparing
them against user-defined policies
"""

import time
from typing import Dict, List, Any
from cost_function import CostFunction, CostWeights, CostTargets  
from value_function import ValueFunctionApproximator, TrainingConfig
from depin_utils import simulate_policy, get_stable_controls
import numpy as np


class OptimalController:
    """main class for optimal control analysis"""
    
    def __init__(self, cost_weights: CostWeights, cost_targets: CostTargets):
        self.cost_function = CostFunction(cost_weights, cost_targets)
        self.value_approximator = None
        self.is_trained = False
    
    def train_optimal_policy(self, training_config: TrainingConfig = None, 
                           verbose: bool = True) -> Dict[str, Any]:
        """train the optimal policy using fitted value iteration"""
        
        if training_config is None:
            training_config = TrainingConfig(
                num_trajectories=200,
                fvi_iterations=6,
                min_iterations=3,
                max_iter=300
            )
        
        if verbose:
            print("🚀 training optimal policy...")
            print(f"trajectories: {training_config.num_trajectories}, "
                  f"fvi_iterations: {training_config.fvi_iterations}")
        
        # create and train value function approximator
        self.value_approximator = ValueFunctionApproximator(
            self.cost_function, training_config
        )
        
        training_result = self.value_approximator.train(verbose=verbose)
        self.is_trained = True
        
        if verbose:
            print(f"✅ optimal policy training complete: {training_result['training_time']:.1f}s")
        
        return training_result
    
    def get_optimal_policy(self, state: Dict[str, float]) -> Dict[str, Any]:
        """get optimal controls for a given state"""
        if not self.is_trained:
            raise ValueError("optimal policy not trained yet - call train_optimal_policy() first")
        
        return self.value_approximator.get_optimal_policy(state)
    
    def compare_policies(self, initial_state: Dict[str, float], 
                        user_policies: Dict[str, Dict[str, float]],
                        simulation_timesteps: int = 50,
                        num_runs: int = 20,
                        random_seed: int = 42) -> Dict[str, Any]:
        """compare user policies against optimal policy"""
        
        if not self.is_trained:
            raise ValueError("optimal policy not trained yet")
        
        print(f"\n📊 comparing policies over {simulation_timesteps} timesteps ({num_runs} runs each)...")
        
        # get optimal policy for initial state
        optimal_result = self.get_optimal_policy(initial_state)
        optimal_policy = optimal_result['controls']
        
        # evaluate all policies with robust multi-run evaluation
        results = {}
        
        # optimal policy
        print("evaluating optimal policy...")
        optimal_evaluation = self.cost_function.evaluate_policy_robust(
            initial_state, optimal_policy, simulation_timesteps, num_runs, random_seed
        )
        results['optimal'] = {
            'policy': optimal_policy,
            'evaluation': optimal_evaluation,
            'cost': optimal_evaluation['total_discounted_cost'],
            'cost_std': optimal_evaluation['cost_std'],
            'confidence_95': optimal_evaluation['confidence_95']
        }
        
        # user policies - use same base seed for fair comparison
        for policy_name, policy in user_policies.items():
            print(f"evaluating {policy_name} policy...")
            evaluation = self.cost_function.evaluate_policy_robust(
                initial_state, policy, simulation_timesteps, num_runs, random_seed
            )
            results[policy_name] = {
                'policy': policy,
                'evaluation': evaluation,
                'cost': evaluation['total_discounted_cost'],
                'cost_std': evaluation['cost_std'],
                'confidence_95': evaluation['confidence_95']
            }
        
        # compute regret (user_cost - optimal_cost)
        optimal_cost = results['optimal']['cost']
        for policy_name in user_policies.keys():
            user_cost = results[policy_name]['cost']
            results[policy_name]['regret'] = user_cost - optimal_cost
            results[policy_name]['regret_percent'] = (user_cost - optimal_cost) / optimal_cost * 100
        
        # rank policies by cost
        policy_ranking = sorted(
            [(name, data['cost']) for name, data in results.items()],
            key=lambda x: x[1]
        )
        
        return {
            'results': results,
            'ranking': policy_ranking,
            'optimal_cost': optimal_cost,
            'num_runs': num_runs,
            'random_seed': random_seed
        }
    
    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """print human-readable comparison summary"""
        print(f"\n🏆 policy comparison results ({comparison.get('num_runs', 1)} runs each):")
        print("=" * 70)
        
        for i, (policy_name, cost) in enumerate(comparison['ranking']):
            rank = i + 1
            result = comparison['results'][policy_name]
            
            print(f"{rank}. {policy_name}:")
            print(f"   cost: {cost:.3f}")
            
            # show confidence interval if available
            if 'cost_std' in result:
                std = result['cost_std']
                ci_low, ci_high = result['confidence_95']
                print(f"   std: ±{std:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            
            if policy_name != 'optimal':
                regret = result['regret']
                regret_pct = result['regret_percent']
                print(f"   regret: {regret:.3f} ({regret_pct:+.1f}%)")
            
            # show policy details
            policy = result['policy']
            print(f"   controls: mint_rate={policy['mint_rate']:,.0f}, "
                  f"burn_share={policy['burn_share']:.2f}")
            
            # show dominant cost component
            breakdown = result['evaluation']['cost_breakdown']
            max_component = max(breakdown.items(), key=lambda x: x[1])
            total = result['evaluation']['total_discounted_cost']
            pct = (max_component[1] / total * 100) if total > 0 else 0
            print(f"   main cost driver: {max_component[0]} ({pct:.0f}%)")
            print()
        
        best_user_policy = None
        best_user_cost = float('inf')
        for name, data in comparison['results'].items():
            if name != 'optimal' and data['cost'] < best_user_cost:
                best_user_policy = name
                best_user_cost = data['cost']
        
        if best_user_policy:
            improvement = comparison['results'][best_user_policy]['regret']
            print(f"💡 optimal policy improves upon best user policy ({best_user_policy}) by {-improvement:.3f}")
            
            # check statistical significance
            if 'cost_std' in comparison['results']['optimal']:
                opt_std = comparison['results']['optimal']['cost_std']
                user_std = comparison['results'][best_user_policy]['cost_std']
                num_runs = comparison.get('num_runs', 1)
                
                # approximate standard error of difference
                se_diff = np.sqrt(opt_std**2 + user_std**2) / np.sqrt(num_runs)
                if abs(improvement) > 1.96 * se_diff:
                    print(f"   (statistically significant at 95% confidence)")
                else:
                    print(f"   (not statistically significant - difference may be due to noise)")


def demo_optimal_control():
    """demonstrate complete optimal control workflow"""
    
    print("🚀 depin optimal control demo")
    print("=" * 50)
    
    # define cost objectives (protocol designer's preferences)
    weights = CostWeights(
        price_growth=1.5,      # moderate concern for price stability
        utilization_min=2.5,   # high priority on network efficiency
        capacity_growth=1.0    # balanced capacity expansion
    )
    
    targets = CostTargets(
        price_growth=0.02,     # 2% target price growth
        utilization_min=0.8,   # 80% minimum utilization
        capacity_growth=0.01   # 1% target capacity growth
    )
    
    # create optimal controller
    controller = OptimalController(weights, targets)
    
    # train optimal policy (smaller config for demo)
    training_config = TrainingConfig(
        num_trajectories=100,
        fvi_iterations=4,
        max_iter=200
    )
    
    training_result = controller.train_optimal_policy(training_config)
    
    # example protocol state
    initial_state = {
        'supply': 50_000_000,
        'usd_reserve': 500_000,
        'price': 1.0,
        'capacity': 5000,
        'token_demand': 40_000_000,
        'service_demand': 4000,
        'market_factor': 0.0
    }
    
    # user-defined policies to compare
    user_policies = {
        'conservative': {'mint_rate': 10_000, 'burn_share': 0.8},
        'aggressive': {'mint_rate': 100_000, 'burn_share': 0.2},
        'balanced': {'mint_rate': 50_000, 'burn_share': 0.5},
        'auto_tuned': get_stable_controls(initial_state)
    }
    
    # compare policies
    comparison = controller.compare_policies(initial_state, user_policies, simulation_timesteps=40)
    
    # show results
    controller.print_comparison_summary(comparison)
    
    print("✅ optimal control demo complete")
    return controller, comparison


if __name__ == "__main__":
    demo_optimal_control() 