"""
optimal control for depin protocols

supports both inflation policy (mint_rate, burn_share) and 
service pricing policy (base_service_price, price_elasticity) optimization
"""

import time
from typing import Dict, List, Any
from cost_function import CostFunction, CostWeights, CostTargets  
from value_function import ValueFunctionApproximator, TrainingConfig
from depin_utils import simulate_policy, get_stable_controls
import numpy as np


class OptimalController:
    def __init__(self, cost_weights: CostWeights, cost_targets: CostTargets, controller_type: str = "inflation policy"):
        self.cost_function = CostFunction(cost_weights, cost_targets)
        self.controller_type = controller_type
        self.value_approximator = None
        self.is_trained = False
        
        if controller_type not in ["inflation policy", "service pricing"]:
            raise ValueError("controller_type must be 'inflation policy' or 'service pricing'")
    
    def train_optimal_policy(self, training_config: TrainingConfig = None, 
                           verbose: bool = True) -> Dict[str, Any]:
        
        if training_config is None:
            training_config = TrainingConfig(
                num_trajectories=200,
                fvi_iterations=6,
                min_iterations=3,
                max_iter=300
            )
        
        if verbose:
            print(f"🚀 training optimal {self.controller_type}...")
            print(f"trajectories: {training_config.num_trajectories}, "
                  f"fvi_iterations: {training_config.fvi_iterations}")
        
        self.value_approximator = ValueFunctionApproximator(
            self.cost_function, training_config, self.controller_type
        )
        
        training_result = self.value_approximator.train(verbose=verbose)
        self.is_trained = True
        
        if verbose:
            print(f"✅ optimal {self.controller_type} training complete: {training_result['training_time']:.1f}s")
        
        return training_result
    
    def get_optimal_policy(self, state: Dict[str, float]) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("optimal policy not trained yet - call train_optimal_policy() first")
        
        return self.value_approximator.get_optimal_policy(state)
    
    def compare_policies(self, initial_state: Dict[str, float], 
                        user_policies: Dict[str, Dict[str, float]],
                        simulation_timesteps: int = 50,
                        num_runs: int = 20,
                        random_seed: int = 42) -> Dict[str, Any]:
        
        if not self.is_trained:
            raise ValueError("optimal policy not trained yet")
        
        print(f"\n📊 comparing {self.controller_type} policies over {simulation_timesteps} timesteps ({num_runs} runs each)...")
        
        optimal_result = self.get_optimal_policy(initial_state)
        optimal_policy = optimal_result['controls']
        
        results = {}
        
        print("evaluating optimal policy...")
        optimal_evaluation = self._evaluate_policy_for_comparison(
            initial_state, optimal_policy, simulation_timesteps, num_runs, random_seed
        )
        results['optimal'] = {
            'policy': optimal_policy,
            'evaluation': optimal_evaluation,
            'cost': optimal_evaluation['total_discounted_cost'],
            'cost_std': optimal_evaluation['cost_std'],
            'confidence_95': optimal_evaluation['confidence_95']
        }
        
        for policy_name, policy in user_policies.items():
            print(f"evaluating {policy_name} policy...")
            evaluation = self._evaluate_policy_for_comparison(
                initial_state, policy, simulation_timesteps, num_runs, random_seed
            )
            results[policy_name] = {
                'policy': policy,
                'evaluation': evaluation,
                'cost': evaluation['total_discounted_cost'],
                'cost_std': evaluation['cost_std'],
                'confidence_95': evaluation['confidence_95']
            }
        
        # compute regret: user_cost - optimal_cost
        optimal_cost = results['optimal']['cost']
        for policy_name in user_policies.keys():
            user_cost = results[policy_name]['cost']
            results[policy_name]['regret'] = user_cost - optimal_cost
            results[policy_name]['regret_percent'] = (user_cost - optimal_cost) / optimal_cost * 100
        
        policy_ranking = sorted(
            [(name, data['cost']) for name, data in results.items()],
            key=lambda x: x[1]
        )
        
        return {
            'results': results,
            'ranking': policy_ranking,
            'optimal_cost': optimal_cost,
            'num_runs': num_runs,
            'random_seed': random_seed,
            'controller_type': self.controller_type
        }
    
    def _evaluate_policy_for_comparison(self, initial_state: Dict[str, float], 
                                       policy: Dict[str, float], 
                                       simulation_timesteps: int, 
                                       num_runs: int, 
                                       random_seed: int) -> Dict[str, Any]:
        
        if self.controller_type == "inflation policy":
            return self.cost_function.evaluate_policy_robust(
                initial_state, policy, simulation_timesteps, num_runs, random_seed
            )
        else:
            return self._evaluate_service_pricing_policy(
                initial_state, policy, simulation_timesteps, num_runs, random_seed
            )
    
    def _evaluate_service_pricing_policy(self, initial_state: Dict[str, float],
                                        policy: Dict[str, float],
                                        simulation_timesteps: int,
                                        num_runs: int,
                                        random_seed: int) -> Dict[str, Any]:
        
        base_price = policy.get('base_service_price', 1.0)
        elasticity = policy.get('price_elasticity', 1.0)
        mint_rate = policy.get('mint_rate', initial_state.get('mint_rate', 50000))
        
        # convert to inflation policy for evaluation
        equivalent_policy = {
            'mint_rate': mint_rate,
            'burn_share': 0.5
        }
        
        return self.cost_function.evaluate_policy_robust(
            initial_state, equivalent_policy, simulation_timesteps, num_runs, random_seed
        )
    
    def print_comparison_summary(self, comparison: Dict[str, Any]):
        controller_type = comparison.get('controller_type', 'inflation policy')
        
        print(f"\n🏆 {controller_type} comparison results ({comparison.get('num_runs', 1)} runs each):")
        print("=" * 70)
        
        for i, (policy_name, cost) in enumerate(comparison['ranking']):
            rank = i + 1
            result = comparison['results'][policy_name]
            
            print(f"{rank}. {policy_name}:")
            print(f"   cost: {cost:.3f}")
            
            if 'cost_std' in result:
                std = result['cost_std']
                ci_low, ci_high = result['confidence_95']
                print(f"   std: ±{std:.3f}, 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
            
            if 'regret' in result:
                regret = result['regret']
                regret_pct = result['regret_percent']
                print(f"   regret: {regret:+.3f} ({regret_pct:+.1f}%)")
            
            if controller_type == "inflation policy":
                policy = result['policy']
                print(f"   mint_rate: {policy['mint_rate']:,.0f}")
                print(f"   burn_share: {policy['burn_share']:.2f}")
            
            print("")


def demo_optimal_control():
    print("🎯 optimal control demo")
    
    weights = CostWeights(price_growth=2.0, utilization_min=3.0, capacity_growth=1.0)
    targets = CostTargets(price_growth=0.015, utilization_min=0.8, capacity_growth=0.02)
    
    controller = OptimalController(weights, targets, "inflation policy")
    
    config = TrainingConfig(
        num_trajectories=100,
        fvi_iterations=4,
        max_iter=200
    )
    
    training_result = controller.train_optimal_policy(config)
    print(f"training completed in {training_result['training_time']:.1f}s")
    
    initial_state = {
        'supply': 50_000_000,
        'usd_reserve': 500_000,
        'price': 1.0,
        'capacity': 5000,
        'token_demand': 40_000_000,
        'service_demand': 4000,
        'market_factor': 0.0
    }
    
    optimal_result = controller.get_optimal_policy(initial_state)
    print(f"optimal policy: {optimal_result['controls']}")
    
    user_policies = {
        'conservative': {'mint_rate': 25000, 'burn_share': 0.3},
        'aggressive': {'mint_rate': 100000, 'burn_share': 0.8},
        'balanced': {'mint_rate': 50000, 'burn_share': 0.5}
    }
    
    comparison = controller.compare_policies(initial_state, user_policies, 
                                           simulation_timesteps=30, num_runs=10)
    controller.print_comparison_summary(comparison)


if __name__ == "__main__":
    demo_optimal_control() 