"""
value function approximator for depin optimal control

uses fitted value iteration with sklearn mlp regressor to approximate
the optimal value function and extract optimal policies

includes comprehensive state space coverage, robust bellman optimization,
and validation methods for optimal policy verification
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass
from scipy.optimize import minimize, differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from cost_function import CostFunction
from depin_utils import simulate_policy, create_system


@dataclass
class TrainingConfig:
    """configuration for value function training"""
    num_trajectories: int = 300              # number of training trajectories for comprehensive coverage
    trajectory_length: int = 50              # timesteps per trajectory  
    fvi_iterations: int = 8                  # fitted value iteration rounds
    convergence_threshold: float = 0.001     # early stopping threshold
    min_iterations: int = 3                  # minimum iterations before early stopping
    max_iter: int = 500                      # max epochs for mlp training
    learning_rate_init: float = 0.001        # initial learning rate
    
    # comprehensive sampling parameters
    utilization_aware_sampling: bool = True   # focus on problematic utilization ranges
    state_space_coverage: str = 'comprehensive'  # 'basic', 'comprehensive', 'extreme'
    
    # robust optimization parameters
    bellman_optimization_attempts: int = 5    # multiple optimization attempts
    use_global_optimization: bool = True      # use differential evolution for global search


class ValueFunctionApproximator:
    """fitted value iteration with comprehensive coverage and robust optimization"""
    
    def __init__(self, cost_function: CostFunction, config: TrainingConfig = None):
        self.cost_function = cost_function
        self.config = config or TrainingConfig()
        
        # create mlp regressor with robust architecture
        self.value_network = MLPRegressor(
            hidden_layer_sizes=(96, 64, 32),  # expanded architecture for better approximation
            activation='tanh',
            solver='adam',
            learning_rate_init=self.config.learning_rate_init,
            max_iter=self.config.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,  # increased validation split
            n_iter_no_change=25
        )
        
        # state normalization
        self.state_scaler = StandardScaler()
        
        self.is_trained = False
        self.training_history = []
        self.validation_history = []
        
        # control bounds
        self.mint_rate_bounds = (0, None)  # will be set based on state
        self.burn_share_bounds = (0.0, 1.0)
    
    def sample_initial_states(self, num_states: int) -> List[Dict[str, float]]:
        """generate diverse initial states with comprehensive coverage"""
        states = []
        
        # define sampling strategies based on configuration
        if self.config.state_space_coverage == 'extreme':
            scale_range = (0.01, 1000.0)  # extreme scale differences
            util_focus_weight = 0.5       # 50% focus on problem utilization ranges
        elif self.config.state_space_coverage == 'comprehensive':
            scale_range = (0.1, 100.0)
            util_focus_weight = 0.3       # 30% focus on problem ranges
        else:  # basic
            scale_range = (0.5, 10.0)
            util_focus_weight = 0.1       # 10% focus
        
        for i in range(num_states):
            # sample across different protocol scales
            scale = np.random.uniform(*scale_range)
            
            base_supply = 10_000_000 * scale
            base_capacity = 1000 * scale
            base_reserve = 100_000 * scale
            
            # utilization-aware sampling for better coverage
            if self.config.utilization_aware_sampling and np.random.random() < util_focus_weight:
                # focus on problematic utilization ranges (0.5-0.9)
                target_utilization = np.random.uniform(0.5, 0.9)
                capacity = base_capacity * np.random.uniform(0.5, 2.0)
                service_demand = capacity * target_utilization * np.random.uniform(0.9, 1.1)
            else:
                # normal sampling
                capacity = base_capacity * np.random.uniform(0.5, 2.0)
                service_demand = capacity * np.random.uniform(0.3, 1.2)  # wider range
            
            state = {
                'supply': base_supply * np.random.uniform(0.5, 2.0),
                'usd_reserve': base_reserve * np.random.uniform(0.5, 3.0),
                'price': np.random.uniform(0.2, 5.0),  # wider price range
                'capacity': capacity,
                'token_demand': base_supply * np.random.uniform(0.4, 1.1),  # wider range
                'service_demand': service_demand,
                'market_factor': np.random.uniform(-0.2, 0.2)  # wider market factor range
            }
            states.append(state)
        
        return states
    
    def generate_training_data(self) -> Tuple[np.ndarray, List[Dict]]:
        """generate training trajectories with comprehensive state coverage"""
        all_states = []
        all_state_dicts = []
        
        print(f"generating {self.config.num_trajectories} training trajectories...")
        
        for i in range(self.config.num_trajectories):
            if i % 50 == 0:
                print(f"  trajectory {i+1}/{self.config.num_trajectories}")
            
            # comprehensive initial state sampling
            initial_states = self.sample_initial_states(1)
            initial_state = initial_states[0]
            
            # diverse policy sampling for exploration
            supply = initial_state['supply']
            mint_rate = supply * np.random.uniform(0.0001, 0.003)  # wider range
            burn_share = np.random.uniform(0.0, 1.0)
            policy = {'mint_rate': mint_rate, 'burn_share': burn_share}
            
            # simulate trajectory
            try:
                trajectory = simulate_policy(initial_state, policy, self.config.trajectory_length)
                
                # extract states with validation
                for t in range(len(trajectory['supply'])):
                    state_dict = {
                        'supply': trajectory['supply'][t],
                        'usd_reserve': trajectory['usd_reserve'][t],
                        'price': trajectory['price'][t],
                        'capacity': trajectory['capacity'][t],
                        'token_demand': trajectory.get('token_demand', [initial_state['token_demand']] * (t+1))[min(t, len(trajectory.get('token_demand', [])) - 1)],
                        'service_demand': trajectory.get('service_demand', [initial_state['service_demand']] * (t+1))[min(t, len(trajectory.get('service_demand', [])) - 1)]
                    }
                    
                    # validate state before adding
                    if all(isinstance(v, (int, float)) and not np.isnan(v) for v in state_dict.values()):
                        state_vector = self.state_to_vector(state_dict)
                        all_states.append(state_vector)
                        all_state_dicts.append(state_dict)
                    
            except Exception as e:
                print(f"  warning: trajectory {i} failed: {e}")
                continue
        
        print(f"generated {len(all_states)} state samples")
        return np.array(all_states), all_state_dicts
    
    def state_to_vector(self, state: Dict[str, float]) -> np.ndarray:
        """convert state dict to vector for neural network"""
        # ensure all required keys exist with defaults
        return np.array([
            state.get('supply', 0),
            state.get('usd_reserve', 0),
            state.get('price', 0),
            state.get('capacity', 0),
            state.get('token_demand', 0),
            state.get('service_demand', 0)
        ])
    
    def solve_bellman_equation(self, state: Dict[str, float], 
                             value_function: MLPRegressor) -> Tuple[Dict[str, float], float]:
        """solve single-step bellman equation with robust optimization"""
        
        def objective(controls):
            mint_rate, burn_share = controls
            policy = {'mint_rate': mint_rate, 'burn_share': burn_share}
            
            try:
                # simulate one step
                system = create_system(state)
                next_state_dict = system.step(policy)
                
                # compute instantaneous cost
                instant_cost = self.cost_function.compute_instant_cost(
                    next_state_dict, state, policy
                )
                
                # compute future value
                next_state_vector = self.state_to_vector(next_state_dict)
                if hasattr(self, 'state_scaler') and hasattr(self.state_scaler, 'mean_'):
                    next_state_normalized = self.state_scaler.transform(next_state_vector.reshape(1, -1))
                    future_value = value_function.predict(next_state_normalized)[0]
                else:
                    future_value = 0.0  # fallback for untrained scaler
                
                # bellman equation: cost + γ * V(next_state)
                total_value = instant_cost['total'] + self.cost_function.gamma * future_value
                return total_value
                
            except Exception as e:
                return 1e6  # large penalty for invalid controls
        
        # adaptive control bounds
        max_mint_rate = state['supply'] * 0.003  # slightly increased bound
        bounds = [(0, max_mint_rate), (0.0, 1.0)]
        
        # multiple optimization attempts with different methods
        best_value = float('inf')
        best_controls = None
        optimization_results = []
        
        # method 1: l-bfgs-b with multiple random starts
        for attempt in range(self.config.bellman_optimization_attempts):
            x0 = [
                np.random.uniform(0, max_mint_rate),
                np.random.uniform(0.0, 1.0)
            ]
            
            try:
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                if result.success:
                    optimization_results.append({
                        'method': 'L-BFGS-B',
                        'value': result.fun,
                        'controls': {'mint_rate': result.x[0], 'burn_share': result.x[1]},
                        'success': True
                    })
                    if result.fun < best_value:
                        best_value = result.fun
                        best_controls = {'mint_rate': result.x[0], 'burn_share': result.x[1]}
            except Exception as e:
                optimization_results.append({
                    'method': 'L-BFGS-B',
                    'success': False,
                    'error': str(e)
                })
        
        # method 2: differential evolution for global optimization
        if self.config.use_global_optimization and best_controls is None:
            try:
                result = differential_evolution(
                    objective, 
                    bounds, 
                    maxiter=50,  # limited iterations for speed
                    seed=42
                )
                if result.success:
                    optimization_results.append({
                        'method': 'DifferentialEvolution',
                        'value': result.fun,
                        'controls': {'mint_rate': result.x[0], 'burn_share': result.x[1]},
                        'success': True
                    })
                    if result.fun < best_value:
                        best_value = result.fun
                        best_controls = {'mint_rate': result.x[0], 'burn_share': result.x[1]}
            except Exception as e:
                optimization_results.append({
                    'method': 'DifferentialEvolution',
                    'success': False,
                    'error': str(e)
                })
        
        # fallback to grid search if all optimizations failed
        if best_controls is None:
            mint_rates = np.linspace(0, max_mint_rate, 5)
            burn_shares = np.linspace(0.0, 1.0, 5)
            
            for mint_rate in mint_rates:
                for burn_share in burn_shares:
                    try:
                        value = objective([mint_rate, burn_share])
                        if value < best_value:
                            best_value = value
                            best_controls = {'mint_rate': mint_rate, 'burn_share': burn_share}
                    except:
                        continue
        
        # final fallback
        if best_controls is None:
            best_controls = {'mint_rate': max_mint_rate * 0.5, 'burn_share': 0.5}
            best_value = objective([best_controls['mint_rate'], best_controls['burn_share']])
        
        return best_controls, best_value
    
    def validate_optimal_policy(self, test_states: List[Dict[str, float]],
                              num_validation_runs: int = 10) -> Dict[str, Any]:
        """validate that learned optimal policies are actually optimal"""
        if not self.is_trained:
            raise ValueError("value function not trained yet")
        
        validation_results = []
        
        for i, state in enumerate(test_states):
            print(f"validating state {i+1}/{len(test_states)}...")
            
            # get learned optimal policy
            learned_optimal = self.get_optimal_policy(state)
            learned_policy = learned_optimal['controls']
            
            # generate random alternative policies for comparison
            alternative_policies = []
            supply = state['supply']
            
            for _ in range(num_validation_runs):
                alt_policy = {
                    'mint_rate': np.random.uniform(0, supply * 0.002),
                    'burn_share': np.random.uniform(0.0, 1.0)
                }
                alternative_policies.append(alt_policy)
            
            # evaluate all policies
            learned_cost = self.cost_function.evaluate_policy(state, learned_policy, timesteps=30)['total_discounted_cost']
            
            alternative_costs = []
            for alt_policy in alternative_policies:
                try:
                    alt_cost = self.cost_function.evaluate_policy(state, alt_policy, timesteps=30)['total_discounted_cost']
                    alternative_costs.append(alt_cost)
                except:
                    alternative_costs.append(float('inf'))
            
            # check if learned policy is better than alternatives
            better_than_alternatives = sum(1 for cost in alternative_costs if learned_cost < cost)
            optimality_score = better_than_alternatives / len(alternative_costs)
            
            validation_results.append({
                'state_index': i,
                'learned_cost': learned_cost,
                'alternative_costs': alternative_costs,
                'optimality_score': optimality_score,
                'learned_policy': learned_policy
            })
        
        # aggregate results
        overall_optimality = np.mean([r['optimality_score'] for r in validation_results])
        
        return {
            'overall_optimality_score': overall_optimality,
            'individual_results': validation_results,
            'summary': f"learned policies beat {overall_optimality:.1%} of random alternatives on average"
        }
    
    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """train value function using fitted value iteration"""
        start_time = time.time()
        
        if verbose:
            print("🚀 starting value function training...")
            print(f"config: {self.config.num_trajectories} trajectories, {self.config.fvi_iterations} fvi iterations")
        
        # generate training data with comprehensive coverage
        training_states, training_state_dicts = self.generate_training_data()
        
        if len(training_states) == 0:
            raise ValueError("no training data generated")
        
        # normalize states for neural network
        self.state_scaler.fit(training_states)
        normalized_states = self.state_scaler.transform(training_states)
        
        # fitted value iteration
        current_values = np.zeros(len(training_states))  # initialize values
        
        for iteration in range(self.config.fvi_iterations):
            iter_start = time.time()
            if verbose:
                print(f"\nfvi iteration {iteration + 1}/{self.config.fvi_iterations}")
            
            # compute bellman targets with robust optimization
            new_values = np.zeros(len(training_states))
            successful_updates = 0
            
            for i, state_dict in enumerate(training_state_dicts):
                if i % 1000 == 0 and verbose:
                    print(f"  computing targets: {i+1}/{len(training_state_dicts)}")
                
                try:
                    _, bellman_value = self.solve_bellman_equation(state_dict, self.value_network)
                    new_values[i] = bellman_value
                    successful_updates += 1
                except Exception as e:
                    new_values[i] = current_values[i]  # keep old value if optimization fails
            
            if verbose:
                print(f"  successful bellman updates: {successful_updates}/{len(training_states)}")
            
            # train neural network on new targets
            if verbose:
                print(f"  training neural network...")
            
            self.value_network.fit(
                normalized_states, 
                new_values
            )
            
            # check convergence
            value_change = np.mean(np.abs(new_values - current_values))
            current_values = new_values.copy()
            
            iter_time = time.time() - iter_start
            if verbose:
                print(f"  iteration {iteration + 1} complete: {iter_time:.1f}s, value_change: {value_change:.6f}")
            
            self.training_history.append({
                'iteration': iteration + 1,
                'value_change': value_change,
                'successful_updates': successful_updates,
                'time': iter_time
            })
            
            # early stopping
            if iteration >= self.config.min_iterations and value_change < self.config.convergence_threshold:
                if verbose:
                    print(f"  early stopping: convergence achieved")
                break
        
        self.is_trained = True
        training_time = time.time() - start_time
        
        if verbose:
            print(f"\n✅ value function training complete: {training_time:.1f}s")
        
        return {
            'training_time': training_time,
            'iterations_completed': len(self.training_history),
            'final_value_change': self.training_history[-1]['value_change'] if self.training_history else 0,
            'training_samples': len(training_states)
        }
    
    def get_optimal_policy(self, state: Dict[str, float]) -> Dict[str, Any]:
        """get optimal controls for a given state"""
        if not self.is_trained:
            raise ValueError("optimal policy not trained yet - call train() first")
        
        optimal_controls, expected_value = self.solve_bellman_equation(state, self.value_network)
        
        return {
            'controls': optimal_controls,
            'expected_value': expected_value,
            'state': state
        }


# validation test
if __name__ == "__main__":
    from cost_function import CostFunction, CostWeights, CostTargets
    
    # quick validation with robust config
    weights = CostWeights(price_growth=1.5, utilization_min=2.0, capacity_growth=1.0)
    targets = CostTargets(price_growth=0.015, utilization_min=0.75, capacity_growth=0.01)
    cost_fn = CostFunction(weights, targets)
    
    config = TrainingConfig(
        num_trajectories=20, 
        fvi_iterations=2, 
        max_iter=200,
        state_space_coverage='comprehensive',
        utilization_aware_sampling=True,
        use_global_optimization=True
    )
    vfa = ValueFunctionApproximator(cost_fn, config)
    
    # train and test
    result = vfa.train(verbose=False)
    
    test_state = {
        'supply': 10_000_000, 'usd_reserve': 100_000, 'price': 1.0, 'capacity': 1000,
        'token_demand': 8_000_000, 'service_demand': 800, 'market_factor': 0.0
    }
    
    optimal = vfa.get_optimal_policy(test_state)
    print(f"✅ robust value function validation: training_time={result['training_time']:.1f}s, "
          f"optimal_value={optimal['expected_value']:.3f}") 