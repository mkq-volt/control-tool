"""
value function approximator for depin optimal control

uses fitted value iteration with sklearn mlp regressor to approximate
the optimal value function and extract optimal policies

includes comprehensive state space coverage, robust bellman optimization,
and validation methods for optimal policy verification

supports both original fitted value iteration and improved training approaches
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass, field
from scipy.optimize import minimize, differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from cost_function import CostFunction
from depin_utils import simulate_policy, create_system

# note: pytorch availability checked only when needed to avoid streamlit conflicts

def _is_running_in_streamlit():
    """detect if we're running in a streamlit environment"""
    try:
        import streamlit as st
        return True
    except ImportError:
        return False


@dataclass
class TrainingConfig:
    """configuration for value function training"""
    num_trajectories: int = 150
    trajectory_length: int = 30
    fvi_iterations: int = 4
    max_iter: int = 200
    convergence_threshold: float = 0.001
    
    state_space_coverage: str = 'comprehensive'
    utilization_aware_sampling: bool = True
    
    use_global_optimization: bool = True
    policy_search_method: str = 'grid'
    
    use_improved_training: bool = True
    enhanced_sampling: bool = True
    use_neural_network: bool = True
    use_cross_entropy_method: bool = True
    
    hidden_size: int = 256
    num_layers: int = 4
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 64
    train_epochs: int = 100
    
    policy_grid_samples: int = 200
    best_policy_samples: int = 50
    cem_elite_fraction: float = 0.1
    cem_iterations: int = 5
    
    bellman_optimization_attempts: int = 3


class ImprovedValueNetwork:
    """enhanced neural network for value function approximation"""
    
    def __init__(self, state_dim: int, config: TrainingConfig):
        self.state_dim = state_dim
        self.config = config
        self.network = None
        self.device = None
        self._setup_network()
    
    def _setup_network(self):
        """setup pytorch network components"""
        import torch
        import torch.nn as nn
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        layers = []
        prev_dim = self.state_dim
        
        for i in range(self.config.num_layers):
            layers.append(nn.Linear(prev_dim, self.config.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout_rate))
            prev_dim = self.config.hidden_size
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers).to(self.device)
        
        # initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def __call__(self, x):
        return self.forward(x)
    
    def parameters(self):
        """return network parameters for optimizer"""
        return self.network.parameters()
    
    def train(self, mode=True):
        """set training mode"""
        self.network.train(mode)
        return self
    
    def eval(self):
        """set evaluation mode"""
        self.network.eval()
        return self


class ValueFunctionApproximator:
    """fitted value iteration with comprehensive coverage and robust optimization"""
    
    def __init__(self, cost_function: CostFunction, config: TrainingConfig, 
                 controller_type: str = 'inflation policy'):
        self.cost_function = cost_function
        self.config = config
        self.controller_type = controller_type
        
        # automatically disable improved training in streamlit to avoid pytorch conflicts
        use_improved = config.use_improved_training and not _is_running_in_streamlit()
        
        # choose training approach - check pytorch availability only when needed
        if use_improved:
            try:
                import torch
                self._init_improved_training()
            except ImportError:
                print("warning: pytorch not available, falling back to original training")
                self._init_original_training()
        else:
            if _is_running_in_streamlit() and config.use_improved_training:
                print("info: running in streamlit, using original training for compatibility")
            self._init_original_training()
        
        # control bounds based on controller type
        if self.controller_type == "inflation policy":
            self.mint_rate_bounds = (0, None)  # will be set based on state
            self.burn_share_bounds = (0.0, 1.0)
        else:  # service pricing
            self.base_price_bounds = (0.1, 10.0)
            self.elasticity_bounds = (0.1, 3.0)
            self.mint_rate_bounds = (0, None)  # still need mint rate for service pricing
        
        # create mlp regressor with robust architecture
        self.value_network = MLPRegressor(
            hidden_layer_sizes=(96, 64, 32),  # expanded architecture for better approximation
            activation='tanh',
            solver='adam',
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
    
    def _init_improved_training(self):
        """initialize improved training components"""
        self.training_mode = 'improved'
        
        # state normalization
        self.state_mean = None
        self.state_std = None
        
        # neural network components - import pytorch here
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state_dim = 7
        self.network = ImprovedValueNetwork(self.state_dim, self.config)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)
        self.criterion = nn.MSELoss()
        
        # training data storage
        self.training_states = []
        self.training_values = []
        self.training_history = []
        
        print(f"initialized improved value function training on {self.device}")
    
    def _init_original_training(self):
        """initialize original training components"""
        self.training_mode = 'original'
        
        # state scaler (will be initialized during training)
        self.state_scaler = None
        
        # training data storage
        self.training_states = []
        self.training_values = []
        self.training_history = []
        
        # value network (will be initialized during training)
        self.value_network = None
        
        print("initialized original value function training")
    
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
        
        print(f"generating {self.config.num_trajectories} training trajectories for {self.controller_type}...")
        
        for i in range(self.config.num_trajectories):
            if i % 50 == 0:
                print(f"  trajectory {i+1}/{self.config.num_trajectories}")
            
            # comprehensive initial state sampling
            initial_states = self.sample_initial_states(1)
            initial_state = initial_states[0]
            
            # diverse policy sampling for exploration based on controller type
            supply = initial_state['supply']
            
            if self.controller_type == "inflation policy":
                # sample inflation policy parameters
                mint_rate = supply * np.random.uniform(0.0001, 0.003)  # wider range
                burn_share = np.random.uniform(0.0, 1.0)
                policy = {'mint_rate': mint_rate, 'burn_share': burn_share}
            else:
                # sample service pricing parameters + mint rate
                base_price = np.random.uniform(0.1, 10.0)
                elasticity = np.random.uniform(0.1, 3.0)
                mint_rate = supply * np.random.uniform(0.0001, 0.003)
                policy = {
                    'base_service_price': base_price,
                    'price_elasticity': elasticity,
                    'mint_rate': mint_rate
                }
            
            # simulate trajectory - for service pricing, we'll use equivalent inflation policy
            try:
                if self.controller_type == "service pricing":
                    # convert to inflation policy for simulation (simplified)
                    equivalent_policy = {
                        'mint_rate': policy['mint_rate'],
                        'burn_share': 0.5  # neutral burn share
                    }
                    trajectory = simulate_policy(initial_state, equivalent_policy, self.config.trajectory_length)
                else:
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
        
        print(f"generated {len(all_states)} state samples for {self.controller_type}")
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
        
        if self.controller_type == "inflation policy":
            return self._solve_inflation_policy_bellman(state, value_function)
        else:
            return self._solve_service_pricing_bellman(state, value_function)
    
    def _solve_inflation_policy_bellman(self, state: Dict[str, float], 
                                       value_function: MLPRegressor) -> Tuple[Dict[str, float], float]:
        """solve bellman equation for inflation policy controls"""
        
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
        
        return self._optimize_bellman_objective(objective, bounds, max_mint_rate)
    
    def _solve_service_pricing_bellman(self, state: Dict[str, float], 
                                      value_function: MLPRegressor) -> Tuple[Dict[str, float], float]:
        """solve bellman equation for service pricing controls"""
        
        def objective(controls):
            base_price, elasticity, mint_rate = controls
            policy = {
                'base_service_price': base_price,
                'price_elasticity': elasticity,
                'mint_rate': mint_rate
            }
            
            try:
                # for service pricing, convert to equivalent inflation policy for simulation
                equivalent_policy = {
                    'mint_rate': mint_rate,
                    'burn_share': 0.5  # neutral burn share
                }
                
                # simulate one step
                system = create_system(state)
                next_state_dict = system.step(equivalent_policy)
                
                # compute instantaneous cost
                instant_cost = self.cost_function.compute_instant_cost(
                    next_state_dict, state, equivalent_policy
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
        
        # control bounds for service pricing
        max_mint_rate = state['supply'] * 0.003
        bounds = [
            self.base_price_bounds,     # base_service_price
            self.elasticity_bounds,     # price_elasticity  
            (0, max_mint_rate)          # mint_rate
        ]
        
        result = self._optimize_bellman_objective(objective, bounds, max_mint_rate, is_service_pricing=True)
        return result
    
    def _optimize_bellman_objective(self, objective, bounds, max_mint_rate, is_service_pricing=False):
        """optimize bellman objective with multiple methods"""
        
        # multiple optimization attempts with different methods
        best_value = float('inf')
        best_controls = None
        optimization_results = []
        
        # method 1: l-bfgs-b with multiple random starts
        for attempt in range(self.config.bellman_optimization_attempts):
            if is_service_pricing:
                x0 = [
                    np.random.uniform(*self.base_price_bounds),
                    np.random.uniform(*self.elasticity_bounds),
                    np.random.uniform(0, max_mint_rate)
                ]
            else:
                x0 = [
                    np.random.uniform(0, max_mint_rate),
                    np.random.uniform(0.0, 1.0)
                ]
            
            try:
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
                if result.success:
                    if is_service_pricing:
                        controls = {
                            'base_service_price': result.x[0],
                            'price_elasticity': result.x[1], 
                            'mint_rate': result.x[2]
                        }
                    else:
                        controls = {'mint_rate': result.x[0], 'burn_share': result.x[1]}
                    
                    optimization_results.append({
                        'method': 'L-BFGS-B',
                        'value': result.fun,
                        'controls': controls,
                        'success': True
                    })
                    if result.fun < best_value:
                        best_value = result.fun
                        best_controls = controls
            except Exception:
                pass
        
        # method 2: global optimization if enabled
        if self.config.use_global_optimization and best_controls is None:
            try:
                result = differential_evolution(objective, bounds, seed=42, maxiter=100)
                if result.success:
                    if is_service_pricing:
                        controls = {
                            'base_service_price': result.x[0],
                            'price_elasticity': result.x[1],
                            'mint_rate': result.x[2]
                        }
                    else:
                        controls = {'mint_rate': result.x[0], 'burn_share': result.x[1]}
                    
                    optimization_results.append({
                        'method': 'Differential Evolution',
                        'value': result.fun,
                        'controls': controls,
                        'success': True
                    })
                    if result.fun < best_value:
                        best_value = result.fun
                        best_controls = controls
            except Exception:
                pass
        
        # fallback to default controls if optimization failed
        if best_controls is None:
            if is_service_pricing:
                best_controls = {
                    'base_service_price': 1.0,
                    'price_elasticity': 1.0,
                    'mint_rate': max_mint_rate * 0.1
                }
            else:
                best_controls = {'mint_rate': max_mint_rate * 0.1, 'burn_share': 0.5}
            best_value = objective([best_controls[k] for k in best_controls.keys()])
        
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
        """train the value function using the selected approach"""
        if self.training_mode == 'improved':
            return self._train_improved(verbose)
        else:
            return self._train_original(verbose)
    
    def _train_improved(self, verbose: bool = True) -> Dict[str, Any]:
        """improved training with PyTorch neural networks and enhanced sampling"""
        start_time = time.time()
        
        if verbose:
            print("🚀 starting improved value function training...")
            print(f"config: {self.config.num_trajectories} trajectories, {self.config.fvi_iterations} iterations")
        
        # generate comprehensive training data
        training_data = self._generate_enhanced_training_data(verbose)
        
        if not training_data:
            raise ValueError("No valid training data generated")
        
        # separate states and values
        states, values = zip(*training_data)
        
        # iterative training approach
        training_history = []
        
        for iteration in range(self.config.fvi_iterations):
            if verbose:
                print(f"\nIteration {iteration + 1}/{self.config.fvi_iterations}")
            
            iter_start_time = time.time()
            
            # train neural network for this iteration
            nn_result = self._train_neural_network(list(states), list(values), verbose=False)
            
            # if not the last iteration, update training data using current value function
            if iteration < self.config.fvi_iterations - 1:
                # import torch for this specific usage
                import torch
                
                # re-evaluate states with current value function to get updated targets
                updated_values = []
                for state in states:
                    try:
                        # use current value function to get updated value estimate
                        state_tensor = self._state_to_tensor(state)
                        with torch.no_grad():
                            updated_value = self.network(state_tensor).item()
                        updated_values.append(updated_value)
                    except:
                        # fallback to original value if neural network fails
                        updated_values.append(values[list(states).index(state)])
                
                values = updated_values
            
            iter_time = time.time() - iter_start_time
            
            # calculate value change (use training loss as proxy)
            current_loss = nn_result.get('final_loss', 0)
            if iteration > 0:
                prev_loss = training_history[-1]['loss']
                value_change = abs(current_loss - prev_loss)
            else:
                value_change = current_loss
            
            training_history.append({
                'iteration': iteration + 1,
                'loss': current_loss,
                'value_change': value_change,
                'time': iter_time
            })
            
            if verbose:
                print(f"  iteration {iteration + 1} complete: {iter_time:.1f}s, loss: {current_loss:.6f}, change: {value_change:.6f}")
            
            # check convergence
            if value_change < self.config.convergence_threshold and iteration > 0:
                if verbose:
                    print("  early stopping: convergence achieved")
                break
        
        training_time = time.time() - start_time
        
        # mark as trained
        self.is_trained = True
        
        if verbose:
            print(f"✅ improved training complete: {training_time:.1f}s")
        
        # return compatible format with original training
        return {
            'training_time': training_time,
            'iterations_completed': len(training_history),
            'final_value_change': training_history[-1]['value_change'] if training_history else 0,
            'training_samples': len(training_data),
            'samples_used': len(training_data),
            'final_loss': training_history[-1]['loss'] if training_history else 0,
            'converged': training_history[-1]['value_change'] < self.config.convergence_threshold if len(training_history) > 1 else False,
            'method': 'improved'
        }
    
    def _train_original(self, verbose: bool = True) -> Dict[str, Any]:
        """original fitted value iteration training"""
        start_time = time.time()
        
        if verbose:
            print("🚀 starting value function training...")
            print(f"config: {self.config.num_trajectories} trajectories, {self.config.fvi_iterations} fvi iterations")
        
        # generate training trajectories using original method
        self._generate_training_trajectories(verbose)
        
        # fitted value iteration
        for iteration in range(self.config.fvi_iterations):
            if verbose:
                print(f"\nfvi iteration {iteration + 1}/{self.config.fvi_iterations}")
            
            iter_start_time = time.time()
            value_change = self._perform_fvi_iteration(iteration, verbose)
            iter_time = time.time() - iter_start_time
            
            if verbose:
                print(f"  iteration {iteration + 1} complete: {iter_time:.1f}s, value_change: {value_change:.6f}")
            
            # store training history
            self.training_history.append({
                'iteration': iteration + 1,
                'value_change': value_change,
                'time': iter_time
            })
            
            # check convergence
            if value_change < self.config.convergence_threshold:
                if verbose:
                    print("  early stopping: convergence achieved")
                break
        
        training_time = time.time() - start_time
        
        # mark as trained
        self.is_trained = True
        
        if verbose:
            print(f"✅ value function training complete: {training_time:.1f}s")
        
        return {
            'training_time': training_time,
            'iterations_completed': len(self.training_history),
            'final_value_change': self.training_history[-1]['value_change'] if self.training_history else float('inf'),
            'training_samples': len(self.training_states),
            'converged': value_change < self.config.convergence_threshold if self.training_history else False,
            'method': 'original'
        }
    
    def _generate_enhanced_training_data(self, verbose: bool = True) -> List[Tuple[Dict, float]]:
        """generate enhanced training data with better policy sampling"""
        if verbose:
            print("generating enhanced training data...")
        
        training_data = []
        
        # 1. grid-based policy sampling
        if verbose:
            print("  grid-based policy sampling...")
        grid_data = self._generate_grid_samples()
        training_data.extend(grid_data)
        
        # 2. trajectory-based sampling
        if verbose:
            print("  trajectory-based sampling...")
        trajectory_data = self._generate_trajectory_samples()
        training_data.extend(trajectory_data)
        
        # 3. best policy region sampling
        if verbose:
            print("  best policy region sampling...")
        best_region_data = self._generate_best_region_samples(training_data)
        training_data.extend(best_region_data)
        
        if verbose:
            print(f"  generated {len(training_data)} training samples")
        
        return training_data
    
    def _generate_grid_samples(self) -> List[Tuple[Dict, float]]:
        """generate samples on a grid of policies and states"""
        samples = []
        
        # create diverse initial states
        base_states = [
            {'supply': 10_000_000, 'usd_reserve': 100_000, 'price': 1.0, 'capacity': 1000, 
             'token_demand': 8_000_000, 'service_demand': 800, 'market_factor': 0.0},
            {'supply': 50_000_000, 'usd_reserve': 500_000, 'price': 1.0, 'capacity': 5000,
             'token_demand': 40_000_000, 'service_demand': 4000, 'market_factor': 0.0},
            {'supply': 100_000_000, 'usd_reserve': 1_000_000, 'price': 0.5, 'capacity': 10000,
             'token_demand': 80_000_000, 'service_demand': 8000, 'market_factor': 0.0}
        ]
        
        # create policy grid
        if self.controller_type == 'inflation policy':
            mint_rates = np.linspace(5_000, 200_000, 20)
            burn_shares = np.linspace(0.1, 0.9, 10)
            
            for state in base_states:
                for mint_rate in mint_rates:
                    for burn_share in burn_shares:
                        policy = {'mint_rate': mint_rate, 'burn_share': burn_share}
                        
                        try:
                            # evaluate policy
                            result = self.cost_function.evaluate_policy(state, policy, timesteps=30)
                            value = result['total_discounted_cost']
                            samples.append((state.copy(), value))
                            
                            # also add intermediate states from simulation
                            sim_result = simulate_policy(state, policy, timesteps=5, use_reactive=True)
                            if sim_result['success']:
                                for intermediate_state in sim_result['states'][1:]:
                                    intermediate_result = self.cost_function.evaluate_policy(
                                        intermediate_state, policy, timesteps=25)
                                    intermediate_value = intermediate_result['total_discounted_cost']
                                    samples.append((intermediate_state.copy(), intermediate_value))
                        except:
                            continue
        else:  # service pricing
            base_prices = np.linspace(0.5, 5.0, 10)
            elasticities = np.linspace(0.5, 2.5, 8)
            mint_rates = np.linspace(5_000, 100_000, 15)
            
            for state in base_states:
                for base_price in base_prices:
                    for elasticity in elasticities:
                        for mint_rate in mint_rates:
                            policy = {
                                'base_service_price': base_price,
                                'price_elasticity': elasticity,
                                'mint_rate': mint_rate
                            }
                            
                            try:
                                result = self.cost_function.evaluate_policy(state, policy, timesteps=30)
                                value = result['total_discounted_cost']
                                samples.append((state.copy(), value))
                            except:
                                continue
        
        return samples[:self.config.policy_grid_samples]
    
    def _generate_trajectory_samples(self) -> List[Tuple[Dict, float]]:
        """generate samples from random trajectories"""
        samples = []
        
        for _ in range(self.config.num_trajectories):
            # random initial state
            state = self._random_initial_state()
            
            # random policy
            policy = self._random_policy()
            
            try:
                # evaluate and simulate
                result = self.cost_function.evaluate_policy(state, policy, timesteps=30)
                value = result['total_discounted_cost']
                samples.append((state.copy(), value))
                
                # add trajectory states
                sim_result = simulate_policy(state, policy, timesteps=10, use_reactive=True)
                if sim_result['success']:
                    for i, traj_state in enumerate(sim_result['states'][1:]):
                        remaining_timesteps = max(1, 30 - i)
                        traj_result = self.cost_function.evaluate_policy(
                            traj_state, policy, timesteps=remaining_timesteps)
                        traj_value = traj_result['total_discounted_cost']
                        samples.append((traj_state.copy(), traj_value))
            except:
                continue
        
        return samples
    
    def _generate_best_region_samples(self, existing_data: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """generate extra samples around the best policies found so far"""
        if not existing_data:
            return []
        
        # find best policies
        best_samples = sorted(existing_data, key=lambda x: x[1])[:10]
        
        samples = []
        for best_state, best_value in best_samples:
            # add noise around best policies
            for _ in range(5):
                noisy_state = self._add_state_noise(best_state)
                
                # evaluate with similar policy
                policy = self._extract_policy_from_state(best_state)
                noisy_policy = self._add_policy_noise(policy)
                
                try:
                    result = self.cost_function.evaluate_policy(noisy_state, noisy_policy, timesteps=30)
                    value = result['total_discounted_cost']
                    samples.append((noisy_state.copy(), value))
                except:
                    continue
        
        return samples[:self.config.best_policy_samples]
    
    def _random_initial_state(self) -> Dict:
        """generate random initial state"""
        supply = np.random.uniform(5_000_000, 200_000_000)
        capacity = np.random.uniform(500, 20_000)
        utilization = np.random.uniform(0.3, 0.95)
        
        return {
            'supply': supply,
            'usd_reserve': supply * np.random.uniform(0.005, 0.02),
            'price': np.random.uniform(0.1, 5.0),
            'capacity': capacity,
            'token_demand': supply * np.random.uniform(0.6, 0.95),
            'service_demand': capacity * utilization,
            'market_factor': np.random.uniform(-0.3, 0.3)
        }
    
    def _random_policy(self) -> Dict:
        """generate random policy"""
        if self.controller_type == 'inflation policy':
            return {
                'mint_rate': np.random.uniform(1_000, 300_000),
                'burn_share': np.random.uniform(0.1, 0.9)
            }
        else:
            return {
                'base_service_price': np.random.uniform(0.1, 10.0),
                'price_elasticity': np.random.uniform(0.5, 3.0),
                'mint_rate': np.random.uniform(1_000, 100_000)
            }
    
    def _add_state_noise(self, state: Dict) -> Dict:
        """add noise to a state"""
        noisy_state = state.copy()
        noise_scale = 0.1
        
        for key in ['supply', 'usd_reserve', 'capacity', 'token_demand', 'service_demand']:
            if key in noisy_state:
                noise = np.random.normal(0, noise_scale * noisy_state[key])
                noisy_state[key] = max(0, noisy_state[key] + noise)
        
        return noisy_state
    
    def _add_policy_noise(self, policy: Dict) -> Dict:
        """add noise to a policy"""
        noisy_policy = policy.copy()
        
        if 'mint_rate' in policy:
            noise = np.random.normal(0, 0.1 * policy['mint_rate'])
            noisy_policy['mint_rate'] = max(1000, policy['mint_rate'] + noise)
        
        if 'burn_share' in policy:
            noise = np.random.normal(0, 0.05)
            noisy_policy['burn_share'] = np.clip(policy['burn_share'] + noise, 0.1, 0.9)
        
        if 'base_service_price' in policy:
            noise = np.random.normal(0, 0.1 * policy['base_service_price'])
            noisy_policy['base_service_price'] = max(0.1, policy['base_service_price'] + noise)
        
        if 'price_elasticity' in policy:
            noise = np.random.normal(0, 0.05)
            noisy_policy['price_elasticity'] = np.clip(policy['price_elasticity'] + noise, 0.1, 3.0)
        
        return noisy_policy
    
    def _extract_policy_from_state(self, state: Dict) -> Dict:
        """extract a reasonable policy from a state (heuristic)"""
        if self.controller_type == 'inflation policy':
            # heuristic: higher utilization -> higher mint rate
            utilization = state['service_demand'] / max(state['capacity'], 1)
            mint_rate = state['supply'] * (0.0002 + 0.001 * utilization)
            burn_share = 0.3 + 0.4 * utilization
            
            return {'mint_rate': mint_rate, 'burn_share': burn_share}
        else:
            return {
                'base_service_price': 1.0,
                'price_elasticity': 1.0,
                'mint_rate': state['supply'] * 0.0005
            }
    
    def get_optimal_policy(self, state: Dict[str, float]) -> Dict[str, Any]:
        """get optimal policy for given state using the selected method"""
        if self.training_mode == 'improved' and self.config.use_cross_entropy_method:
            return self._get_optimal_policy_cem(state)
        else:
            return self._get_optimal_policy_original(state)
    
    def _get_optimal_policy_cem(self, state: Dict) -> Dict[str, Any]:
        """use cross-entropy method to find optimal policy"""
        
        if self.controller_type == 'inflation policy':
            # initialize policy distribution
            mint_mean = state['supply'] * 0.001
            mint_std = state['supply'] * 0.0005
            burn_mean = 0.5
            burn_std = 0.2
            
            best_policy = None
            best_value = float('inf')
            
            for iteration in range(self.config.cem_iterations):
                # sample policies
                policies = []
                values = []
                
                for _ in range(100):
                    mint_rate = max(1000, np.random.normal(mint_mean, mint_std))
                    burn_share = np.clip(np.random.normal(burn_mean, burn_std), 0.1, 0.9)
                    
                    policy = {'mint_rate': mint_rate, 'burn_share': burn_share}
                    
                    # evaluate using cost function (fast approximation)
                    value = self._evaluate_policy_fast(state, policy)
                    
                    policies.append(policy)
                    values.append(value)
                
                # select elite policies
                elite_indices = np.argsort(values)[:int(len(values) * self.config.cem_elite_fraction)]
                elite_policies = [policies[i] for i in elite_indices]
                
                # update best policy
                if values[elite_indices[0]] < best_value:
                    best_value = values[elite_indices[0]]
                    best_policy = elite_policies[0]
                
                # update distribution
                elite_mints = [p['mint_rate'] for p in elite_policies]
                elite_burns = [p['burn_share'] for p in elite_policies]
                
                mint_mean = np.mean(elite_mints)
                mint_std = max(1000, np.std(elite_mints))  # ensure minimum std
                burn_mean = np.mean(elite_burns)
                burn_std = max(0.05, np.std(elite_burns))  # ensure minimum std
            
            return {
                'controls': best_policy,
                'expected_value': best_value,
                'method': 'cross_entropy_method'
            }
        
        else:  # service pricing
            # initialize policy distribution
            price_mean = 1.0
            price_std = 0.5
            elast_mean = 1.0
            elast_std = 0.3
            mint_mean = state['supply'] * 0.0005
            mint_std = state['supply'] * 0.0002
            
            best_policy = None
            best_value = float('inf')
            
            for iteration in range(self.config.cem_iterations):
                policies = []
                values = []
                
                for _ in range(100):
                    base_price = max(0.1, np.random.normal(price_mean, price_std))
                    elasticity = np.clip(np.random.normal(elast_mean, elast_std), 0.1, 3.0)
                    mint_rate = max(1000, np.random.normal(mint_mean, mint_std))
                    
                    policy = {
                        'base_service_price': base_price,
                        'price_elasticity': elasticity,
                        'mint_rate': mint_rate
                    }
                    
                    value = self._evaluate_policy_fast(state, policy)
                    
                    policies.append(policy)
                    values.append(value)
                
                # select elite policies
                elite_indices = np.argsort(values)[:int(len(values) * self.config.cem_elite_fraction)]
                elite_policies = [policies[i] for i in elite_indices]
                
                # update best policy
                if values[elite_indices[0]] < best_value:
                    best_value = values[elite_indices[0]]
                    best_policy = elite_policies[0]
                
                # update distributions
                elite_prices = [p['base_service_price'] for p in elite_policies]
                elite_elasts = [p['price_elasticity'] for p in elite_policies]
                elite_mints = [p['mint_rate'] for p in elite_policies]
                
                price_mean = np.mean(elite_prices)
                price_std = max(0.1, np.std(elite_prices))
                elast_mean = np.mean(elite_elasts)
                elast_std = max(0.05, np.std(elite_elasts))
                mint_mean = np.mean(elite_mints)
                mint_std = max(1000, np.std(elite_mints))
            
            return {
                'controls': best_policy,
                'expected_value': best_value,
                'method': 'cross_entropy_method'
            }
    
    def _get_optimal_policy_original(self, state: Dict) -> Dict[str, Any]:
        """original bellman optimization method"""
        try:
            optimal_controls, expected_value = self.solve_bellman_equation(state, self.value_network)
            return {
                'controls': optimal_controls,
                'expected_value': expected_value,
                'method': 'bellman_optimization'
            }
        except Exception as e:
            # fallback to heuristic policy
            fallback_policy = self._extract_policy_from_state(state)
            fallback_value = self._evaluate_policy_fast(state, fallback_policy)
            
            return {
                'controls': fallback_policy,
                'expected_value': fallback_value,
                'method': 'fallback_heuristic',
                'error': str(e)
            }
    
    def _evaluate_policy_fast(self, state: Dict, policy: Dict) -> float:
        """fast policy evaluation for optimization"""
        try:
            # use shorter horizon for faster evaluation during optimization
            result = self.cost_function.evaluate_policy(state, policy, timesteps=10)
            return result['total_discounted_cost']
        except:
            return 10.0  # penalty for invalid policies
    
    def _train_neural_network(self, states: List[Dict], values: List[float], verbose: bool = True) -> Dict[str, Any]:
        """train the neural network on the collected data"""
        import torch
        
        if verbose:
            print(f"training neural network on {len(states)} samples...")
        
        # convert to tensors
        state_tensors = torch.stack([self._state_to_tensor(s) for s in states])
        value_tensors = torch.FloatTensor(values)
        
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        
        for epoch in range(self.config.train_epochs):
            # shuffle data
            indices = torch.randperm(len(state_tensors))
            epoch_loss = 0
            num_batches = 0
            
            # batch training
            for i in range(0, len(state_tensors), self.config.batch_size):
                batch_indices = indices[i:i + self.config.batch_size]
                batch_states = state_tensors[batch_indices]
                batch_values = value_tensors[batch_indices]
                
                # forward pass
                predictions = self.network(batch_states).squeeze()
                loss = self.criterion(predictions, batch_values)
                
                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and epoch % 20 == 0:
                print(f"  epoch {epoch}: loss {avg_loss:.6f}")
            
            # early stopping
            if patience_counter >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch}")
                break
        
        return {'final_loss': best_loss}
    
    def _state_to_tensor(self, state: Dict):
        """convert state dict to normalized tensor"""
        import torch
        
        features = self._state_to_features(state)
        
        # normalize if we have statistics
        if self.state_mean is not None:
            features = (features - self.state_mean) / (self.state_std + 1e-8)
        
        return torch.FloatTensor(features)
    
    def _state_to_features(self, state: Dict) -> np.ndarray:
        """convert state dict to feature vector"""
        return np.array([
            state.get('supply', 0),
            state.get('usd_reserve', 0), 
            state.get('price', 0),
            state.get('capacity', 0),
            state.get('token_demand', 0),
            state.get('service_demand', 0),
            state.get('market_factor', 0.0)
        ], dtype=np.float32)
    
    def _generate_training_trajectories(self, verbose: bool = True):
        """generate training trajectories using original method"""
        if verbose:
            print("generating training trajectories...")
        
        # generate diverse initial states
        initial_states = self.sample_initial_states(self.config.num_trajectories)
        
        # clear previous training data
        self.training_states = []
        self.training_values = []
        
        for i, initial_state in enumerate(initial_states):
            if verbose and i % 50 == 0:
                print(f"  trajectory {i+1}/{len(initial_states)}")
            
            try:
                # simulate trajectory with random policy
                policy = self._random_policy()
                sim_result = simulate_policy(initial_state, policy, 
                                           timesteps=self.config.trajectory_length, 
                                           use_reactive=True)
                
                if sim_result['success']:
                    # add all states from trajectory
                    for state in sim_result['states']:
                        # ensure market_factor exists
                        if 'market_factor' not in state:
                            state['market_factor'] = initial_state.get('market_factor', 0.0)
                        
                        state_features = self._state_to_features(state)
                        self.training_states.append(state_features)
                        # initialize with zero values (will be updated in FVI)
                        self.training_values.append(0.0)
            except Exception as e:
                if verbose:
                    print(f"    warning: trajectory {i} failed: {e}")
                continue
        
        # check if we have any training data before converting to numpy
        if len(self.training_states) == 0:
            if verbose:
                print("warning: no valid training trajectories generated, creating fallback data")
            
            # create fallback training data from initial states
            for state in initial_states[:10]:  # use first 10 states
                if 'market_factor' not in state:
                    state['market_factor'] = 0.0
                
                state_features = self._state_to_features(state)
                self.training_states.append(state_features)
                self.training_values.append(1.0)  # dummy value
        
        # convert to numpy arrays only after all data is collected
        self.training_states = np.array(self.training_states)
        self.training_values = np.array(self.training_values)
        
        # initialize scaler for original method
        if self.training_mode == 'original':
            self.state_scaler = StandardScaler()
            if len(self.training_states) > 0:
                self.state_scaler.fit(self.training_states)
        
        if verbose:
            print(f"generated {len(self.training_states)} training samples")
    
    def _perform_fvi_iteration(self, iteration: int, verbose: bool = True) -> float:
        """perform one fitted value iteration step"""
        if verbose:
            print(f"  computing bellman targets...")
        
        # compute new values using bellman optimization
        new_values = np.zeros(len(self.training_states))
        successful_updates = 0
        
        # convert training states back to dict format for bellman computation
        state_dicts = []
        for state_features in self.training_states:
            state_dict = {
                'supply': state_features[0],
                'usd_reserve': state_features[1],
                'price': state_features[2],
                'capacity': state_features[3],
                'token_demand': state_features[4],
                'service_demand': state_features[5],
                'market_factor': state_features[6]
            }
            state_dicts.append(state_dict)
        
        for i, state_dict in enumerate(state_dicts):
            if i % 1000 == 0 and verbose:
                print(f"    computing targets: {i+1}/{len(state_dicts)}")
            
            try:
                # solve bellman equation for this state
                if self.training_mode == 'original':
                    _, bellman_value = self.solve_bellman_equation(state_dict, self.value_network)
                else:
                    # for improved mode, use fast evaluation
                    policy = self._extract_policy_from_state(state_dict)
                    bellman_value = self._evaluate_policy_fast(state_dict, policy)
                
                new_values[i] = bellman_value
                successful_updates += 1
            except Exception as e:
                # keep old value if optimization fails
                new_values[i] = self.training_values[i]
        
        if verbose:
            print(f"    successful bellman updates: {successful_updates}/{len(self.training_states)}")
            print(f"  training value function...")
        
        # train the value function on new targets
        if self.training_mode == 'original':
            # use MLPRegressor for original mode
            if not hasattr(self, 'value_network'):
                from sklearn.neural_network import MLPRegressor
                self.value_network = MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    max_iter=200,
                    random_state=42,
                    warm_start=True
                )
            
            normalized_states = self.state_scaler.transform(self.training_states)
            self.value_network.fit(normalized_states, new_values)
        else:
            # use pytorch network for improved mode
            self._train_neural_network_step(self.training_states, new_values)
        
        # compute value change for convergence check
        value_change = np.mean(np.abs(new_values - self.training_values))
        self.training_values = new_values.copy()
        
        return value_change
    
    def _train_neural_network_step(self, states: np.ndarray, values: np.ndarray):
        """single training step for pytorch network during FVI"""
        if self.training_mode != 'improved':
            return
        
        import torch
        
        # convert to tensors
        state_dicts = []
        for state_features in states:
            state_dict = {
                'supply': state_features[0], 'usd_reserve': state_features[1],
                'price': state_features[2], 'capacity': state_features[3],
                'token_demand': state_features[4], 'service_demand': state_features[5],
                'market_factor': state_features[6]
            }
            state_dicts.append(state_dict)
        
        state_tensors = torch.stack([self._state_to_tensor(s) for s in state_dicts])
        value_tensors = torch.FloatTensor(values)
        
        # single epoch of training
        for i in range(0, len(state_tensors), self.config.batch_size):
            batch_states = state_tensors[i:i + self.config.batch_size]
            batch_values = value_tensors[i:i + self.config.batch_size]
            
            predictions = self.network(batch_states).squeeze()
            loss = self.criterion(predictions, batch_values)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


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