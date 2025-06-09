"""
system dynamics for depin protocol optimization

implements network-aware dynamics where service demand grows with 
network capacity and token value

includes reactive capacity dynamics that eliminate binary threshold cycling
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SystemParams:
    """network-aware parameters for the depin system dynamics"""
    
    # demand evolution with network effects
    token_demand_vol: float = 0.05
    market_vol: float = 0.03
    
    # service demand coupled to network value
    service_demand_vol: float = 0.05
    service_network_effect: float = 0.02
    service_value_effect: float = 0.015
    
    # price formation
    base_utility_value: float = 1.0
    speculation_weight: float = 0.3
    utilization_price_boost: float = 0.2
    market_sensitivity: float = 0.5
    
    # service economics
    service_price: float = 1.0
    
    # balanced node economics
    node_cost_ratio: float = 0.012
    capacity_response_speed: float = 0.008
    max_capacity_growth: float = 0.02
    capacity_depreciation: float = 0.005
    
    def get_node_operating_cost(self, mint_rate: float, token_price: float, 
                               capacity: float) -> float:
        """calculate scale-appropriate node operating cost"""
        expected_reward_per_node = (mint_rate * token_price) / max(capacity, 1)
        return expected_reward_per_node * self.node_cost_ratio
    
    def validate_and_scale_inputs(self, initial_state: Dict[str, float], 
                                 mint_rate: float) -> Dict[str, Any]:
        supply = initial_state['supply']
        capacity = initial_state['capacity'] 
        service_demand = initial_state['service_demand']
        price = initial_state['price']
        
        warnings = []
        suggestions = {}
        
        # mint rate should be 0.05-0.2% per timestep
        mint_rate_ratio = mint_rate / supply
        if mint_rate_ratio > 0.002:
            warnings.append(f"mint rate ({mint_rate:,.0f}) very high relative to supply")
            suggestions['mint_rate'] = supply * 0.001
        elif mint_rate_ratio < 0.0002:
            warnings.append(f"mint rate ({mint_rate:,.0f}) very low relative to supply") 
            suggestions['mint_rate'] = supply * 0.0005
            
        # utilization should be 0.6-0.9
        utilization = service_demand / max(capacity, 1)
        if utilization > 1.2:
            warnings.append(f"initial utilization ({utilization:.2f}) too high")
            suggestions['capacity'] = service_demand / 0.8
        elif utilization < 0.3:
            warnings.append(f"initial utilization ({utilization:.2f}) too low")  
            suggestions['service_demand'] = capacity * 0.7
            
        # price reasonableness
        if price > 100:
            warnings.append(f"token price (${price}) very high - consider normalizing")
        elif price < 0.01:
            warnings.append(f"token price (${price}) very low - consider scaling up")
            
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'stability_score': len(warnings)
        }
    
    @classmethod
    def suggest_stable_controls(cls, initial_state: Dict[str, float]) -> Dict[str, float]:
        supply = initial_state['supply']
        capacity = initial_state['capacity']
        utilization = initial_state['service_demand'] / max(capacity, 1)
        
        # mint rate: 0.03-0.06% of supply per timestep
        suggested_mint = supply * 0.0004
        
        # burn share: adaptive based on utilization
        if utilization > 0.9:
            suggested_burn = 0.7
        elif utilization < 0.5:
            suggested_burn = 0.3
        else:
            suggested_burn = 0.5
            
        return {
            'mint_rate': suggested_mint,
            'burn_share': suggested_burn
        }


@dataclass 
class ReactiveCapacityParams(SystemParams):
    """parameters for reactive capacity dynamics with smooth responses"""
    
    # node economics
    base_node_cost: float = 8.0
    cost_scale_factor: float = 0.0001
    
    # utilization response
    utilization_target: float = 0.8
    utilization_sensitivity: float = 3.0
    
    # profit margin response
    profit_margin_sensitivity: float = 1.5
    baseline_profit_margin: float = 0.25
    
    # market equilibrium
    capacity_elasticity: float = 0.5
    growth_momentum: float = 0.1
    
    # dynamic depreciation
    base_depreciation: float = 0.005
    utilization_depreciation_factor: float = 0.02


class SystemState:
    """manages depin protocol state evolution with network-aware dynamics"""
    
    def __init__(self, initial_state: Dict[str, float], params: SystemParams):
        self.initial_state = initial_state.copy()
        
        # state variables
        self.supply = [initial_state['supply']]
        self.usd_reserve = [initial_state['usd_reserve']]  
        self.price = [initial_state['price']]
        self.capacity = [initial_state['capacity']]
        
        # exogenous variables
        self.token_demand = [initial_state['token_demand']]
        self.service_demand = [initial_state['service_demand']]
        self.market_factor = [initial_state['market_factor']]
        
        # derived state
        self.utilization = [self._calculate_utilization(
            initial_state['service_demand'], 
            initial_state['capacity']
        )]
        
        self.params = params
        self.timestep = 0
        
    def _calculate_utilization(self, service_demand: float, capacity: float) -> float:
        return min(1.0, service_demand / max(capacity, 1e-6))
    
    def _sample_exogenous(self) -> Dict[str, float]:
        """sample next period's exogenous variables with network effects"""
        
        # token demand random walk
        token_shock = np.random.normal(0, self.params.token_demand_vol)
        new_token_demand = max(0.1, self.token_demand[-1] + token_shock)
        
        # service demand with network effects
        current_service_demand = self.service_demand[-1]
        current_capacity = self.capacity[-1]
        current_price = self.price[-1]
        
        # network effect: C(t)/C(0) - 1
        network_growth = self.params.service_network_effect * (
            current_capacity / self.initial_state['capacity'] - 1
        )
        
        # value effect: P(t)/P(0) - 1
        value_growth = self.params.service_value_effect * (
            current_price / self.initial_state['price'] - 1
        )
        
        service_shock = np.random.normal(0, self.params.service_demand_vol)
        growth_rate = network_growth + value_growth + service_shock
        new_service_demand = max(0.0, current_service_demand * (1 + growth_rate))
        
        # market factor mean-reverting
        market_shock = np.random.normal(0, self.params.market_vol)
        new_market_factor = self.market_factor[-1] * 0.9 + market_shock
        
        return {
            'token_demand': new_token_demand,
            'service_demand': new_service_demand,
            'market_factor': new_market_factor
        }
    
    def _update_supply(self, mint_rate: float, burn_tokens: float) -> float:
        """new supply: S(t+1) = S(t) + mint_rate - burn_tokens"""
        return max(0, self.supply[-1] + mint_rate - burn_tokens)
    
    def _update_usd_reserve(self, service_revenue: float, burn_tokens: float, 
                           burn_share: float) -> float:
        """new reserve: R(t+1) = R(t) + service_revenue + burn_tokens × P(t) × (1 - burn_share)"""
        burned_value = burn_tokens * self.price[-1] * (1 - burn_share)
        return max(0, self.usd_reserve[-1] + service_revenue + burned_value)
    
    def _update_price(self, new_supply: float, new_token_demand: float, 
                     new_utilization: float, new_market_factor: float) -> float:
        """price formation based on utility value and market factors"""
        
        # utility-based price
        if new_supply > 0:
            utility_price = self.params.base_utility_value * (new_token_demand / new_supply)
        else:
            utility_price = self.params.base_utility_value
        
        # utilization premium
        utilization_premium = 1 + self.params.utilization_price_boost * new_utilization
        
        # market factor adjustment
        market_adjustment = 1 + self.params.market_sensitivity * new_market_factor
        
        # speculation weight
        current_price = self.price[-1]
        speculative_price = current_price * (1 + new_market_factor)
        
        # weighted combination
        new_price = (
            (1 - self.params.speculation_weight) * utility_price * utilization_premium * market_adjustment +
            self.params.speculation_weight * speculative_price
        )
        
        return max(0.01, new_price)
    
    def _update_capacity(self, current_capacity: float, mint_rate: float, 
                        current_price: float) -> float:
        """simple capacity evolution based on node economics"""
        
        # calculate profitability
        node_cost = self.params.get_node_operating_cost(mint_rate, current_price, current_capacity)
        expected_revenue_per_node = (mint_rate * current_price) / max(current_capacity, 1)
        
        # growth rate based on profitability
        if expected_revenue_per_node > node_cost:
            profitability_ratio = expected_revenue_per_node / node_cost
            growth_rate = min(
                self.params.max_capacity_growth,
                self.params.capacity_response_speed * (profitability_ratio - 1)
            )
        else:
            growth_rate = -self.params.capacity_depreciation
        
        new_capacity = current_capacity * (1 + growth_rate)
        return max(1, new_capacity)
    
    def step(self, control_actions: Dict[str, float]) -> Dict[str, float]:
        """advance system by one timestep"""
        
        # extract controls
        mint_rate = control_actions.get('mint_rate', 0)
        burn_share = control_actions.get('burn_share', 0)
        
        # sample exogenous variables
        exogenous = self._sample_exogenous()
        
        # compute service revenue
        service_revenue = min(
            exogenous['service_demand'], 
            self.capacity[-1]
        ) * self.params.service_price
        
        # compute burned tokens
        burn_tokens = service_revenue * burn_share / max(self.price[-1], 0.01)
        
        # update endogenous state
        new_supply = self._update_supply(mint_rate, burn_tokens)
        new_utilization = self._calculate_utilization(
            exogenous['service_demand'], 
            self.capacity[-1]
        )
        new_price = self._update_price(
            new_supply, 
            exogenous['token_demand'], 
            new_utilization, 
            exogenous['market_factor']
        )
        new_usd_reserve = self._update_usd_reserve(
            service_revenue, 
            burn_tokens, 
            burn_share
        )
        new_capacity = self._update_capacity(
            self.capacity[-1], 
            mint_rate, 
            new_price
        )
        new_utilization_next = self._calculate_utilization(
            exogenous['service_demand'], 
            new_capacity
        )
        
        # record state
        self.supply.append(new_supply)
        self.usd_reserve.append(new_usd_reserve)
        self.price.append(new_price)
        self.capacity.append(new_capacity)
        self.token_demand.append(exogenous['token_demand'])
        self.service_demand.append(exogenous['service_demand'])
        self.market_factor.append(exogenous['market_factor'])
        self.utilization.append(new_utilization_next)
        
        self.timestep += 1
        
        return {
            'supply': new_supply,
            'usd_reserve': new_usd_reserve,
            'price': new_price,
            'capacity': new_capacity,
            'utilization': new_utilization_next,
            'token_demand': exogenous['token_demand'],
            'service_demand': exogenous['service_demand'],
            'market_factor': exogenous['market_factor'],
            'burn_tokens': burn_tokens,
            'service_revenue_usd': service_revenue
        }
    
    def get_current_state(self) -> Dict[str, float]:
        return {
            'supply': self.supply[-1],
            'usd_reserve': self.usd_reserve[-1],
            'price': self.price[-1],
            'capacity': self.capacity[-1],
            'utilization': self.utilization[-1],
            'token_demand': self.token_demand[-1],
            'service_demand': self.service_demand[-1],
            'market_factor': self.market_factor[-1]
        }
    
    def get_history(self) -> Dict[str, list]:
        return {
            'supply': self.supply,
            'usd_reserve': self.usd_reserve,
            'price': self.price,
            'capacity': self.capacity,
            'utilization': self.utilization,
            'token_demand': self.token_demand,
            'service_demand': self.service_demand,
            'market_factor': self.market_factor
        }


class ReactiveSystemState(SystemState):
    """system with reactive capacity dynamics for smooth responses"""
    
    def __init__(self, initial_state: Dict[str, float], params: ReactiveCapacityParams):
        super().__init__(initial_state, params)
    
    def _update_capacity(self, current_capacity: float, mint_rate: float, 
                        current_price: float) -> float:
        """reactive capacity with smooth utilization and profit-based responses"""
        
        current_utilization = self.utilization[-1]
        
        # utilization response: (u - u_target) × sensitivity
        utilization_gap = current_utilization - self.params.utilization_target
        utilization_response = self.params.utilization_sensitivity * utilization_gap
        
        # profit response based on margin calculation
        num_nodes = max(current_capacity, 1)
        revenue_per_node = (mint_rate * current_price) / num_nodes
        
        node_cost = self.params.base_node_cost * (1 + self.params.cost_scale_factor * num_nodes)
        
        if revenue_per_node > 0:
            profit_margin = (revenue_per_node - node_cost) / revenue_per_node
        else:
            profit_margin = -1
        
        # profit margin response
        margin_gap = profit_margin - self.params.baseline_profit_margin
        profit_response = self.params.profit_margin_sensitivity * margin_gap
        
        # capacity elasticity: revenue decreases as capacity increases
        capacity_ratio = current_capacity / self.initial_state['capacity']
        revenue_adjustment = max(0.1, 1 - self.params.capacity_elasticity * (capacity_ratio - 1))
        
        # combined growth rate
        base_growth = utilization_response + profit_response * revenue_adjustment
        
        # apply momentum smoothing
        if hasattr(self, 'last_growth_rate'):
            smoothed_growth = (
                self.params.growth_momentum * self.last_growth_rate +
                (1 - self.params.growth_momentum) * base_growth
            )
        else:
            smoothed_growth = base_growth
        
        self.last_growth_rate = smoothed_growth
        
        # dynamic depreciation
        depreciation = self.params.base_depreciation
        if current_utilization < 0.5:
            depreciation += self.params.utilization_depreciation_factor * (0.5 - current_utilization)
        
        # final growth rate
        final_growth = smoothed_growth - depreciation
        
        # apply growth
        new_capacity = current_capacity * (1 + final_growth)
        return max(1, new_capacity)


if __name__ == "__main__":
    # test state
    test_state = {
        'supply': 50_000_000,
        'usd_reserve': 500_000,
        'price': 1.0,
        'capacity': 5000,
        'token_demand': 40_000_000,
        'service_demand': 4000,
        'market_factor': 0.0
    }
    
    # test controls
    controls = {'mint_rate': 50000, 'burn_share': 0.5}
    
    # test both systems
    for system_type, use_reactive in [("original", False), ("reactive", True)]:
        if use_reactive:
            params = ReactiveCapacityParams()
            system = ReactiveSystemState(test_state, params)
        else:
            params = SystemParams()
            system = SystemState(test_state, params)
        
        # simulate a few steps
        for _ in range(5):
            system.step(controls)
        
        final_state = system.get_current_state()
        print(f"✅ {system_type} system: final_capacity={final_state['capacity']:.0f}, "
              f"final_utilization={final_state['utilization']:.2f}") 