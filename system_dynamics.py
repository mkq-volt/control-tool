"""
system dynamics for depin protocol optimization

implements network-aware dynamics where service demand grows with 
network capacity and token value, providing better utilization equilibrium
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SystemParams:
    """network-aware parameters for the depin system dynamics"""
    
    # demand evolution with network effects
    token_demand_vol: float = 0.05          # volatility of token demand
    market_vol: float = 0.03                # market factor volatility
    
    # service demand coupled to network value
    service_demand_vol: float = 0.05        # base volatility of service demand
    service_network_effect: float = 0.02    # service demand grows with capacity
    service_value_effect: float = 0.015     # service demand grows with token price
    
    # price formation
    base_utility_value: float = 1.0         # base token value from utility
    speculation_weight: float = 0.3         # how much speculation affects price
    utilization_price_boost: float = 0.2    # price premium when highly utilized
    market_sensitivity: float = 0.5         # sensitivity to market conditions
    
    # service economics
    service_price: float = 1.0              # fixed price per service unit
    
    # balanced node economics
    node_cost_ratio: float = 0.012          # node cost as fraction of expected rewards
    capacity_response_speed: float = 0.008  # how quickly capacity responds to profitability
    max_capacity_growth: float = 0.02       # maximum capacity growth rate per timestep
    capacity_depreciation: float = 0.005    # natural capacity decay rate
    
    def get_node_operating_cost(self, mint_rate: float, token_price: float, 
                               capacity: float) -> float:
        """calculate scale-appropriate node operating cost"""
        # cost scales with expected rewards per node
        expected_reward_per_node = (mint_rate * token_price) / max(capacity, 1)
        return expected_reward_per_node * self.node_cost_ratio
    
    def validate_and_scale_inputs(self, initial_state: Dict[str, float], 
                                 mint_rate: float) -> Dict[str, Any]:
        """validate inputs and suggest scaling adjustments for stability"""
        supply = initial_state['supply']
        capacity = initial_state['capacity'] 
        service_demand = initial_state['service_demand']
        price = initial_state['price']
        
        warnings = []
        suggestions = {}
        
        # check mint rate relative to supply (should be 0.05-0.2% per timestep)
        mint_rate_ratio = mint_rate / supply
        if mint_rate_ratio > 0.002:  # > 0.2% 
            warnings.append(f"mint rate ({mint_rate:,.0f}) very high relative to supply")
            suggestions['mint_rate'] = supply * 0.001  # suggest 0.1%
        elif mint_rate_ratio < 0.0002:  # < 0.02%
            warnings.append(f"mint rate ({mint_rate:,.0f}) very low relative to supply") 
            suggestions['mint_rate'] = supply * 0.0005  # suggest 0.05%
            
        # check utilization (service_demand / capacity should be 0.6-0.9)
        utilization = service_demand / max(capacity, 1)
        if utilization > 1.2:
            warnings.append(f"initial utilization ({utilization:.2f}) too high")
            suggestions['capacity'] = service_demand / 0.8
        elif utilization < 0.3:
            warnings.append(f"initial utilization ({utilization:.2f}) too low")  
            suggestions['service_demand'] = capacity * 0.7
            
        # check price reasonableness (should be 0.1-10.0 range typically)
        if price > 100:
            warnings.append(f"token price (${price}) very high - consider normalizing")
        elif price < 0.01:
            warnings.append(f"token price (${price}) very low - consider scaling up")
            
        return {
            'warnings': warnings,
            'suggestions': suggestions,
            'stability_score': len(warnings)  # 0 = stable, higher = more issues
        }
    
    @classmethod
    def suggest_stable_controls(cls, initial_state: Dict[str, float]) -> Dict[str, float]:
        """suggest stable control policy based on system scale"""
        supply = initial_state['supply']
        capacity = initial_state['capacity']
        utilization = initial_state['service_demand'] / max(capacity, 1)
        
        # mint rate: conservative 0.03-0.06% of supply per timestep
        suggested_mint = supply * 0.0004  # 0.04% default
        
        # burn share: adaptive based on utilization
        if utilization > 0.9:
            suggested_burn = 0.7  # more aggressive deflation
        elif utilization < 0.5:
            suggested_burn = 0.3  # moderate deflation
        else:
            suggested_burn = 0.5  # balanced
            
        return {
            'mint_rate': suggested_mint,
            'burn_share': suggested_burn
        }


class SystemState:
    """manages depin protocol state evolution with network-aware dynamics"""
    
    def __init__(self, initial_state: Dict[str, float], params: SystemParams):
        # preserve initial state for network effect calculations
        self.initial_state = initial_state.copy()
        
        # state variables (endogenous)
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
        """deterministic utilization calculation"""
        return min(1.0, service_demand / max(capacity, 1e-6))
    
    def _sample_exogenous(self) -> Dict[str, float]:
        """sample next period's exogenous variables with network effects"""
        
        # token demand random walk (unchanged)
        token_shock = np.random.normal(0, self.params.token_demand_vol)
        new_token_demand = max(0.1, self.token_demand[-1] + token_shock)
        
        # service demand with network effects
        current_service_demand = self.service_demand[-1]
        current_capacity = self.capacity[-1]
        current_price = self.price[-1]
        
        # network effect: more capacity attracts more demand
        network_growth = self.params.service_network_effect * (
            current_capacity / self.initial_state['capacity'] - 1
        )
        
        # value effect: higher token price increases service attractiveness  
        value_growth = self.params.service_value_effect * (
            current_price / self.initial_state['price'] - 1
        )
        
        # combined growth + random shock
        service_shock = np.random.normal(0, self.params.service_demand_vol)
        growth_rate = network_growth + value_growth + service_shock
        new_service_demand = max(0.0, current_service_demand * (1 + growth_rate))
        
        # market factor mean-reverting
        market_shock = np.random.normal(0, self.params.market_vol)
        new_market_factor = np.clip(
            self.market_factor[-1] * 0.95 + market_shock,  # mean reversion to 0
            -0.5, 0.5
        )
        
        return {
            'token_demand': new_token_demand,
            'service_demand': new_service_demand, 
            'market_factor': new_market_factor
        }
    
    def _update_price(self, new_supply: float, new_token_demand: float, 
                     new_utilization: float, new_market_factor: float) -> float:
        """realistic price formation combining utility and speculation"""
        # base utility value
        utility_component = self.params.base_utility_value
        
        # speculative component (demand/supply pressure)
        speculation_component = (new_token_demand / new_supply) * self.params.speculation_weight
        
        # utilization boost (higher utilization = higher token value)
        utilization_boost = new_utilization * self.params.utilization_price_boost
        
        # combine components
        base_price = utility_component + speculation_component + utilization_boost
        
        # apply market factor
        market_adjusted_price = base_price * (1 + new_market_factor * self.params.market_sensitivity)
        
        return max(0.01, market_adjusted_price)  # minimum price floor
    
    def _update_capacity(self, current_capacity: float, mint_rate: float, 
                        current_price: float) -> float:
        """capacity evolution with balanced growth and depreciation"""
        if current_capacity <= 0:
            return 1.0  # minimum capacity
            
        # node economics
        emissions_per_node = mint_rate / current_capacity if current_capacity > 0 else 0
        node_revenue = emissions_per_node * current_price
        node_cost = self.params.get_node_operating_cost(mint_rate, current_price, current_capacity)
        node_profit = node_revenue - node_cost
        
        # profitability signal (>0 means profitable)
        profitability_ratio = node_profit / node_cost if node_cost > 0 else 0
        
        # capacity growth response with sigmoid to bound growth
        growth_signal = np.tanh(profitability_ratio * self.params.capacity_response_speed)
        capacity_growth_rate = growth_signal * self.params.max_capacity_growth
        
        # natural capacity depreciation
        depreciation_rate = self.params.capacity_depreciation
        
        # net capacity change
        net_growth_rate = capacity_growth_rate - depreciation_rate
        new_capacity = current_capacity * (1 + net_growth_rate)
        
        return max(1.0, new_capacity)  # minimum capacity
    
    def step(self, control_actions: Dict[str, float]) -> Dict[str, float]:
        """advance system by one timestep"""
        
        # sample exogenous variables
        exogenous = self._sample_exogenous()
        
        # extract control actions
        mint_rate = max(0, control_actions.get('mint_rate', 0))
        burn_share = np.clip(control_actions.get('burn_share', 0), 0, 1)  # must be 0-1
        
        # service revenue flow
        service_revenue_usd = exogenous['service_demand'] * self.params.service_price
        service_revenue_tokens = service_revenue_usd / self.price[-1] if self.price[-1] > 0 else 0
        
        # split service revenue between burn and reserves
        burn_tokens = service_revenue_tokens * burn_share
        reserve_tokens = service_revenue_tokens * (1 - burn_share)
        
        # update token supply (mint new tokens, burn service revenue tokens)
        new_supply = self.supply[-1] + mint_rate - burn_tokens
        new_supply = max(1000, new_supply)  # minimum supply for stability
        
        # update reserves (receive reserve portion of service revenue)
        new_reserves = self.usd_reserve[-1] + reserve_tokens * self.price[-1]
        new_reserves = max(0, new_reserves)  # can't go negative
        
        # update capacity based on node economics
        new_capacity = self._update_capacity(
            self.capacity[-1], 
            mint_rate, 
            self.price[-1]
        )
        
        # update utilization
        new_utilization = self._calculate_utilization(
            exogenous['service_demand'], 
            new_capacity
        )
        
        # update price
        new_price = self._update_price(
            new_supply,
            exogenous['token_demand'],
            new_utilization, 
            exogenous['market_factor']
        )
        
        # store new state
        self.supply.append(new_supply)
        self.usd_reserve.append(new_reserves)
        self.price.append(new_price)
        self.capacity.append(new_capacity)
        self.utilization.append(new_utilization)
        
        self.token_demand.append(exogenous['token_demand'])
        self.service_demand.append(exogenous['service_demand'])
        self.market_factor.append(exogenous['market_factor'])
        
        self.timestep += 1
        
        return {
            'supply': new_supply,
            'usd_reserve': new_reserves,
            'price': new_price,
            'capacity': new_capacity,
            'utilization': new_utilization,
            'token_demand': exogenous['token_demand'],
            'service_demand': exogenous['service_demand'],
            'market_factor': exogenous['market_factor'],
            'burn_tokens': burn_tokens,
            'mint_rate': mint_rate,
            'service_revenue_usd': service_revenue_usd
        }
    
    def get_current_state(self) -> Dict[str, float]:
        """return current state as dictionary"""
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
        """return full state history"""
        return {
            'supply': self.supply.copy(),
            'usd_reserve': self.usd_reserve.copy(),
            'price': self.price.copy(),
            'capacity': self.capacity.copy(),
            'utilization': self.utilization.copy(),
            'token_demand': self.token_demand.copy(),
            'service_demand': self.service_demand.copy(),
            'market_factor': self.market_factor.copy(),
            'timesteps': list(range(len(self.supply)))
        }


# example usage and testing
if __name__ == "__main__":
    # initialize system with network-aware defaults
    params = SystemParams()
    
    initial_state = {
        'supply': 10_000_000,
        'usd_reserve': 100_000,
        'price': 1.0,
        'capacity': 1000,
        'token_demand': 8_000_000,
        'service_demand': 800,
        'market_factor': 0.0
    }
    
    # get suggested stable controls
    suggested_controls = SystemParams.suggest_stable_controls(initial_state)
    print(f"suggested controls: mint_rate={suggested_controls['mint_rate']:,.0f}, burn_share={suggested_controls['burn_share']:.1f}")
    
    # validate inputs
    validation = params.validate_and_scale_inputs(initial_state, suggested_controls['mint_rate'])
    if validation['warnings']:
        print("warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    else:
        print("✅ system configuration looks stable")
    
    # run brief demonstration
    system = SystemState(initial_state, params)
    print(f"\ninitial: price=${system.price[-1]:.2f}, capacity={system.capacity[-1]:.0f}, util={system.utilization[-1]:.2f}")
    
    for t in range(3):
        result = system.step(suggested_controls)
        print(f"step {t+1}: price=${result['price']:.2f}, capacity={result['capacity']:.0f}, util={result['utilization']:.2f}")
    
    print("\n✅ network-aware system dynamics ready for use") 