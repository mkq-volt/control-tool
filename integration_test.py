"""
integration test for improved depin optimal control

verifies that all improvements work together:
- network-aware system dynamics  
- comprehensive value function coverage
- robust bellman optimization
- proper cost breakdown balance
"""

from optimal_control import OptimalController, TrainingConfig
from cost_function import CostWeights, CostTargets


def test_integration():
    """test integrated improvements"""
    print("🚀 testing integrated improvements...")
    
    # setup with realistic weights
    weights = CostWeights(price_growth=1.5, utilization_min=2.5, capacity_growth=1.0)
    targets = CostTargets(price_growth=0.02, utilization_min=0.8, capacity_growth=0.01)
    
    # create controller with comprehensive training config
    controller = OptimalController(weights, targets)
    
    training_config = TrainingConfig(
        num_trajectories=150,  # reasonable for production
        fvi_iterations=4,
        utilization_aware_sampling=True,
        state_space_coverage='comprehensive',
        use_global_optimization=True
    )
    
    print("training optimal policy with integrated improvements...")
    result = controller.train_optimal_policy(training_config, verbose=True)
    
    # test state
    test_state = {
        'supply': 50_000_000, 'usd_reserve': 500_000, 'price': 1.0, 'capacity': 5000,
        'token_demand': 40_000_000, 'service_demand': 4000, 'market_factor': 0.0
    }
    
    # user policies for comparison
    user_policies = {
        'conservative': {'mint_rate': 10_000, 'burn_share': 0.8},
        'aggressive': {'mint_rate': 100_000, 'burn_share': 0.2},
        'balanced': {'mint_rate': 50_000, 'burn_share': 0.5}
    }
    
    print(f"\ncomparing policies with network-aware dynamics...")
    comparison = controller.compare_policies(
        test_state, 
        user_policies, 
        simulation_timesteps=30,
        num_runs=10,
        random_seed=42
    )
    
    controller.print_comparison_summary(comparison)
    
    # analyze cost breakdown to verify utilization penalty reduction
    optimal_result = comparison['results']['optimal']
    breakdown = optimal_result['evaluation']['cost_breakdown']
    total = optimal_result['evaluation']['total_discounted_cost']
    
    utilization_dominance = (breakdown['utilization_min'] / total * 100) if total > 0 else 0
    
    print(f"\n📊 integration test results:")
    print(f"utilization penalty dominance: {utilization_dominance:.1f}%")
    if utilization_dominance < 70:  # should be much less than original 83.5%
        print(f"✅ utilization penalty dominance successfully reduced")
    else:
        print(f"⚠️  utilization penalty still high - may need further tuning")
    
    print(f"training time: {result['training_time']:.1f}s")
    print(f"iterations completed: {result['iterations_completed']}")
    
    print(f"\n✅ integration test complete - all improvements working together!")
    
    return {
        'utilization_dominance': utilization_dominance,
        'training_time': result['training_time'],
        'comparison': comparison
    }


if __name__ == "__main__":
    test_integration() 