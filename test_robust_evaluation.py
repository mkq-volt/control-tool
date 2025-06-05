"""
test robust policy evaluation

demonstrate how the enhanced evaluation methodology properly handles
stochasticity in depin simulations with confidence intervals
"""

from optimal_control import OptimalController, TrainingConfig
from cost_function import CostWeights, CostTargets


def test_robust_evaluation():
    """test robust multi-run policy evaluation"""
    
    print("🧪 testing robust policy evaluation with stochasticity")
    print("=" * 60)
    
    # setup
    weights = CostWeights(price_growth=1.5, utilization_min=2.5, capacity_growth=1.0)
    targets = CostTargets(price_growth=0.02, utilization_min=0.8, capacity_growth=0.01)
    controller = OptimalController(weights, targets)
    
    # train optimal policy
    training_config = TrainingConfig(
        num_trajectories=100,
        fvi_iterations=4, 
        min_iterations=2,
        max_iter=200
    )
    
    print("training optimal policy...")
    controller.train_optimal_policy(training_config, verbose=False)
    
    # test state
    test_state = {
        'supply': 50_000_000, 'usd_reserve': 500_000, 'price': 1.0, 'capacity': 5000,
        'token_demand': 40_000_000, 'service_demand': 4000, 'market_factor': 0.0
    }
    
    # user policies
    user_policies = {
        'conservative': {'mint_rate': 10_000, 'burn_share': 0.8},
        'aggressive': {'mint_rate': 100_000, 'burn_share': 0.2},
        'balanced': {'mint_rate': 50_000, 'burn_share': 0.5}
    }
    
    print(f"\n🔬 comparison with robust evaluation (20 runs each):")
    comparison = controller.compare_policies(
        test_state, 
        user_policies, 
        simulation_timesteps=30,
        num_runs=20,
        random_seed=42
    )
    
    controller.print_comparison_summary(comparison)
    
    # demonstrate single-run vs multi-run difference
    print(f"\n📈 single-run variance demonstration:")
    optimal_policy = controller.get_optimal_policy(test_state)['controls']
    
    print("single-run costs for optimal policy (5 different runs):")
    for i in range(5):
        single_eval = controller.cost_function.evaluate_policy(
            test_state, optimal_policy, timesteps=30
        )
        cost = single_eval['total_discounted_cost']
        print(f"  run {i+1}: {cost:.4f}")
    
    # robust evaluation
    robust_eval = controller.cost_function.evaluate_policy_robust(
        test_state, optimal_policy, timesteps=30, num_runs=20, random_seed=42
    )
    
    print(f"\nrobust evaluation (20 runs):")
    print(f"  mean: {robust_eval['total_discounted_cost']:.4f}")
    print(f"  std:  {robust_eval['cost_std']:.4f}")
    print(f"  95% CI: [{robust_eval['confidence_95'][0]:.4f}, {robust_eval['confidence_95'][1]:.4f}]")
    print(f"  min/max: {robust_eval['cost_min']:.4f} / {robust_eval['cost_max']:.4f}")
    
    print(f"\n✅ robust evaluation complete - stochasticity handled properly!")
    
    return controller, comparison


if __name__ == "__main__":
    test_robust_evaluation() 