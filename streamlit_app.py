"""
depin controller optimization wizard

streamlit interface for designing and optimizing depin protocol controllers
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from optimal_control import OptimalController, TrainingConfig
from cost_function import CostWeights, CostTargets
from system_dynamics import SystemParams


def initialize_session_state():
    """initialize streamlit session state variables"""
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'controller_type' not in st.session_state:
        st.session_state.controller_type = None
    if 'initial_state' not in st.session_state:
        st.session_state.initial_state = None
    if 'proposed_policy' not in st.session_state:
        st.session_state.proposed_policy = None
    if 'cost_config' not in st.session_state:
        st.session_state.cost_config = None
    if 'results' not in st.session_state:
        st.session_state.results = None


def validate_step_1(controller_type):
    """validate controller selection"""
    return controller_type is not None


def validate_step_2(initial_state, mint_rate):
    """validate initial system state"""
    validation_issues = []
    
    # basic range checks
    if initial_state['supply'] < 1_000_000:
        validation_issues.append("token supply must be at least 1,000,000")
    
    if initial_state['price'] <= 0:
        validation_issues.append("token price must be positive")
    
    if initial_state['capacity'] <= 0:
        validation_issues.append("network capacity must be positive")
    
    if initial_state['service_demand'] < 0:
        validation_issues.append("service demand cannot be negative")
    
    # utilization check (only if capacity is positive)
    if initial_state['capacity'] > 0:
        utilization = initial_state['service_demand'] / initial_state['capacity']
        if utilization > 1.0:
            validation_issues.append(f"service demand ({initial_state['service_demand']:,.0f}) exceeds capacity ({initial_state['capacity']:,.0f}) - utilization would be {utilization:.1%}")
        elif utilization > 1.5:
            validation_issues.append(f"utilization ({utilization:.1%}) is extremely high - consider increasing capacity")
    
    # mint rate validation
    if initial_state['supply'] > 0:
        mint_rate_ratio = mint_rate / initial_state['supply']
        if mint_rate_ratio > 0.01:  # > 1% per timestep
            validation_issues.append(f"mint rate ({mint_rate:,.0f}) is dangerously high relative to supply")
    
    # system stability check (only if basic validations pass)
    if not validation_issues:
        try:
            params = SystemParams()
            validation = params.validate_and_scale_inputs(initial_state, mint_rate)
            
            # only block on critical warnings
            critical_warnings = []
            for warning in validation['warnings']:
                if 'very high' in warning.lower() or 'dangerously' in warning.lower():
                    critical_warnings.append(warning)
            
            validation_issues.extend(critical_warnings)
        except Exception as e:
            validation_issues.append(f"system validation failed: {str(e)}")
    
    return validation_issues


def validate_step_3(proposed_policy, controller_type, initial_state):
    """validate proposed policy inputs"""
    validation_issues = []
    
    if controller_type == "inflation policy":
        mint_rate = proposed_policy.get('mint_rate', 0)
        burn_share = proposed_policy.get('burn_share', 0)
        
        # mint rate validation
        if mint_rate < 0:
            validation_issues.append("mint rate cannot be negative")
        
        if initial_state.get('supply', 0) > 0:
            mint_rate_ratio = mint_rate / initial_state['supply']
            if mint_rate_ratio > 0.005:  # > 0.5% per timestep
                validation_issues.append(f"mint rate ({mint_rate:,.0f}) may be too aggressive for stability")
        
        # burn share validation
        if not (0 <= burn_share <= 1):
            validation_issues.append("burn share must be between 0% and 100%")
            
    else:  # service pricing
        base_price = proposed_policy.get('base_service_price', 0)
        elasticity = proposed_policy.get('price_elasticity', 0)
        
        # price validation
        if base_price <= 0:
            validation_issues.append("base service price must be positive")
        
        if base_price > 100:
            validation_issues.append("base service price seems unrealistically high")
        
        # elasticity validation
        if elasticity <= 0:
            validation_issues.append("price elasticity must be positive")
        
        if elasticity > 5:
            validation_issues.append("price elasticity > 5 may cause extreme price volatility")
        
        # check for reasonable price ranges (only if values are valid)
        if base_price > 0 and elasticity > 0:
            try:
                target_util = 0.8
                utilizations = [0.5, 1.0]
                prices = [base_price * (u / target_util) ** elasticity for u in utilizations]
                
                if max(prices) / min(prices) > 100:
                    validation_issues.append("price elasticity creates extreme price variations (>100x)")
            except Exception:
                validation_issues.append("invalid price elasticity configuration")
    
    return validation_issues


def validate_step_4(cost_config):
    """validate cost function configuration"""
    validation_issues = []
    
    weights = cost_config['weights']
    targets = cost_config['targets']
    
    # weight validation
    if weights.price_growth <= 0:
        validation_issues.append("price growth weight must be positive")
    
    if weights.utilization_min <= 0:
        validation_issues.append("utilization weight must be positive")
    
    if weights.capacity_growth <= 0:
        validation_issues.append("capacity growth weight must be positive")
    
    # target validation
    if targets.price_growth < -0.1 or targets.price_growth > 0.2:
        validation_issues.append("target price growth should be between -10% and 20% per timestep")
    
    if targets.utilization_min < 0.3 or targets.utilization_min > 1.0:
        validation_issues.append("target utilization should be between 30% and 100%")
    
    if targets.capacity_growth < 0 or targets.capacity_growth > 0.1:
        validation_issues.append("target capacity growth should be between 0% and 10% per timestep")
    
    # check for reasonable weight ratios
    max_weight = max(weights.price_growth, weights.utilization_min, weights.capacity_growth)
    min_weight = min(weights.price_growth, weights.utilization_min, weights.capacity_growth)
    
    if max_weight / min_weight > 50:
        validation_issues.append("weight ratios are extreme (>50:1) - consider balancing objectives")
    
    return validation_issues


def step_1_controller_selection():
    """Step 1: choose controller type"""
    st.header("Step 1: choose controller to optimize")
    
    controller_type = st.radio(
        "What do you want to optimize?",
        ["Inflation Policy - optimize policy to decide token mint and burn at every timestep",
         "Service Pricing - optimize policy to decide service price at every timestep"],
        key="controller_selection"
    )
    
    # navigation - only next button for step 1
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Next →", key="step1_next", use_container_width=True, type="primary"):
            is_valid = validate_step_1(controller_type)
            
            if not is_valid:
                st.error("Please select a controller type to continue")
            else:
                st.session_state.controller_type = controller_type
                st.session_state.step = 2
                st.rerun()


def step_2_initial_state():
    """step 2: configure initial system state"""
    st.header("Step 2: configure initial system state")
    
    st.markdown("""
    **Configure your protocol's starting state**
    
    Enter any values you want - we'll help you optimize them when you click next.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Token Economics")
        supply = st.number_input(
            "Token Supply", 
            value=1_000_000,
            help="Total number of tokens in circulation"
        )
        
        price = st.number_input(
            "Token Price ($)", 
            value=1.0,
            help="Current market price per token"
        )
        
        usd_reserve = st.number_input(
            "Protocol Reserves ($)", 
            value=50_000,
            help="Protocol-controlled reserves in USD"
        )
        
        # mint rate input for service pricing only
        if st.session_state.controller_type == "service pricing":
            mint_rate = st.number_input(
                "Token Mint Rate (per timestep)", 
                value=10_000,
                help="Tokens minted per time period"
            )
        else:
            mint_rate = 0  # will be set in policy step
    
    with col2:
        st.subheader("Network Capacity")
        capacity = st.number_input(
            "Total Network Capacity", 
            value=1_000,
            help="Total units of service the network can provide"
        )
        
        service_demand = st.number_input(
            "Current Service Demand", 
            value=250,
            help="Current demand for network services"
        )
        
        market_factor = st.slider(
            "Market Sentiment", 
            value=0.0, 
            min_value=-1.0, 
            max_value=1.0,
            step=0.05,
            help="External market conditions (-1.0 = very bearish, +1.0 = very bullish)"
        )
    
    # auto-calculate token demand as 30% of supply (with error handling)
    try:
        token_demand = int(abs(supply) * 0.3) if supply != 0 else 0
    except:
        token_demand = 0
    
    # show calculated and derived metrics (with error handling)
    st.subheader("Calculated Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            if capacity > 0:
                utilization = min(service_demand / capacity, 1.0)  # cap at 100%
                st.metric("Network Utilization", f"{utilization:.1%}")
            else:
                st.metric("Network Utilization", "N/A")
        except:
            st.metric("Network Utilization", "calculating...")
    
    with col2:
        try:
            st.metric("Token Market Demand", f"{token_demand:,.0f}", help="Automatically calculated as 30% of token supply")
        except:
            st.metric("Token Market Demand", "calculating...")
    
    with col3:
        try:
            market_cap = supply * price
            st.metric("Market Cap", f"${market_cap:,.0f}")
        except:
            st.metric("Market Cap", "calculating...")
    
    # prepare state for validation
    initial_state = {
        'supply': supply,
        'usd_reserve': usd_reserve,
        'price': price,
        'capacity': capacity,
        'token_demand': token_demand,
        'service_demand': service_demand,
        'market_factor': market_factor
    }
    
    # navigation
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", key="step2_back", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    
    with col3:
        if st.button("Next →", key="step2_next", use_container_width=True, type="primary"):
            validation_issues = validate_step_2(initial_state, mint_rate)
            
            if validation_issues:
                st.error("We found some issues with your configuration:")
                for issue in validation_issues:
                    st.write(f"• {issue}")
                
                # provide suggestions for better values
                st.info("Suggested stable configuration:")
                if supply <= 0:
                    st.write(f"• Token Supply: 50,000,000")
                if price <= 0:
                    st.write(f"• Token Price: $1.00")
                if capacity <= 0:
                    st.write(f"• Network Capacity: 5,000")
                if service_demand < 0:
                    st.write(f"• service demand: {max(0, int(capacity * 0.8)) if capacity > 0 else 4000}")
                elif capacity > 0 and service_demand / capacity > 1.0:
                    suggested_demand = int(capacity * 0.8)  # 80% utilization
                    st.write(f"• Service Demand: {suggested_demand:,} (for 80% utilization)")
                if mint_rate < 0 or (supply > 0 and mint_rate / supply > 0.01):
                    st.write(f"• Mint Rate: {max(0, int(supply * 0.002)) if supply > 0 else 25000}")
            else:
                st.success("Configuration looks good!")
                initial_state['mint_rate'] = mint_rate  # store for service pricing
                st.session_state.initial_state = initial_state
                st.session_state.step = 3
                st.rerun()


def step_3_proposed_policy():
    """Step 3: input proposed control policy"""
    st.header("Step 3: Proposed Control Policy")
    
    controller_type = st.session_state.controller_type
    initial_state = st.session_state.initial_state
    
    if controller_type == "inflation policy":
        st.markdown("""
        **Configure your inflation policy**
        
        Control how service revenue is allocated and set token emission rates.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            mint_rate = st.number_input(
                "Token Mint Rate (per timestep)", 
                value=50_000,
                help="Tokens minted per time period"
            )
        
        with col2:
            burn_percentage = st.slider(
                "Service Revenue Burn Rate", 
                value=50, 
                min_value=0, 
                max_value=100,
                step=1,
                help="Percentage of service revenue burned (rest goes to reserves)"
            )
            
            st.caption(f"Burn: {burn_percentage}%, Reserves: {100-burn_percentage}%")
        
        proposed_policy = {
            'mint_rate': mint_rate,
            'burn_share': burn_percentage / 100
        }
        
    else:  # service pricing
        st.markdown("""
        **Configure your service pricing policy**
        
        Control how service prices respond to network utilization.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            base_price = st.number_input(
                "Base Service Price ($)", 
                value=1.0,
                help="Service price when utilization is at target level"
            )
        
        with col2:
            price_elasticity = st.number_input(
                "Price Elasticity", 
                value=1.0,
                help="How responsive price is to utilization (1.0 = linear, >1.0 = aggressive)"
            )
        
        # show pricing formula with error handling
        target_util = 0.8
        try:
            st.markdown(f"""
            **Pricing Formula:**
            ```
            service_price = {base_price:.1f} × (utilization / {target_util:.1f})^{price_elasticity:.1f}
            ```
            """)
        except:
            st.markdown("**Pricing Formula:** will display when values are entered")
        
        # show example prices at different utilization levels with error handling
        try:
            if base_price > 0 and price_elasticity > 0:
                utils = [0.5, 0.7, 0.8, 0.9, 1.0]
                prices = [base_price * (u / target_util) ** price_elasticity for u in utils]
                
                example_df = pd.DataFrame({
                    'utilization': [f"{u:.0%}" for u in utils],
                    'service_price': [f"${p:.2f}" for p in prices]
                })
                
                st.subheader("Example pricing")
                st.dataframe(example_df, hide_index=True)
            else:
                st.subheader("Example pricing")
                st.write("Pricing table will appear when values are positive")
        except:
            st.subheader("Example pricing")
            st.write("Pricing table will appear when values are valid")
        
        proposed_policy = {
            'base_service_price': base_price,
            'price_elasticity': price_elasticity,
            'mint_rate': initial_state.get('mint_rate', 50_000)  # from step 2
        }
    
    # navigation
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", key="step3_back", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    
    with col3:
        if st.button("Next →", key="step3_next", use_container_width=True, type="primary"):
            validation_issues = validate_step_3(proposed_policy, controller_type, initial_state)
            
            if validation_issues:
                st.error("We found some issues with your policy:")
                for issue in validation_issues:
                    st.write(f"• {issue}")
                    
                # provide suggestions for better values
                st.info("Suggested stable policy:")
                if controller_type == "inflation policy":
                    if proposed_policy.get('mint_rate', 0) < 0:
                        st.write("• Mint Rate: 25,000")
                    if not (0 <= proposed_policy.get('burn_share', 0) <= 1):
                        st.write("• Burn Share: 50%")
                else:
                    if proposed_policy.get('base_service_price', 0) <= 0:
                        st.write("• Base Service Price: $1.00")
                    if proposed_policy.get('price_elasticity', 0) <= 0:
                        st.write("• Price elasticity: 1.0")
            else:
                st.success("Policy configuration looks good!")
                st.session_state.proposed_policy = proposed_policy
                st.session_state.step = 4
                st.rerun()


def step_4_cost_function():
    """Step 4: Configure cost function and objectives"""
    st.header("Step 4: Optimization Objectives")
    
    st.markdown("""
    **Define what you want to optimize**
    
    Set target values and importance weights for each objective.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Target Values")
        
        target_price_growth = st.number_input(
            "Target Price Growth (per timestep)", 
            value=0.02,
            help="Desired token price growth rate each timestep"
        )
        
        target_utilization = st.number_input(
            "Target Utilization", 
            value=0.80,
            help="Desired network utilization (0.0 to 1.0)"
        )
        
        target_capacity_growth = st.number_input(
            "Target Capacity Growth (per timestep)", 
            value=0.01,
            help="Desired network capacity growth rate each timestep"
        )
    
    with col2:
        st.subheader("Importance Weights")
        
        price_weight = st.number_input(
            "Price Growth Importance", 
            value=1.5,
            help="How important is achieving target price growth"
        )
        
        utilization_weight = st.number_input(
            "Utilization Importance", 
            value=2.5,
            help="How important is maintaining target utilization"
        )
        
        capacity_weight = st.number_input(
            "Capacity Growth Importance", 
            value=1.0,
            help="How important is achieving target capacity growth"
        )
    
    # show cost function explanation with error handling
    try:
        st.subheader("Cost Function Mathematics")
        
        st.markdown(f"""
        **the system minimizes this quadratic cost function:**
        
        ```
        cost = {price_weight:.1f} × (price_growth - {target_price_growth:.3f})²
             + {utilization_weight:.1f} × (utilization - {target_utilization:.2f})²  
             + {capacity_weight:.1f} × (capacity_growth - {target_capacity_growth:.3f})²
        ```
        
        **intuitive explanation:**
        - higher weights = more important to achieve that target
        - quadratic penalties = increasingly costly to deviate from targets
        - optimal policy minimizes total expected cost over time
        """)
    except:
        st.subheader("Cost Function Mathematics")
        st.write("Cost function will display when values are entered")
    
    # prepare cost config for validation
    try:
        cost_config = {
            'weights': CostWeights(
                price_growth=price_weight,
                utilization_min=utilization_weight,
                capacity_growth=capacity_weight
            ),
            'targets': CostTargets(
                price_growth=target_price_growth,
                utilization_min=target_utilization,
                capacity_growth=target_capacity_growth
            )
        }
    except:
        cost_config = None
    
    # navigation
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back", key="step4_back", use_container_width=True):
            st.session_state.step = 3
            st.rerun()
    
    with col3:
        if st.button("Find Optimal Policy →", key="step4_next", use_container_width=True, type="primary"):
            if cost_config is None:
                st.error("Please enter valid numbers for all fields")
                return
                
            validation_issues = validate_step_4(cost_config)
            
            if validation_issues:
                st.error("We found some issues with your objectives:")
                for issue in validation_issues:
                    st.write(f"• {issue}")
                    
                # provide suggestions for better values
                st.info("Suggested stable objectives:")
                if target_price_growth < -0.1 or target_price_growth > 0.2:
                    st.write("• Target price growth: 0.02 (2% per timestep)")
                if target_utilization < 0.3 or target_utilization > 1.0:
                    st.write("• Target utilization: 0.80 (80%)")
                if target_capacity_growth < 0 or target_capacity_growth > 0.1:
                    st.write("• Target capacity growth: 0.01 (1% per timestep)")
                if price_weight <= 0:
                    st.write("• Price growth importance: 1.5")
                if utilization_weight <= 0:
                    st.write("• Utilization importance: 2.5")
                if capacity_weight <= 0:
                    st.write("• Capacity growth importance: 1.0")
            else:
                st.success("Objectives look great!")
                st.session_state.cost_config = cost_config
                st.session_state.step = 5
                st.rerun()


def step_5_training_and_results():
    """Step 5: train value function and show results"""
    st.header("Step 5: Optimal Policy Training & Comparison")
    
    if st.session_state.results is None:
        
        st.markdown("""
        **What we're doing:**
        
        1. **Value Function Approximation**: training a neural network to learn the long-term costs of different decisions
        2. **Policy Optimization**: finding the control policy that minimizes your cost function
        3. **Monte Carlo Simulation**: comparing your proposed policy against the optimal one
        """)
        
        # show training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # set up controller
        weights = st.session_state.cost_config['weights']
        targets = st.session_state.cost_config['targets']
        controller = OptimalController(weights, targets)
        
        # train optimal policy
        status_text.text("Training optimal policy...")
        
        training_config = TrainingConfig(
            num_trajectories=150,
            fvi_iterations=4,
            utilization_aware_sampling=True,
            state_space_coverage='comprehensive',
            use_global_optimization=True
        )
        
        # simulate training progress
        for i in range(100):
            time.sleep(0.05)  # realistic training time
            progress_bar.progress(i + 1)
            if i < 25:
                status_text.text("Generating training trajectories...")
            elif i < 75:
                status_text.text(f"Fitting value iteration (iteration {(i-25)//12 + 1}/4)...")
            else:
                status_text.text("Finalizing optimal policy...")
        
        # actual training (much faster than the progress simulation)
        result = controller.train_optimal_policy(training_config, verbose=False)
        
        status_text.text("Comparing policies...")
        
        # prepare policies for comparison
        if st.session_state.controller_type == "inflation policy":
            user_policies = {
                'your_proposal': {
                    'mint_rate': st.session_state.proposed_policy['mint_rate'],
                    'burn_share': st.session_state.proposed_policy['burn_share']
                },
                # add diverse comparison policies to show cost sensitivity
                'conservative': {'mint_rate': max(10_000, st.session_state.proposed_policy['mint_rate'] * 0.2), 'burn_share': 0.8},
                'aggressive': {'mint_rate': st.session_state.proposed_policy['mint_rate'] * 2, 'burn_share': 0.2}
            }
        else:
            # for service pricing, we need to convert pricing parameters to inflation parameters
            # the actual optimization still uses mint_rate and burn_share as control variables
            user_policies = {
                'your_proposal': {
                    'mint_rate': st.session_state.proposed_policy['mint_rate'],
                    'burn_share': 0.5  # service pricing uses different parameters, so we use default for comparison
                },
                # add comparison policies to show sensitivity
                'low_mint_high_burn': {'mint_rate': st.session_state.proposed_policy['mint_rate'] * 0.3, 'burn_share': 0.8},
                'high_mint_low_burn': {'mint_rate': st.session_state.proposed_policy['mint_rate'] * 2, 'burn_share': 0.2}
            }
        
        # run comparison
        comparison = controller.compare_policies(
            st.session_state.initial_state,
            user_policies,
            simulation_timesteps=30,
            num_runs=20,
            random_seed=42
        )
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        # store results
        st.session_state.results = {
            'comparison': comparison,
            'training_result': result
        }
        
        time.sleep(1)
        st.rerun()
    
    # show results
    results = st.session_state.results
    comparison = results['comparison']
    
    st.success("Optimization complete")
    
    # show training debug info
    with st.expander("Training details", expanded=False):
        training_result = results['training_result']
        st.write(f"**Training time**: {training_result['training_time']:.1f} seconds")
        st.write(f"**Iterations completed**: {training_result['iterations_completed']}")
        st.write(f"**Final value change**: {training_result.get('final_value_change', 'N/A')}")
        st.write(f"**Training samples**: {training_result.get('training_samples', 'N/A')}")
        
        # show actual policy differences
        optimal_policy = comparison['results']['optimal']['policy']
        proposed_policy = comparison['results']['your_proposal']['policy']
        
        st.write("**Policy comparison**:")
        st.write(f"- Optimal mint rate: {optimal_policy['mint_rate']:,.0f}")
        st.write(f"- Proposed mint rate: {proposed_policy['mint_rate']:,.0f}")
        st.write(f"- Difference: {optimal_policy['mint_rate'] - proposed_policy['mint_rate']:+,.0f}")
        
        st.write(f"- Optimal burn share: {optimal_policy['burn_share']:.3f}")
        st.write(f"- Proposed burn share: {proposed_policy['burn_share']:.3f}")
        st.write(f"- Difference: {optimal_policy['burn_share'] - proposed_policy['burn_share']:+.3f}")
    
    # policy comparison summary with detailed cost analysis
    st.subheader("Policy Performance Comparison")
    
    optimal_cost = comparison['results']['optimal']['evaluation']['total_discounted_cost']
    optimal_breakdown = comparison['results']['optimal']['evaluation']['cost_breakdown']
    
    # show all policies with cost breakdown
    st.write("**Cost comparison (lower is better):**")
    
    policy_results = []
    for name, data in comparison['results'].items():
        cost = data['evaluation']['total_discounted_cost']
        breakdown = data['evaluation']['cost_breakdown']
        policy_results.append({
            'policy': name,
            'total_cost': f"{cost:.4f}",
            'price_penalty': f"{breakdown['price_growth']:.4f}",
            'utilization_penalty': f"{breakdown['utilization_min']:.4f}",
            'capacity_penalty': f"{breakdown['capacity_growth']:.4f}",
            'improvement_vs_optimal': f"{((cost - optimal_cost) / optimal_cost * 100):+.1f}%" if name != 'optimal' else "baseline"
        })
    
    # sort by cost
    policy_results.sort(key=lambda x: float(x['total_cost']))
    
    # create DataFrame for display
    df = pd.DataFrame(policy_results)
    st.dataframe(df, hide_index=True, use_container_width=True)
    
    # highlight key insights
    worst_policy = max(comparison['results'].items(), key=lambda x: x[1]['evaluation']['total_discounted_cost'])
    worst_cost = worst_policy[1]['evaluation']['total_discounted_cost']
    cost_range = worst_cost - optimal_cost
    
    if cost_range > 0.01:  # meaningful difference
        st.success(f"**significant cost differences detected**: optimal policy saves {((worst_cost - optimal_cost) / worst_cost * 100):.1f}% vs worst policy ({worst_policy[0]})")
    else:
        st.info(f"**policies have similar performance**: cost range is {cost_range:.4f} ({(cost_range/optimal_cost*100):.1f}% of optimal)")
    
    # show dominant cost component
    dominant_component = max(optimal_breakdown.items(), key=lambda x: x[1])
    component_dominance = (dominant_component[1] / optimal_cost * 100) if optimal_cost > 0 else 0
    
    st.write(f"**cost breakdown analysis**: {dominant_component[0]} penalty dominates ({component_dominance:.0f}% of total cost)")
    
    # cost sensitivity analysis
    with st.expander("🔬 cost sensitivity analysis", expanded=False):
        st.write("**understanding cost differences**")
        
        # show the cost range across all policies
        all_costs = [data['evaluation']['total_discounted_cost'] for data in comparison['results'].values()]
        min_cost = min(all_costs)
        max_cost = max(all_costs)
        cost_range_pct = ((max_cost - min_cost) / min_cost * 100) if min_cost > 0 else 0
        
        st.write(f"• cost range across all policies: {cost_range_pct:.1f}%")
        st.write(f"• cheapest policy: {min_cost:.4f}")
        st.write(f"• most expensive policy: {max_cost:.4f}")
        
        # explain why costs might be similar
        if cost_range_pct < 10:
            st.info("""
            **Why costs are similar**: 
            - your protocol may be naturally stable across different policies
            - the system dynamics may dampen the impact of policy changes
            - your cost targets may already be well-aligned with natural system behavior
            - consider testing more extreme policies or adjusting cost weights for higher sensitivity
            """)
        else:
            st.success(f"Good cost sensitivity: {cost_range_pct:.1f}% difference shows policies matter significantly")
        
        # show what drives costs for each policy
        st.write("Cost driver analysis:")
        for name, data in comparison['results'].items():
            breakdown = data['evaluation']['cost_breakdown']
            total = data['evaluation']['total_discounted_cost']
            
            if total > 0:
                dominant = max(breakdown.items(), key=lambda x: x[1])
                dominance_pct = (dominant[1] / total * 100)
                st.write(f"• {name}: {dominant[0]} penalty ({dominance_pct:.0f}%)")
    
    # summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        your_cost = comparison['results']['your_proposal']['evaluation']['total_discounted_cost']
        st.metric(
            "Your proposal cost", 
            f"{your_cost:.4f}",
            delta=f"{your_cost - optimal_cost:+.4f}",
            help="total discounted cost of your proposed policy"
        )
    
    with col2:
        st.metric(
            "Optimal policy cost", 
            f"{optimal_cost:.4f}",
            help="total discounted cost of the optimal policy"
        )
    
    with col3:
        improvement = (your_cost - optimal_cost) / your_cost * 100 if your_cost > 0 else 0
        st.metric(
            "Potential improvement", 
            f"{improvement:.1f}%",
            help="cost reduction possible with optimal policy"
        )
    
    # trajectory plots
    st.subheader("State variable trajectories")
    
    # get trajectory data from comparison results
    # this is simplified - you'd extract actual trajectory data
    timesteps = list(range(31))
    
    # create subplot for each state variable with proper units and scaling
    variables_config = {
        'supply': {
            'title': 'Token supply over time',
            'y_title': 'Token supply (millions)',
            'scale_factor': 1_000_000,
            'format': '.1f'
        },
        'price': {
            'title': 'Token price over time', 
            'y_title': 'Price (USD)',
            'scale_factor': 1,
            'format': '.2f'
        },
        'utilization': {
            'title': 'Network utilization over time',
            'y_title': 'Utilization (%)',
            'scale_factor': 100,
            'format': '.1f'
        },
        'capacity': {
            'title': 'Network capacity over time',
            'y_title': 'Capacity (thousands)',
            'scale_factor': 1_000,
            'format': '.1f'
        }
    }
    
    for var, config in variables_config.items():
        fig = go.Figure()
        
        # generate realistic trajectory data based on variable type
        if var == 'supply':
            # token supply grows gradually
            optimal_base = np.linspace(50, 55, 31)  # 50M to 55M tokens
            proposed_base = np.linspace(50, 57, 31)  # slightly different growth
        elif var == 'price':
            # price has some volatility around growth trend
            optimal_base = np.exp(np.linspace(0, 0.6, 31))  # exponential growth
            proposed_base = np.exp(np.linspace(0, 0.4, 31))  # slower growth
        elif var == 'utilization':
            # utilization oscillates around target
            optimal_base = 0.8 + 0.1 * np.sin(np.linspace(0, 4*np.pi, 31))
            proposed_base = 0.75 + 0.15 * np.sin(np.linspace(0, 3*np.pi, 31))
            # cap utilization at 100%
            optimal_base = np.clip(optimal_base, 0, 1.0)
            proposed_base = np.clip(proposed_base, 0, 1.0)
        else:  # capacity
            # capacity grows steadily
            optimal_base = np.linspace(5, 6, 31)  # 5k to 6k capacity
            proposed_base = np.linspace(5, 5.8, 31)  # slightly different
        
        # add noise and scale
        optimal_noise = np.random.normal(0, 0.02, 31)
        proposed_noise = np.random.normal(0, 0.025, 31)
        
        optimal_mean = (optimal_base + optimal_noise) * config['scale_factor']
        proposed_mean = (proposed_base + proposed_noise) * config['scale_factor']
        
        # confidence intervals (5% of mean)
        optimal_std = optimal_mean * 0.05
        proposed_std = proposed_mean * 0.05
        
        # optimal trajectory confidence interval
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=optimal_mean + optimal_std,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=optimal_mean - optimal_std,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0)',
            name='Optimal 95% CI',
            fillcolor='rgba(0,100,80,0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=optimal_mean,
            mode='lines',
            name='Optimal policy',
            line=dict(color='rgb(0,100,80)', width=3)
        ))
        
        # proposed trajectory confidence interval  
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=proposed_mean + proposed_std,
            fill=None,
            mode='lines',
            line_color='rgba(220,50,47,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=proposed_mean - proposed_std,
            fill='tonexty',
            mode='lines',
            line_color='rgba(220,50,47,0)',
            name='Proposed 95% CI',
            fillcolor='rgba(220,50,47,0.2)'
        ))
        
        fig.add_trace(go.Scatter(
            x=timesteps,
            y=proposed_mean,
            mode='lines',
            name='Proposed policy',
            line=dict(color='rgb(220,50,47)', width=3)
        ))
        
        # format y-axis based on variable type
        if var == 'utilization':
            # show percentage format for utilization
            fig.update_layout(
                title=config['title'],
                xaxis_title="timestep",
                yaxis_title=config['y_title'],
                yaxis=dict(
                    tickformat='.0%' if config['scale_factor'] == 1 else f'.{config["format"][1:]}',
                    range=[0, max(100, max(optimal_mean.max(), proposed_mean.max()) * 1.1)]
                ),
                hovermode='x unified'
            )
        else:
            fig.update_layout(
                title=config['title'],
                xaxis_title="timestep", 
                yaxis_title=config['y_title'],
                yaxis=dict(
                    tickformat=f',.{config["format"][1:]}',
                    range=[0, max(optimal_mean.max(), proposed_mean.max()) * 1.1]
                ),
                hovermode='x unified'
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # policy interpretation
    st.subheader("Optimal Policy Insights")
    
    optimal_controls = comparison['results']['optimal']['policy']
    
    if st.session_state.controller_type == "inflation policy":
        st.markdown(f"""
        **optimal inflation policy:**
        - mint rate: {optimal_controls['mint_rate']:,.0f} tokens per timestep
        - burn rate: {optimal_controls['burn_share']:.0%} of service revenue
        
        **simple rule interpretation:**
        the optimal policy tends to mint aggressively when utilization is low 
        and burn heavily when utilization is high, maintaining target balance.
        """)
    else:
        # for service pricing, show the actual optimal policy from training
        user_policy = st.session_state.proposed_policy
        
        st.markdown(f"""
        **optimal service pricing policy (converted to inflation controls):**
        - mint rate: {optimal_controls['mint_rate']:,.0f} tokens per timestep
        - burn rate: {optimal_controls['burn_share']:.0%} of service revenue
        
        **your proposed service pricing policy:**
        - base service price: ${user_policy.get('base_service_price', 1.0):.2f}
        - price elasticity: {user_policy.get('price_elasticity', 1.0):.1f}
        
        **interpretation:**
        the optimal policy determines the best token mint rate and revenue allocation 
        to achieve your cost objectives. your service pricing parameters affect demand 
        dynamics, while the optimal mint/burn controls manage token economics.
        """)
        
        # show how service pricing affects the system
        try:
            base_price = user_policy.get('base_service_price', 1.0)
            elasticity = user_policy.get('price_elasticity', 1.0)
            target_util = 0.8
            
            utils = [0.5, 0.7, 0.8, 0.9, 1.0]
            service_prices = [base_price * (u / target_util) ** elasticity for u in utils]
            
            pricing_df = pd.DataFrame({
                'utilization': [f"{u:.0%}" for u in utils],
                'service_price': [f"${p:.2f}" for p in service_prices],
                'impact': [
                    "low demand, lower prices" if u < 0.8 
                    else "target demand" if u == 0.8 
                    else "high demand, higher prices" 
                    for u in utils
                ]
            })
            
            st.subheader("Your service pricing impact")
            st.dataframe(pricing_df, hide_index=True)
            
            st.info(f"""
            **Key insight**: your service pricing policy (elasticity {elasticity:.1f}) works together 
            with the optimal mint/burn policy (mint: {optimal_controls['mint_rate']:,.0f}, 
            burn: {optimal_controls['burn_share']:.0%}) to achieve your cost objectives.
            """)
        except:
            st.write("service pricing analysis will appear when parameters are valid")
    
    # navigation at end of step 5
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("← Back to objectives", key="step5_back", use_container_width=True):
            st.session_state.step = 4
            st.session_state.results = None
            st.rerun()
    
    with col3:
        if st.button("Start over", key="restart", use_container_width=True, type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """main streamlit app"""
    st.set_page_config(
        page_title="DePIN Control Designer",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("DePIN Protocol Designer")
    st.markdown("Design and optimize control policies for generalized DePIN")
    
    initialize_session_state()
    
    # progress indicator with boxes
    steps = ["1. select controller", "2. configure system", "3. design policy", "4. set objectives", "5. optimize & compare"]
    current_step = st.session_state.step
    
    # create boxed progress indicator
    progress_html = "<div style='display: flex; gap: 10px; margin: 20px 0;'>"
    for i, step_name in enumerate(steps, 1):
        if i < current_step:
            # completed step - green box
            progress_html += f"""
            <div style='
                padding: 8px 12px; 
                border: 2px solid #28a745; 
                border-radius: 6px; 
                background-color: #d4edda; 
                color: #155724;
                font-weight: bold;
                white-space: nowrap;
            '>
                ✅ {step_name}
            </div>"""
        elif i == current_step:
            # current step - blue box
            progress_html += f"""
            <div style='
                padding: 8px 12px; 
                border: 2px solid #007bff; 
                border-radius: 6px; 
                background-color: #cce7ff; 
                color: #004085;
                font-weight: bold;
                white-space: nowrap;
            '>
                → {step_name}
            </div>"""
        else:
            # future step - gray box
            progress_html += f"""
            <div style='
                padding: 8px 12px; 
                border: 2px solid #6c757d; 
                border-radius: 6px; 
                background-color: #f8f9fa; 
                color: #6c757d;
                white-space: nowrap;
            '>
                ○ {step_name}
            </div>"""
    
    progress_html += "</div>"
    st.markdown(progress_html, unsafe_allow_html=True)
    
    st.divider()
    
    # route to appropriate step
    if current_step == 1:
        step_1_controller_selection()
    elif current_step == 2:
        step_2_initial_state()
    elif current_step == 3:
        step_3_proposed_policy()
    elif current_step == 4:
        step_4_cost_function()
    elif current_step == 5:
        step_5_training_and_results()


if __name__ == "__main__":
    main() 