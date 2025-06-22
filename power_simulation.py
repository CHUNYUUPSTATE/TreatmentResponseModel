import numpy as np
from scipy import stats

def simulate_phenotype_and_test(N: int, HI: float, MAF: float, beta: float, SNR: float, alpha: float = 0.05):
    """
    Simulates genotypes and phenotypes for a given set of parameters and performs
    a statistical test for association.

    Args:
        N (int): Total sample size.
        HI (float): Heterogeneity index (proportion of effective samples, 0 to 1).
        MAF (float): Minor allele frequency (0 to 0.5).
        beta (float): Effect size (change in phenotype per copy of the minor allele).
        SNR (float): Signal-to-noise ratio (variance_genotype / variance_noise).
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        bool: True if the association is statistically significant (p < alpha), False otherwise.
                 Returns None if effective sample size is too small for variance calculation or regression.
    """
    # Calculate effective sample size
    N_effective = int(N * HI)

    if N_effective < 2: # Need at least 2 samples for variance and regression
        # print(f"Warning: Effective sample size {N_effective} is too small. Skipping simulation.")
        return None

    # Simulate genotypes for N_effective individuals
    # Genotypes are 0, 1, or 2 copies of the minor allele
    # Assuming Hardy-Weinberg equilibrium for allele distribution to form genotypes
    # P(AA) = (1-MAF)^2, P(Aa) = 2*MAF*(1-MAF), P(aa) = MAF^2
    # We can simulate counts of minor alleles for each individual (0, 1, or 2)
    # Simulating two alleles for each individual and summing them up
    allele1 = np.random.binomial(1, MAF, N_effective)
    allele2 = np.random.binomial(1, MAF, N_effective)
    genotypes = allele1 + allele2

    # Simulate a continuous phenotype
    # Genetic component
    genetic_value = genotypes * beta

    # Calculate variance due to genotype
    # Var(genotype) = E[X^2] - (E[X])^2. E[X] = 2*MAF. E[X^2] for binomial B(n,p) is np(1-p) + (np)^2
    # For sum of two B(1,p) variables, Var(G) = 2 * MAF * (1-MAF)
    # Variance of genetic value = beta^2 * Var(genotype)
    var_genotype_population = 2 * MAF * (1 - MAF) * (beta**2)

    if var_genotype_population == 0 and SNR > 0:
        # This can happen if MAF is 0 or 1, or beta is 0.
        # If beta is 0, var_genotype is 0. If SNR is also 0, it's fine (noise can be anything).
        # If beta is 0, genetic_value is always 0. Phenotype is just noise.
        # If MAF is 0 or 1, all genotypes are the same. No genetic variance.
        # print(f"Warning: Population genetic variance is 0. MAF={MAF}, beta={beta}")
        # In this case, if beta=0, any association is spurious. If MAF=0/1, no variable to test.
        # We can proceed, noise will be 0 if SNR > 0, or phenotype will be all noise if SNR=0.
        # If var_genotype_population is 0 and SNR > 0, var_noise must be 0.
        var_noise = 0
    elif SNR == 0: # Infinite noise or zero genetic variance and zero SNR specified
        if var_genotype_population > 0: # Purely noise driven, but need a scale for noise
            # This case implies that genetic variance is present but SNR is zero, meaning noise variance is infinite.
            # This is problematic for simulation. We'll assume a default noise variance if SNR is 0.
            # Or, interpret SNR=0 as "no signal", so phenotype is pure noise.
            # Let's make noise relative to effect size if SNR is 0 and beta is not.
            # This part is tricky. If SNR=0, it means var_noise is infinitely larger than var_genotype.
            # For practical simulation, we might need to cap noise or define SNR differently.
            # Let's assume if SNR = 0, we use a default noise variance, e.g. 1.
            # However, the definition is var_noise = var_genotype / SNR.
            # If SNR -> 0 and var_genotype > 0, then var_noise -> infinity.
            # If var_genotype = 0, then var_noise is undefined by SNR.
            # Let's consider the scenario where beta is non-zero, MAF creates variation, but SNR=0.
            # This should mean that the noise overwhelms the signal.
            # For now, if SNR is 0, let's make var_noise large, e.g. set std_dev_noise to a high value.
            # A robust way: if SNR is very close to 0, var_noise is very large.
            # If var_genotype_population is 0 (e.g. beta=0), then phenotype is just noise.
            # Let's assume std_dev_noise = 1 if var_genotype_population = 0 and SNR = 0.
             std_dev_noise = 1.0 # Default noise level if no genetic signal or SNR is zero
        else: # var_genotype_population is 0
             std_dev_noise = 1.0 # Default noise if no signal and SNR=0
        var_noise = std_dev_noise**2
    else: # SNR > 0 and var_genotype_population > 0
        var_noise = var_genotype_population / SNR

    if var_noise < 0: # Should not happen with current logic
        var_noise = 1e-9 # Floor for variance

    std_dev_noise = np.sqrt(var_noise)
    noise = np.random.normal(0, std_dev_noise, N_effective)

    phenotypes = genetic_value + noise

    # Perform statistical test (linear regression)
    # Check for variance in genotypes. If all genotypes are the same (e.g., MAF=0 or MAF=1, or by chance in small samples),
    # regression is not possible or meaningful.
    if np.var(genotypes) == 0:
        # print(f"Warning: Genotype variance is 0 for N_effective={N_effective}. MAF={MAF}. Cannot perform regression.")
        return False # No association can be detected if there's no variation in the predictor.

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(genotypes, phenotypes)
    except ValueError as e:
        # This can happen for very small N_effective or other edge cases with inputs to linregress
        # print(f"Warning: scipy.stats.linregress failed for N_effective={N_effective}. Error: {e}")
        return False # Consider this as non-significant if the test cannot be performed.

    return p_value < alpha


def estimate_power(num_simulations: int, N: int, HI: float, MAF: float, beta: float, SNR: float, alpha: float = 0.05):
    """
    Estimates statistical power by running multiple simulations.

    Args:
        num_simulations (int): Number of simulations to run.
        N (int): Total sample size.
        HI (float): Heterogeneity index.
        MAF (float): Minor allele frequency.
        beta (float): Effect size.
        SNR (float): Signal-to-noise ratio.
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        float: Estimated statistical power (proportion of significant results).
               Returns np.nan if all simulations resulted in N_effective < 2.
    """
    significant_results = 0
    valid_simulations = 0

    for _ in range(num_simulations):
        result = simulate_phenotype_and_test(N, HI, MAF, beta, SNR, alpha)
        if result is not None: # Only count valid simulations
            valid_simulations += 1
            if result: # Result is True if significant
                significant_results += 1

    if valid_simulations == 0:
        print(f"Warning: No valid simulations for N={N}, HI={HI}. Effective sample size might be too small consistently.")
        return np.nan # Or handle as 0 power, but NaN indicates an issue with parameters.

    power = significant_results / valid_simulations
    return power

# Step 6: Develop a script to run experiments and visualize results will go here (likely in a separate main block or script)

if __name__ == '__main__':
    # Example usage (will be expanded later)
    print("Simulation script initialized.")
    # N_test = 1000
    # HI_test = 0.1
    # MAF_test = 0.2
    # beta_test = 0.5
    # SNR_test = 0.1 # Low SNR
    # alpha_test = 0.05
    # num_sims_test = 100

    # power = estimate_power(num_sims_test, N_test, HI_test, MAF_test, beta_test, SNR_test, alpha_test)
    # print(f"Estimated power for single test run: {power:.4f}")

    import matplotlib.pyplot as plt

    # --- Experiment 1: Power vs. Sample Size (N) ---
    print("\nRunning Experiment 1: Power vs. Sample Size (N)")
    sample_sizes = np.array([100, 200, 500, 1000, 2000, 5000, 10000])
    powers_n = []

    # Fixed parameters for this experiment
    HI_exp1 = 0.5      # 50% effective samples
    MAF_exp1 = 0.3     # Minor allele frequency
    beta_exp1 = 0.2    # Effect size
    SNR_exp1 = 0.05    # Signal-to-Noise Ratio (lower means more noise)
    num_sims_exp = 500 # Number of simulations per point for speed; increase for accuracy
    alpha_exp = 0.05

    for n_val in sample_sizes:
        print(f"  Simulating for N = {n_val}...")
        p = estimate_power(num_sims_exp, n_val, HI_exp1, MAF_exp1, beta_exp1, SNR_exp1, alpha_exp)
        powers_n.append(p)
        print(f"    Estimated Power: {p:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, powers_n, marker='o', linestyle='-')
    plt.title(f'Power vs. Sample Size (N)\n(HI={HI_exp1}, MAF={MAF_exp1}, beta={beta_exp1}, SNR={SNR_exp1}, Sims={num_sims_exp})')
    plt.xlabel('Total Sample Size (N)')
    plt.ylabel('Statistical Power')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig('power_vs_sample_size.png')
    print("Saved plot to power_vs_sample_size.png")
    # plt.show() # Uncomment to display plot directly

    # --- Experiment 2: Power vs. Heterogeneity Index (HI) ---
    print("\nRunning Experiment 2: Power vs. Heterogeneity Index (HI)")
    heterogeneity_indices = np.linspace(0.05, 1.0, 10) # From 5% to 100% effective
    powers_hi = []

    # Fixed parameters for this experiment
    N_exp2 = 2000      # Total sample size
    MAF_exp2 = 0.3
    beta_exp2 = 0.2
    SNR_exp2 = 0.05

    for hi_val in heterogeneity_indices:
        print(f"  Simulating for HI = {hi_val:.2f}...")
        p = estimate_power(num_sims_exp, N_exp2, hi_val, MAF_exp2, beta_exp2, SNR_exp2, alpha_exp)
        powers_hi.append(p)
        print(f"    Estimated Power: {p:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(heterogeneity_indices, powers_hi, marker='o', linestyle='-')
    plt.title(f'Power vs. Heterogeneity Index (HI)\n(N={N_exp2}, MAF={MAF_exp2}, beta={beta_exp2}, SNR={SNR_exp2}, Sims={num_sims_exp})')
    plt.xlabel('Heterogeneity Index (HI)')
    plt.ylabel('Statistical Power')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig('power_vs_heterogeneity_index.png')
    print("Saved plot to power_vs_heterogeneity_index.png")
    # plt.show()

    # --- Experiment 3: Power vs. Signal-to-Noise Ratio (SNR) ---
    print("\nRunning Experiment 3: Power vs. Signal-to-Noise Ratio (SNR)")
    snr_values = np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]) # SNR values
    powers_snr = []

    # Fixed parameters for this experiment
    N_exp3 = 2000
    HI_exp3 = 0.5
    MAF_exp3 = 0.3
    beta_exp3 = 0.2 # Keep beta constant to see SNR effect clearly

    for snr_val in snr_values:
        print(f"  Simulating for SNR = {snr_val:.3f}...")
        p = estimate_power(num_sims_exp, N_exp3, HI_exp3, MAF_exp3, beta_exp3, snr_val, alpha_exp)
        powers_snr.append(p)
        print(f"    Estimated Power: {p:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, powers_snr, marker='o', linestyle='-')
    plt.title(f'Power vs. Signal-to-Noise Ratio (SNR)\n(N={N_exp3}, HI={HI_exp3}, MAF={MAF_exp3}, beta={beta_exp3}, Sims={num_sims_exp})')
    plt.xlabel('Signal-to-Noise Ratio (SNR)')
    plt.ylabel('Statistical Power')
    plt.xscale('log') # SNR often viewed on log scale
    plt.grid(True, which="both", ls="-")
    plt.ylim(0, 1.05)
    plt.savefig('power_vs_snr.png')
    print("Saved plot to power_vs_snr.png")
    # plt.show() # plt.close() is now handled by plot_parameter_vs_power

    # --- Demonstrate plot_parameter_vs_power ---
    print("\n--- Demonstrating generalized plotting function ---")

    # Base parameters for demonstrations
    base_params = {'N': 2000, 'HI': 0.5, 'MAF': 0.2, 'beta': 0.15, 'SNR': 0.05, 'alpha': 0.05}
    num_sims_exp_main = 300 # Reduced for faster demo in main; increase for smoother curves

    # Vary N
    # Create a specific dict for fixed_params by removing the varying key from base_params
    fixed_params_N = {k: v for k, v in base_params.items() if k != 'N'}
    plot_parameter_vs_power(
        fixed_params=fixed_params_N,
        varying_param_name='N',
        varying_param_values=np.array([100, 500, 1000, 2000, 4000, 8000]),
        num_sims_per_point=num_sims_exp_main,
        title=f'Power vs. Sample Size (N)\n(HI={base_params["HI"]}, MAF={base_params["MAF"]}, beta={base_params["beta"]}, SNR={base_params["SNR"]})',
        xlabel='Total Sample Size (N)',
        filename='main_power_vs_sample_size.png'
    )

    # Vary HI
    fixed_params_HI = {k: v for k, v in base_params.items() if k != 'HI'}
    plot_parameter_vs_power(
        fixed_params=fixed_params_HI,
        varying_param_name='HI',
        varying_param_values=np.linspace(0.05, 1.0, 10),
        num_sims_per_point=num_sims_exp_main,
        title=f'Power vs. Heterogeneity Index (HI)\n(N={base_params["N"]}, MAF={base_params["MAF"]}, beta={base_params["beta"]}, SNR={base_params["SNR"]})',
        xlabel='Heterogeneity Index (HI)',
        filename='main_power_vs_hi.png'
    )

    # Vary MAF
    fixed_params_MAF = {k: v for k, v in base_params.items() if k != 'MAF'}
    plot_parameter_vs_power(
        fixed_params=fixed_params_MAF,
        varying_param_name='MAF',
        varying_param_values=np.linspace(0.01, 0.5, 10),
        num_sims_per_point=num_sims_exp_main,
        title=f'Power vs. Minor Allele Frequency (MAF)\n(N={base_params["N"]}, HI={base_params["HI"]}, beta={base_params["beta"]}, SNR={base_params["SNR"]})',
        xlabel='Minor Allele Frequency (MAF)',
        filename='main_power_vs_maf.png'
    )

    # Vary beta (Effect Size)
    fixed_params_beta = {k: v for k, v in base_params.items() if k != 'beta'}
    plot_parameter_vs_power(
        fixed_params=fixed_params_beta,
        varying_param_name='beta',
        varying_param_values=np.linspace(0.05, 0.35, 10), # Adjusted beta range for more variability
        num_sims_per_point=num_sims_exp_main,
        title=f'Power vs. Effect Size (beta)\n(N={base_params["N"]}, HI={base_params["HI"]}, MAF={base_params["MAF"]}, SNR={base_params["SNR"]})',
        xlabel='Effect Size (beta)',
        filename='main_power_vs_beta.png'
    )

    # Vary SNR
    fixed_params_SNR = {k: v for k, v in base_params.items() if k != 'SNR'}
    plot_parameter_vs_power(
        fixed_params=fixed_params_SNR,
        varying_param_name='SNR',
        varying_param_values=np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
        num_sims_per_point=num_sims_exp_main,
        title=f'Power vs. Signal-to-Noise Ratio (SNR)\n(N={base_params["N"]}, HI={base_params["HI"]}, MAF={base_params["MAF"]}, beta={base_params["beta"]})',
        xlabel='Signal-to-Noise Ratio (SNR)',
        filename='main_power_vs_snr.png',
        x_log_scale=True
    )

    # --- Demonstrate estimate_hi_for_target_power ---
    print("\n--- Demonstrating HI estimation for target power ---")
    target_p_demo = 0.80
    N_for_hi_est_demo = 3000
    MAF_for_hi_est_demo = 0.10
    beta_for_hi_est_demo = 0.20 # Adjusted effect size for demonstration
    SNR_for_hi_est_demo = 0.08  # Adjusted SNR for demonstration

    estimated_hi_demo = estimate_hi_for_target_power(
        target_power=target_p_demo,
        N=N_for_hi_est_demo,
        MAF=MAF_for_hi_est_demo,
        beta=beta_for_hi_est_demo,
        SNR=SNR_for_hi_est_demo,
        alpha=0.05,
        num_simulations_per_hi_evaluation=num_sims_exp_main, # Use same num_sims for consistency
        tolerance=0.03,
        max_iterations=15
    )

    if estimated_hi_demo is not None:
        print(f"\nTo achieve ~{target_p_demo*100}% power with N={N_for_hi_est_demo}, MAF={MAF_for_hi_est_demo}, beta={beta_for_hi_est_demo}, SNR={SNR_for_hi_est_demo},")
        print(f"the estimated Heterogeneity Index (HI) required is: {estimated_hi_demo:.4f}")
        # Verification step
        print("Verifying power with the estimated HI...")
        power_at_estimated_hi_demo = estimate_power(num_sims_exp_main + 200, N_for_hi_est_demo, estimated_hi_demo, MAF_for_hi_est_demo, beta_for_hi_est_demo, SNR_for_hi_est_demo, 0.05)
        print(f"Power achieved with estimated HI ({estimated_hi_demo:.4f}): {power_at_estimated_hi_demo:.4f if power_at_estimated_hi_demo is not np.nan else 'NaN'}")
    else:
        print(f"\nCould not estimate HI to achieve {target_p_demo*100}% power for the given parameters.")
        print(f"(N={N_for_hi_est_demo}, MAF={MAF_for_hi_est_demo}, beta={beta_for_hi_est_demo}, SNR={SNR_for_hi_est_demo})")

    print("\nAll demonstrations complete.")


# --- Reusable plotting function (to be implemented in next step) ---
def plot_parameter_vs_power(fixed_params: dict,
                            varying_param_name: str,
                            varying_param_values: list,
                            num_sims_per_point: int,
                            title: str,
                            xlabel: str,
                            filename: str,
                            x_log_scale: bool = False):
    """
    Simulates power across a range of values for a single parameter and plots the results.

    Args:
        fixed_params (dict): A dictionary of parameters that will be held constant.
                             Expected keys: N, HI, MAF, beta, SNR, alpha.
                             One of these will be overridden by varying_param_name.
        varying_param_name (str): The name of the parameter to vary (e.g., 'N', 'HI').
        varying_param_values (list): A list of values for the varying parameter.
        num_sims_per_point (int): Number of simulations to run for each parameter value.
        title (str): The title for the plot.
        xlabel (str): The label for the x-axis.
        filename (str): The filename to save the plot (e.g., 'plot.png').
        x_log_scale (bool, optional): Whether to use a log scale for the x-axis. Defaults to False.
    """
    # Implementation will involve:
    # 1. Initializing a list to store power values.
    # 2. Looping through varying_param_values:
    #    a. Creating a copy of fixed_params and updating it with the current varying value.
    #    b. Calling estimate_power with these params.
    #    c. Storing the result.
    # 3. Using matplotlib to generate and save the plot.

    import matplotlib.pyplot as plt # Ensure pyplot is imported locally or globally

    powers = []

    # Default parameters - ensure all necessary keys exist in fixed_params or provide defaults
    # This is important because estimate_power expects N, HI, MAF, beta, SNR, alpha
    current_sim_params = {
        'N': 1000, 'HI': 0.5, 'MAF': 0.1, 'beta': 0.1, 'SNR': 0.1, 'alpha': 0.05
    }
    current_sim_params.update(fixed_params) # Override defaults with provided fixed_params

    print(f"\nRunning experiment for plot: {title}")
    for val in varying_param_values:
        # Create a mutable copy for the current iteration
        iter_params = current_sim_params.copy()

        # Update the specific parameter that is varying
        if varying_param_name not in iter_params:
            print(f"Warning: varying_param_name '{varying_param_name}' not in default simulation parameters. Adding it.")
        iter_params[varying_param_name] = val

        print(f"  Simulating for {varying_param_name} = {val}...")

        # Ensure all required parameters for estimate_power are present
        power = estimate_power(
            num_simulations=num_sims_per_point,
            N=int(iter_params['N']), # Ensure N is int
            HI=float(iter_params['HI']),
            MAF=float(iter_params['MAF']),
            beta=float(iter_params['beta']),
            SNR=float(iter_params['SNR']),
            alpha=float(iter_params['alpha'])
        )
        powers.append(power)
        print(f"    Estimated Power: {power:.4f}" if power is not np.nan else "    Estimated Power: NaN")

    plt.figure(figsize=(10, 6))
    plt.plot(varying_param_values, powers, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Statistical Power')

    if x_log_scale:
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
    else:
        plt.grid(True)

    plt.ylim(0, 1.05)
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    # plt.show() # Uncomment to display plot directly
    plt.close() # Close the figure to free memory


def estimate_hi_for_target_power(target_power: float,
                                 N: int,
                                 MAF: float,
                                 beta: float,
                                 SNR: float,
                                 alpha: float = 0.05,
                                 num_simulations_per_hi_evaluation: int = 500,
                                 hi_search_min: float = 0.01,
                                 hi_search_max: float = 1.0,
                                 tolerance: float = 0.02,
                                 max_iterations: int = 10): # Max iterations for bisection search
    """
    Estimates the Heterogeneity Index (HI) required to achieve a target statistical power
    for a given set of other parameters, using a bisection search method.

    Args:
        target_power (float): The desired statistical power (e.g., 0.8 for 80%).
        N (int): Total sample size.
        MAF (float): Minor allele frequency.
        beta (float): Effect size.
        SNR (float): Signal-to-noise ratio.
        alpha (float, optional): Significance level. Defaults to 0.05.
        num_simulations_per_hi_evaluation (int, optional): Number of simulations to run
            for each HI value tested. Defaults to 500.
        hi_search_min (float, optional): Minimum HI to consider in search. Defaults to 0.01.
        hi_search_max (float, optional): Maximum HI to consider in search. Defaults to 1.0.
        tolerance (float, optional): The acceptable difference between achieved power and
                                     target_power. Defaults to 0.02.
        max_iterations (int, optional): Maximum number of iterations for the bisection search.
                                       Defaults to 10.

    Returns:
        float or None: The estimated HI value if found within tolerance and iterations,
                       otherwise None.
    """
    print(f"\nEstimating HI for target power={target_power} (N={N}, MAF={MAF}, beta={beta}, SNR={SNR})")

    low_hi = hi_search_min
    high_hi = hi_search_max

    # Evaluate power at boundary HIs
    power_at_low_hi = estimate_power(num_simulations_per_hi_evaluation, N, low_hi, MAF, beta, SNR, alpha)
    if power_at_low_hi is np.nan: power_at_low_hi = 0 # Treat NaN as 0 power for search
    print(f"  Power at HI={low_hi:.3f}: {power_at_low_hi:.4f}")
    if power_at_low_hi >= target_power:
        print(f"  Target power achieved or exceeded at minimum HI search boundary ({low_hi:.3f}).")
        return low_hi

    power_at_high_hi = estimate_power(num_simulations_per_hi_evaluation, N, high_hi, MAF, beta, SNR, alpha)
    if power_at_high_hi is np.nan: power_at_high_hi = 0 # Treat NaN as 0 power
    print(f"  Power at HI={high_hi:.3f}: {power_at_high_hi:.4f}")
    if power_at_high_hi < target_power:
        print(f"  Target power not achievable even at maximum HI search boundary ({high_hi:.3f}). Power was {power_at_high_hi:.4f}")
        return None # Target power might be too high for these parameters

    for iteration in range(max_iterations):
        mid_hi = (low_hi + high_hi) / 2
        if mid_hi <= 0: # Safety break if HI becomes non-positive
            print("  Warning: mid_hi became non-positive during search.")
            return None

        current_power = estimate_power(num_simulations_per_hi_evaluation, N, mid_hi, MAF, beta, SNR, alpha)
        if current_power is np.nan: current_power = 0 # Treat NaN as 0 power for search logic

        print(f"  Iter {iteration+1}/{max_iterations}: HI={mid_hi:.4f}, Power={current_power:.4f}")

        if abs(current_power - target_power) <= tolerance:
            print(f"  Found HI={mid_hi:.4f} achieving power {current_power:.4f} (target {target_power:.4f})")
            return mid_hi

        if current_power < target_power:
            low_hi = mid_hi
        else:
            high_hi = mid_hi

        if (high_hi - low_hi) < 0.005 : # If interval is too small, stop
            print(f"  Search interval {high_hi - low_hi:.4f} too small. Returning best guess HI based on current bounds.")
            # Return the HI that's closer or average, or the one that gives power closer to target
            # For simplicity, return mid_hi from last valid estimate, or average of bounds
            return (low_hi + high_hi) / 2


    print(f"  Failed to converge to target power within {max_iterations} iterations and tolerance {tolerance}.")
    # Check if the power at the final low_hi or high_hi is close enough, as the loop might terminate due to iterations
    # This is a bit redundant if the loop condition (high_hi - low_hi) is small enough.
    # The bisection method naturally finds a value. The question is if that value's power is within tolerance.
    # The final mid_hi from the loop might be the best estimate.
    # Let's re-evaluate the last 'low_hi' and 'high_hi' as they bracket the solution
    final_power_low = estimate_power(num_simulations_per_hi_evaluation, N, low_hi, MAF, beta, SNR, alpha)
    if final_power_low is not np.nan and abs(final_power_low - target_power) <= tolerance :
        return low_hi
    final_power_high = estimate_power(num_simulations_per_hi_evaluation, N, high_hi, MAF, beta, SNR, alpha)
    if final_power_high is not np.nan and abs(final_power_high - target_power) <= tolerance:
        return high_hi

    return None # Failed to find suitable HI
