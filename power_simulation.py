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
    # plt.show()

    print("\nAll experiments complete.")
