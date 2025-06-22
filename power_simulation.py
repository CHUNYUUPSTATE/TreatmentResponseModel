"""
Power Simulation Script for Genetic Association Studies

This script provides tools to:
1. Simulate phenotypes and genotypes to estimate statistical power under various scenarios.
2. Plot relationships between power and parameters like sample size (N), Minor Allele
   Frequency (MAF), effect size (beta), Signal-to-Noise Ratio (SNR), and
   Heterogeneity Index (HI).
3. Estimate the Heterogeneity Index (HI) required to achieve a target statistical power.

The simulations assume a quantitative phenotype model.

Command-Line Interface (CLI):
The script uses `argparse` to provide a CLI for its main functionalities:
  - `plot`: Generates power curves by varying one parameter while others are fixed.
  - `estimate_hi`: Estimates the HI needed for a specified target power.

Handling Case/Control Sample Sizes (N):
The core simulation (`simulate_phenotype_and_test`) uses a single total sample size `N`
and models a quantitative trait.
- For the `plot` command:
    - If 'N' is in `--fixed_params_json`, it should be the total sample size.
    - If `--varying_param_name` is 'N', the values represent total sample size.
- For the `estimate_hi` command:
    - Provide `--N_cases` and `--N_controls` separately. The script will use their sum as total N
      for the simulation.
In all cases, the effect size 'beta' and SNR should be considered in terms of their
impact on an underlying quantitative liability scale if you are applying results to a
case/control design.

Typical workflow for HI estimation based on observed signals (e.g., from a GWAS):
If you observed 'k' signals at a p-value 'alpha_obs' (e.g., 5e-8) with N_cases and N_controls,
and you assume these signals were detected with approximately 'target_P' power (e.g., 80%),
you can use the 'estimate_hi' command with:
  --target_power [target_P]
  --N_cases [num_cases]
  --N_controls [num_controls]
  --alpha [alpha_obs]
  --MAF [typical_MAF_for_signals]
  --beta [typical_beta_for_signals]
  --SNR [typical_SNR_for_signals]
This will find an HI consistent with these assumptions.
"""
import numpy as np
from scipy import stats
import argparse
import json

# Helper function to parse varying_param_values_spec
def parse_varying_param_values(spec_str: str) -> np.ndarray:
    """
    Parses a string specification for numerical values into a NumPy array.

    The string can be:
    1. A comma-separated list of numbers (e.g., "1,2,3,4,5").
    2. A "start,stop,num" specification for np.linspace (e.g., "0.1,1.0,10").
       This is interpreted as np.linspace(start, stop, int(num)) if:
       - 'num' is an integer >= 2.
       - 'start' <= 'stop'.
       Otherwise, a 3-element list is treated as explicit values.

    Args:
        spec_str (str): The string specification.

    Returns:
        np.ndarray: The array of numerical values.

    Raises:
        ValueError: If the string format is invalid or results in an empty array.
    """
    parts = [float(p.strip()) for p in spec_str.split(',')]
    if len(parts) == 3:
        # Check if it's likely meant for linspace
        # A simple heuristic: if the third number is an integer and >= 2, assume linspace
        if parts[2] == int(parts[2]) and parts[2] >= 2:
             # Check if start <= stop for linspace
            if parts[0] <= parts[1]:
                return np.linspace(parts[0], parts[1], int(parts[2]))
            else: # if start > stop, it's probably a list of 3 values
                return np.array(parts)
        else: # if third number is not int or <2, assume it's a list of 3 values
            return np.array(parts)
    elif len(parts) > 0:
        return np.array(parts)
    else:
        raise ValueError("Invalid format for varying_param_values_spec. Use 'start,stop,num' or 'v1,v2,v3,...'.")

def create_parser():
    """
    Creates and configures the argparse parser for the script's command-line interface.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Simulate and analyze statistical power for genetic association studies. See script docstring for more details on N/beta/SNR for case-control.",
        formatter_class=argparse.RawTextHelpFormatter # To allow newlines in help messages
        )
    parser.epilog = """\
Examples:

1. Plot Power vs. Sample Size (N):
   python power_simulation.py plot \\
     --fixed_params_json "{\\"HI\\": 0.5, \\"MAF\\": 0.1, \\"beta\\": 0.05, \\"SNR\\": 0.01, \\"alpha\\": 5e-8}" \\
     --varying_param_name N \\
     --varying_param_values_spec "10000,50000,5" \\
     --num_sims_per_point 100 \\
     --filename N_vs_power.png \\
     --title "Power vs. Sample Size"

2. Estimate HI for 80% power (e.g., for observed GWAS signals):
   python power_simulation.py estimate_hi \\
     --target_power 0.8 \\
     --N_cases 10000 \\
     --N_controls 20000 \\
     --MAF 0.05 \\
     --beta 0.075 \\
     --SNR 0.02 \\
     --alpha 5e-8 \\
     --num_sims_eval 200
"""
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands. Use [command] --help for more details.")

    # --- Plot Subcommand ---
    plot_parser = subparsers.add_parser("plot", help="Generate power plots by varying one parameter.")
    plot_parser.add_argument("--fixed_params_json", type=str, required=True,
                             help="JSON string of fixed parameters (e.g., '{\"N\": 1000, \"MAF\": 0.1, ...}'). "
                                  "For N in case/control, use total N = Ncases + Ncontrols. "
                                  "Beta/SNR should be on quantitative scale.")
    plot_parser.add_argument("--varying_param_name", type=str, required=True,
                             choices=['N', 'HI', 'MAF', 'beta', 'SNR', 'alpha'],
                             help="Name of the parameter to vary.")
    plot_parser.add_argument("--varying_param_values_spec", type=str, required=True,
                             help="Specification for varying parameter values. "
                                  "Use 'start,stop,num' for np.linspace (e.g., '0.1,1.0,10') "
                                  "or comma-separated values (e.g., '100,500,1000').")
    plot_parser.add_argument("--num_sims_per_point", type=int, default=500,
                             help="Number of simulations for each point on the plot (default: 500).")
    plot_parser.add_argument("--title", type=str, default="Power Analysis Plot",
                             help="Title for the plot.")
    plot_parser.add_argument("--xlabel", type=str, default=None,
                             help="Label for the x-axis (defaults to varying_param_name).")
    plot_parser.add_argument("--filename", type=str, default="power_plot.png",
                             help="Filename to save the plot (default: power_plot.png).")
    plot_parser.add_argument("--x_log_scale", action="store_true",
                             help="Use a logarithmic scale for the x-axis.")

    # --- Estimate HI Subcommand ---
    est_hi_parser = subparsers.add_parser("estimate_hi",
                                          help="Estimate Heterogeneity Index (HI) for a target power. "
                                               "Useful for scenarios like: given N_cases/N_controls, alpha, and "
                                               "typical MAF/beta/SNR for observed signals, what HI yields a target power (e.g., 0.8)?")
    est_hi_parser.add_argument("--target_power", type=float, required=True, help="Target statistical power (e.g., 0.8).")
    est_hi_parser.add_argument("--N_cases", type=int, required=True, help="Number of cases.")
    est_hi_parser.add_argument("--N_controls", type=int, required=True, help="Number of controls.")
    est_hi_parser.add_argument("--MAF", type=float, required=True, help="Typical Minor Allele Frequency for the signals.")
    est_hi_parser.add_argument("--beta", type=float, required=True,
                               help="Typical effect size (on quantitative scale) for the signals. "
                                    "For case/control, this should correspond to an effect on the liability scale.")
    est_hi_parser.add_argument("--SNR", type=float, required=True,
                               help="Typical Signal-to-Noise Ratio for the signals (on quantitative scale).")
    est_hi_parser.add_argument("--alpha", type=float, default=0.05,
                               help="Significance level (e.g., 5e-8 for GWAS). Default: 0.05.")
    est_hi_parser.add_argument("--num_sims_eval", type=int, default=500,
                               help="Number of simulations per HI evaluation in search (default: 500).")
    est_hi_parser.add_argument("--hi_min", type=float, default=0.01, help="Minimum HI for search (default: 0.01).")
    est_hi_parser.add_argument("--hi_max", type=float, default=1.0, help="Maximum HI for search (default: 1.0).")
    est_hi_parser.add_argument("--tolerance", type=float, default=0.02,
                               help="Tolerance for power difference in HI search (default: 0.02).")
    est_hi_parser.add_argument("--max_iter", type=int, default=10,
                               help="Maximum iterations for HI bisection search (default: 10).")

    return parser

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

    parser = create_parser()
    args = parser.parse_args()

    if args.command == "plot":
        try:
            fixed_params = json.loads(args.fixed_params_json)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string for fixed_params_json: {e}")
            parser.print_help()
            return

        try:
            varying_values = parse_varying_param_values(args.varying_param_values_spec)
        except ValueError as e:
            print(f"Error: Invalid format for varying_param_values_spec: {e}")
            parser.print_help()
            return

        xlabel_to_use = args.xlabel if args.xlabel else args.varying_param_name

        # Ensure fixed_params has all necessary keys for plot_parameter_vs_power defaults,
        # though plot_parameter_vs_power itself has defaults.
        # It's good practice for fixed_params to be comprehensive excluding the varying one.
        # Example: if varying 'N', fixed_params should ideally contain HI, MAF, beta, SNR, alpha.
        # The plot_parameter_vs_power function handles merging these with its own internal defaults.

        plot_parameter_vs_power(
            fixed_params=fixed_params,
            varying_param_name=args.varying_param_name,
            varying_param_values=varying_values,
            num_sims_per_point=args.num_sims_per_point,
            title=args.title,
            xlabel=xlabel_to_use,
            filename=args.filename,
            x_log_scale=args.x_log_scale
        )
        print(f"\nPlot generated: {args.filename}")

    elif args.command == "estimate_hi":
        N_total = args.N_cases + args.N_controls
        print(f"Calculating total N = {args.N_cases} (cases) + {args.N_controls} (controls) = {N_total}")
        estimated_hi = estimate_hi_for_target_power(
            target_power=args.target_power,
            N=N_total, # Pass the calculated total N
            MAF=args.MAF,
            beta=args.beta,
            SNR=args.SNR,
            alpha=args.alpha,
            num_simulations_per_hi_evaluation=args.num_sims_eval,
            hi_search_min=args.hi_min,
            hi_search_max=args.hi_max,
            tolerance=args.tolerance,
            max_iterations=args.max_iter
        )

        if estimated_hi is not None:
            print(f"\nEstimated HI to achieve ~{args.target_power*100:.1f}% power: {estimated_hi:.4f}")
            print("Parameters used for estimation:")
            print(f"  N_cases    : {args.N_cases}")
            print(f"  N_controls : {args.N_controls}")
            print(f"  N_total    : {N_total}")
            print(f"  MAF        : {args.MAF}")
            print(f"  Beta       : {args.beta}")
            print(f"  SNR        : {args.SNR}")
            print(f"  Alpha      : {args.alpha}")
            print(f"  Sims/eval  : {args.num_sims_eval}")
            print(f"  Tolerance  : {args.tolerance}")
        else:
            print(f"\nCould not estimate HI for {args.target_power*100:.1f}% power with the given parameters and search settings.")
    else:
        parser.print_help()

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
