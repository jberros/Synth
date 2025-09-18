# Synth
Generating simulated price paths for the Synth Subnet competition

## Baseline
The provided notebook is an improvement upon a basic basline using Geometric Brownian Motion (GBM) model for generating simulated price paths, which serves as the baseline for the Synth Subnet competition (as described in the whitepaper and baseline notebook). 
This baseline estimates drift (mu) and volatility (sigma) from short-term historical log returns and simulates paths using standard GBM assumptions (normal shocks, constant volatility). 
However, the whitepaper and README emphasize that superior performance requires capturing real-world dynamics like volatility clustering, fat-tailed distributions, and skewed returns to minimize the Continuous Ranked Probability Score (CRPS) across evaluation horizons (5min, 30min, 3h, 24h). 
CRPS penalizes poorly calibrated distributions, so improvements focus on better uncertainty quantification.Key insights from documentation:Supported assets and params: BTC, ETH, XAU, SOL; fixed to 100 simulations, 5min increments (300s), 24h horizon (86400s), start_time ~1min in future.

## Data 
Use Pyth for historical prices (up to start_time); baseline fetches 2 days back at 5min resolution.
Evaluation: CRPS on relative price changes in basis points (bp): sum over horizons; lower is better. Ensemble of 100 paths represents the distribution.
Baseline limitations: Simple GBM ignores vol clustering (use EWMA for recent vol emphasis); assumes Gaussian shocks (use Student's t for fat tails); includes historical drift (set mu=0 for short-term neutrality); short history (extend to 30 days for robust estimates); dt scaling bug (historical returns are per 5min, but code treats mu/sigma as hourly—fix by treating step as unit time since increment matches resolution).
Advanced suggestions: Whitepaper recommends GARCH/stochastic vol; notebook uses asset-specific vol scaling. Since env lacks 'arch' lib but has scipy/statsmodels/numpy, use EWMA (simple GARCH proxy) + t-dist shocks.

## Main Improvements
- Improvements prioritize simplicity, speed (fit/simulate in <1s for 100 paths), and CRPS gains without external deps beyond scipy (available). 
- No model loading in __init__ (stateless, fits per call).
### Fix historical data fetching:
- Extend to 30 days (timedelta(days=30)) for ~8640 points: Improves mu/sigma stability (2 days too noisy for vol clustering).
### Improve parameter estimation in `generate_simulations`:
- Compute returns = np.diff(np.log(past_prices)).
- Set mu = 0: Historical mu is noisy/unpredictable for short horizons; zero drift better aligns with neutral forecasting (reduces CRPS bias).
- Use EWMA for sigma: Weights recent returns more (alpha=0.94, standard for finance), proxying vol clustering without full GARCH fit (fast, no optimization).
- Apply asset-specific multipliers (from baseline notebook): Scales sigma to match observed vols (e.g., BTC high vol). Dict: {"BTC": 3.0, "ETH": 1.25, "XAU": 0.5, "SOL": 0.75}.
- Fallback: If <2 prices, sigma=0.01 (arbitrary low vol).
- Compute num_steps = time_length // time_increment (integer div for exact steps).
### Refactor simulation methods:
- simulate_crypto_price_paths: Now takes mu/sigma/num_steps; loops over 
- self.simulate_single_price_path(current_price, mu, sigma, num_steps). Returns np.array of paths.
- simulate_single_price_path: 
Remove past_prices (unused post-estimation).
Fix GBM scaling: Treat 5min as unit time (no dt or * sqrt(dt)—historical mu/sigma are per-step). Set loc = mu - 0.5 * sigma**2, scale = sigma.
Add fat tails: Use Student's t (df=5, common for finance) for shocks. Adjust scale: scale_t = scale * np.sqrt((df - 2) / df) to match variance (t has var = df/(df-2) >1).
Keep iterative exp multiplication (equivalent to cumsum on logs).
### Others:
- Efficiency: EWMA O(n) on 8640 points negligible; 100 paths x 288 steps (24h/5min) fast.
- Expected impact: EWMA + t-dist + fixes should lower CRPS vs. baseline (better tail capture, recent vol); scaling ensures asset parity.

## Conclusion
These changes elevate the baseline GBM to a more sophisticated model (EWMA vol + t-shocks + fixes) while staying lightweight and aligned with Synth requirements. 
If env allows, future iter: manual GARCH(1,1) recursion (estimate params via optimization in numpy, but adds complexity).

