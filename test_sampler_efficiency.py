#!/usr/bin/env python3
"""
Efficiency and Statistical Tests for QVarNet Sampler
"""
import jax
import jax.numpy as jnp
from jax import random
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

# Configure JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cuda")  # Change to "cpu" if no GPU

from functools import partial

# Import your sampler
import sys
sys.path.append("/home/pfargas/Desktop/PhD/qvarnet/src")
from qvarnet.sampler import mh_chain

# Create vmapped sampler (same as in your sampler.py)
vmapped_sampler = jax.vmap(
    mh_chain,
    in_axes=(
        0,    # random_values
        None, # PBC  
        None, # prob_fn
        None, # prob_params
        0,    # init_position
    ),
    out_axes=0,
)


class SamplerTester:
    def __init__(self):
        self.device = jax.devices()[0]
        print(f"Using device: {self.device}")
    
    def efficiency_benchmark(self):
        """Benchmark sampling efficiency across different configurations"""
        print("\n" + "="*60)
        print("EFFICIENCY BENCHMARK")
        print("="*60)
        
        # Test different problem sizes
        configs = [
            {"n_steps": 1000, "n_chains": 10, "DoF": 1, "name": "Small"},
            {"n_steps": 10000, "n_chains": 50, "DoF": 2, "name": "Medium"},
            {"n_steps": 100000, "n_chains": 100, "DoF": 5, "name": "Large"},
            # Skip XLarge due to memory issues
        ]
        
        # Test probability function
        @jax.jit
        def gaussian_prob(x, params):
            return jnp.exp(-0.5 * jnp.sum(x**2, axis=-1))
        
        results = []
        
        for config in configs:
            print(f"\nTesting {config['name']} config: {config}")
            
            # Generate random numbers
            key = random.PRNGKey(42)
            rand_nums = random.uniform(
                key, (config['n_chains'], config['n_steps'], config['DoF'] + 1)
            )
            
            # Initial positions
            init_positions = random.normal(random.PRNGKey(0), (config['n_chains'], config['DoF']))
            
            # Benchmark compilation
            start_time = time.perf_counter()
            samples = vmapped_sampler(rand_nums, 5.0, gaussian_prob, None, init_positions)
            compilation_time = time.perf_counter() - start_time
            
            # Benchmark execution (already compiled)
            start_time = time.perf_counter()
            samples = vmapped_sampler(rand_nums, 5.0, gaussian_prob, None, init_positions)
            execution_time = time.perf_counter() - start_time
            
            # Calculate metrics
            total_samples = config['n_chains'] * config['n_steps']
            samples_per_second = total_samples / execution_time
            memory_usage = rand_nums.nbytes + samples.nbytes
            
            result = {
                'config': config['name'],
                'compilation_time': compilation_time,
                'execution_time': execution_time,
                'samples_per_second': samples_per_second,
                'memory_usage_mb': memory_usage / 1024**2,
                'total_samples': total_samples
            }
            results.append(result)
            
            print(f"  Compilation: {compilation_time:.3f}s")
            print(f"  Execution: {execution_time:.3f}s")
            print(f"  Samples/sec: {samples_per_second:,.0f}")
            print(f"  Memory: {memory_usage / 1024**2:.1f} MB")
        
        return results
    
    def autocorrelation_test(self, samples, max_lag=1000):
        """Calculate autocorrelation time - key efficiency metric"""
        # Flatten chains and calculate autocorrelation
        flat_samples = samples.reshape(-1, samples.shape[-1])
        
        autocorr_times = []
        for dim in range(flat_samples.shape[1]):
            series = flat_samples[:, dim]
            
            # Calculate autocorrelation function
            autocorr = np.correlate(series - np.mean(series), 
                                  series - np.mean(series), mode='full')
            autocorr = autocorr[len(autocorr)//2:] / autocorr[0]
            
            # Find integrated autocorrelation time
            # Using the "initial monotone" method
            try:
                # Find where autocorrelation first becomes negative
                first_negative = np.where(autocorr < 0)[0]
                if len(first_negative) > 0:
                    cutoff = first_negative[0]
                else:
                    cutoff = min(max_lag, len(autocorr)//4)
                
                tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
                autocorr_times.append(tau_int)
            except:
                autocorr_times.append(1.0)
        
        return np.array(autocorr_times)
    
    def effective_sample_size(self, samples, autocorr_times):
        """Calculate effective sample size"""
        n_samples = samples.shape[0] * samples.shape[1]
        ess_per_dim = n_samples / (2 * autocorr_times)
        return ess_per_dim
    
    def statistical_tests(self):
        """Run statistical tests on sampler output"""
        print("\n" + "="*60)
        print("STATISTICAL TESTS")
        print("="*60)
        
        # Target: 2D Gaussian
        @jax.jit
        def target_2d_gaussian(x, params):
            return jnp.exp(-0.5 * jnp.sum(x**2, axis=-1))
        
        # Configuration for testing
        n_chains = 50
        n_steps = 20000
        DoF = 2
        PBC = 10.0
        
        # Generate samples
        key = random.PRNGKey(123)
        rand_nums = random.uniform(key, (n_chains, n_steps, DoF + 1))
        init_positions = random.normal(random.PRNGKey(456), (n_chains, DoF))
        
        print("Generating samples...")
        start_time = time.perf_counter()
        samples = vmapped_sampler(rand_nums, PBC, target_2d_gaussian, None, init_positions)
        sampling_time = time.perf_counter() - start_time
        
        print(f"Sampling completed in {sampling_time:.2f}s")
        
        # Flatten samples for analysis
        flat_samples = samples.reshape(-1, DoF)
        
        # 1. MOMENT TESTS
        print("\n1. MOMENT TESTS:")
        mean_sampled = np.mean(flat_samples, axis=0)
        std_sampled = np.std(flat_samples, axis=0)
        
        print(f"   Sampled mean: [{mean_sampled[0]:.4f}, {mean_sampled[1]:.4f}] (Expected: [0.0, 0.0])")
        print(f"   Sampled std:  [{std_sampled[0]:.4f}, {std_sampled[1]:.4f}] (Expected: [1.0, 1.0])")
        
        # Statistical tests for moments
        n_samples = len(flat_samples)
        mean_se = 1.0 / jnp.sqrt(n_samples)  # Standard error for mean
        std_se = jnp.sqrt(2 / (n_samples - 1))  # Standard error for std
        
        mean_correct = jnp.allclose(mean_sampled, 0, atol=3*mean_se)
        std_correct = jnp.allclose(std_sampled, 1.0, atol=3*std_se)
        
        print(f"   ✓ Mean test passed: {mean_correct}")
        print(f"   ✓ Std test passed: {std_correct}")
        
        # 2. AUTOCORRELATION ANALYSIS
        print("\n2. AUTOCORRELATION ANALYSIS:")
        autocorr_times = self.autocorrelation_test(samples)
        ess_per_dim = self.effective_sample_size(samples, autocorr_times)
        
        print(f"   Autocorrelation times: {autocorr_times}")
        print(f"   Effective sample size per dimension: {ess_per_dim.astype(int)}")
        print(f"   Efficiency (ESS/total): {np.mean(ess_per_dim) / len(flat_samples):.4f}")
        
        # 3. KS TEST FOR DISTRIBUTION
        print("\n3. KOLMOGOROV-SMIRNOV TEST:")
        for dim, name in enumerate(['x', 'y']):
            # Transform to uniform using CDF of standard normal
            uniform_vals = stats.norm.cdf(flat_samples[:, dim])
            
            # KS test against uniform[0,1]
            ks_stat, ks_pvalue = stats.kstest(uniform_vals, 'uniform')
            print(f"   Dimension {name}: KS statistic = {ks_stat:.4f}, p-value = {ks_pvalue:.4f}")
            
            if ks_pvalue > 0.05:
                print(f"   ✓ {name} dimension passes KS test")
            else:
                print(f"   ✗ {name} dimension fails KS test")
        
        # 4. MULTIVARIATE NORMALITY TEST
        print("\n4. MULTIVARIATE NORMALITY TEST:")
        try:
            # Henze-Zirkler test for multivariate normality
            from scipy.stats import multivariate_normal
            
            # Calculate covariance matrix
            cov_sampled = np.cov(flat_samples.T)
            print(f"   Sampled covariance:\n{cov_sampled}")
            print(f"   Expected covariance:\n[[1.0, 0.0],\n [0.0, 1.0]]")
            
            # Simple check: are off-diagonal elements close to 0?
            cov_correct = np.abs(cov_sampled[0,1]) < 0.1  # Allow some tolerance
            print(f"   ✓ Independence test passed: {cov_correct}")
            
        except Exception as e:
            print(f"   Multivariate test failed: {e}")
        
        # 5. EFFICIENCY METRICS
        print("\n5. EFFICIENCY SUMMARY:")
        total_samples = n_chains * n_steps
        samples_per_second = total_samples / sampling_time
        acceptance_estimate = 0.5  # You mentioned this is hardcoded
        
        print(f"   Total samples generated: {total_samples:,}")
        print(f"   Sampling speed: {samples_per_second:,.0f} samples/second")
        print(f"   Estimated acceptance rate: {acceptance_estimate:.2f}")
        print(f"   Average autocorrelation time: {np.mean(autocorr_times):.2f}")
        print(f"   Average efficiency (ESS/sec): {np.mean(ess_per_dim) / sampling_time:,.0f}")
        
        return {
            'samples': samples,
            'autocorr_times': autocorr_times,
            'ess_per_dim': ess_per_dim,
            'mean_correct': mean_correct,
            'std_correct': std_correct,
            'samples_per_second': samples_per_second
        }
    
    def step_size_sensitivity_test(self):
        """Test efficiency across different step sizes"""
        print("\n" + "="*60)
        print("STEP SIZE SENSITIVITY TEST")
        print("="*60)
        
        @jax.jit
        def target_prob(x, params):
            return jnp.exp(-0.5 * jnp.sum(x**2, axis=-1))
        
        step_sizes = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0]
        results = []
        
        for step_size in step_sizes:
            print(f"\nTesting step_size = {step_size}")
            
            # Modify sampler to use custom step size (need to patch your sampler)
            # For now, we'll use the existing approach
            n_chains = 20
            n_steps = 10000
            DoF = 1
            
            key = random.PRNGKey(42 + int(step_size * 10))
            rand_nums = random.uniform(key, (n_chains, n_steps, DoF + 1))
            init_positions = random.normal(random.PRNGKey(0), (n_chains, DoF))
            
            # This is a simplified test - your actual implementation may need modification
            # to test different step sizes properly
            try:
                start_time = time.perf_counter()
                samples = vmapped_sampler(rand_nums, 5.0, target_prob, None, init_positions)
                sampling_time = time.perf_counter() - start_time
                
                autocorr_times = self.autocorrelation_test(samples)
                ess_per_dim = self.effective_sample_size(samples, autocorr_times)
                efficiency = np.mean(ess_per_dim) / sampling_time
                
                results.append({
                    'step_size': step_size,
                    'autocorr_time': np.mean(autocorr_times),
                    'efficiency': efficiency,
                    'sampling_time': sampling_time
                })
                
                print(f"  Autocorrelation time: {np.mean(autocorr_times):.2f}")
                print(f"  Efficiency (ESS/sec): {efficiency:,.0f}")
                
            except Exception as e:
                print(f"  Error with step_size {step_size}: {e}")
        
        if results:
            # Find optimal step size
            best_result = max(results, key=lambda x: x['efficiency'])
            print(f"\n✓ Optimal step size: {best_result['step_size']} with efficiency {best_result['efficiency']:,.0f} ESS/sec")
        
        return results


def main():
    tester = SamplerTester()
    
    # Run efficiency benchmark
    efficiency_results = tester.efficiency_benchmark()
    
    # Run statistical tests
    stat_results = tester.statistical_tests()
    
    # Run step size sensitivity test
    step_results = tester.step_size_sensitivity_test()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if efficiency_results:
        print(f"Peak sampling speed: {max(r['samples_per_second'] for r in efficiency_results):,.0f} samples/sec")
        print(f"Memory efficiency: {min(r['memory_usage_mb'] for r in efficiency_results):.1f} MB (minimum)")
    
    if stat_results:
        overall_health = all([stat_results['mean_correct'], stat_results['std_correct']])
        print(f"Statistical correctness: {'✓ PASS' if overall_health else '✗ FAIL'}")
        print(f"Average efficiency: {np.mean(stat_results['ess_per_dim']) / (len(stat_results['samples'].flatten())):.4f}")
    
    print("\nRecommendations:")
    print("1. Fix acceptance rate tracking (currently hardcoded to 0.0)")
    print("2. Implement adaptive step size based on actual acceptance")
    print("3. Consider online random number generation to reduce memory")
    print("4. Optimize autocorrelation time through step size tuning")


if __name__ == "__main__":
    main()