"""
Auto-run experiments with different configurations
Supports parallel execution and configurable experiment programs
"""

import argparse
import subprocess
import sys
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path

# ============ CONFIGURATION ============
experiment_configs = {
    'context_lens': [64, 128, 256],
    'decode_lens': [64, 128, 256],
    'draft_decode_lens': [5],  # Can add more values like [3, 5, 7]
    'nb_samples': 4,  # Number of samples per configuration
    'proj_name': 'eagle3_sim_v1',  # Project name for output directory
    'accept_prob': 0.8  # Token acceptance probability in sampling
}

# Default settings
DEFAULT_PROGRAM = "main_speculative_v5.py"
DEFAULT_PARALLEL_JOBS = 10
# =======================================


def run_single_experiment(args_tuple):
    """
    Run a single experiment configuration
    Returns: (config_dict, success_bool, duration_seconds, error_msg)
    """
    prefill_len, decode_len, draft_len, nb_samples, proj_name, accept_prob, program = args_tuple

    config_name = f"prefill{prefill_len}_decode{decode_len}_draft{draft_len}"

    cmd = [
        sys.executable,
        program,
        "--prefill_len", str(prefill_len),
        "--decode_len", str(decode_len),
        "--draft_len_per_verify", str(draft_len),
        "--nb_samples", str(nb_samples),
        "--proj_name", str(proj_name),
        "--accept_prob", str(accept_prob)
    ]

    start_time = datetime.now()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=36000  # 10 hour timeout
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            'config': config_name,
            'prefill_len': prefill_len,
            'decode_len': decode_len,
            'draft_len': draft_len,
            'nb_samples': nb_samples,
            'status': 'success',
            'duration': duration
        }, True, duration, None

    except subprocess.TimeoutExpired:
        error_msg = "Timeout after 10 hours"
        return {
            'config': config_name,
            'prefill_len': prefill_len,
            'decode_len': decode_len,
            'draft_len': draft_len,
            'nb_samples': nb_samples,
            'status': 'timeout',
            'error': error_msg
        }, False, 0, error_msg

    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code {e.returncode}: {e.stderr[:500]}"
        return {
            'config': config_name,
            'prefill_len': prefill_len,
            'decode_len': decode_len,
            'draft_len': draft_len,
            'nb_samples': nb_samples,
            'status': 'failed',
            'error': error_msg
        }, False, 0, error_msg


def run_experiments_parallel(program, max_workers=10, save_results=True):
    """Run all experiment combinations in parallel"""

    # Generate all experiment combinations
    experiments = []
    for prefill_len, decode_len, draft_len in product(
        experiment_configs['context_lens'],
        experiment_configs['decode_lens'],
        experiment_configs['draft_decode_lens']
    ):
        experiments.append((
            prefill_len,
            decode_len,
            draft_len,
            experiment_configs['nb_samples'],
            experiment_configs['proj_name'],
            experiment_configs['accept_prob'],
            program
        ))

    total_configs = len(experiments)

    print(f"{'='*80}")
    print(f"Starting parallel auto-run with {total_configs} configurations")
    print(f"Program: {program}")
    print(f"Parallel workers: {max_workers}")
    print(f"Context lengths: {experiment_configs['context_lens']}")
    print(f"Decode lengths: {experiment_configs['decode_lens']}")
    print(f"Draft decode lengths: {experiment_configs['draft_decode_lens']}")
    print(f"Samples per config: {experiment_configs['nb_samples']}")
    print(f"Project name: {experiment_configs['proj_name']}")
    print(f"Accept probability: {experiment_configs['accept_prob']}")
    print(f"{'='*80}\n")

    results = []
    successful = 0
    failed = 0

    # Run experiments in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_exp = {
            executor.submit(run_single_experiment, exp): exp
            for exp in experiments
        }

        # Process completed jobs
        for i, future in enumerate(as_completed(future_to_exp), 1):
            exp = future_to_exp[future]
            prefill_len, decode_len, draft_len, _, _, _, _ = exp

            try:
                result_dict, success, duration, error = future.result()
                results.append(result_dict)

                if success:
                    successful += 1
                    print(f"[{i}/{total_configs}] ✓ prefill={prefill_len}, decode={decode_len}, "
                          f"draft={draft_len} - Completed in {duration:.2f}s")
                else:
                    failed += 1
                    print(f"[{i}/{total_configs}] ✗ prefill={prefill_len}, decode={decode_len}, "
                          f"draft={draft_len} - Failed: {error}")

            except Exception as e:
                failed += 1
                print(f"[{i}/{total_configs}] ✗ prefill={prefill_len}, decode={decode_len}, "
                      f"draft={draft_len} - Exception: {str(e)}")
                results.append({
                    'config': f"prefill{prefill_len}_decode{decode_len}_draft{draft_len}",
                    'status': 'exception',
                    'error': str(e)
                })

    # Print summary
    total_duration = sum(r.get('duration', 0) for r in results)
    print(f"\n{'='*80}")
    print(f"Auto-run completed!")
    print(f"Total experiments: {total_configs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total computation time: {total_duration:.2f}s ({total_duration/60:.2f}min)")
    if successful > 0:
        print(f"Average time per experiment: {total_duration/successful:.2f}s")
    print(f"{'='*80}")

    # Save results to JSON
    if save_results:
        # Save logs under outputs/proj_name/experiment_logs/
        log_dir = Path("outputs") / experiment_configs['proj_name'] / "experiment_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = log_dir / f"results_{timestamp}.json"

        output = {
            'program': program,
            'timestamp': timestamp,
            'config': experiment_configs,
            'max_workers': max_workers,
            'results': results,
            'summary': {
                'total': total_configs,
                'successful': successful,
                'failed': failed,
                'total_duration_seconds': total_duration,
                'average_duration_seconds': total_duration / successful if successful > 0 else 0
            }
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    return results, successful, failed


def run_experiments_sequential(program):
    """Run all experiment combinations sequentially (original behavior)"""

    total_configs = (
        len(experiment_configs['context_lens']) *
        len(experiment_configs['decode_lens']) *
        len(experiment_configs['draft_decode_lens'])
    )

    print(f"Starting sequential auto-run with {total_configs} configurations")
    print(f"Program: {program}")
    print(f"Context lengths: {experiment_configs['context_lens']}")
    print(f"Decode lengths: {experiment_configs['decode_lens']}")
    print(f"Draft decode lengths: {experiment_configs['draft_decode_lens']}")
    print(f"Samples per config: {experiment_configs['nb_samples']}")
    print(f"Project name: {experiment_configs['proj_name']}")
    print(f"Accept probability: {experiment_configs['accept_prob']}\n")

    successful = 0
    failed = 0

    for i, (prefill_len, decode_len, draft_len) in enumerate(product(
        experiment_configs['context_lens'],
        experiment_configs['decode_lens'],
        experiment_configs['draft_decode_lens']
    ), 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{total_configs}] Running: prefill={prefill_len}, decode={decode_len}, "
              f"draft={draft_len}, samples={experiment_configs['nb_samples']}")
        print(f"{'='*80}")

        _, success, duration, error = run_single_experiment((
            prefill_len,
            decode_len,
            draft_len,
            experiment_configs['nb_samples'],
            experiment_configs['proj_name'],
            experiment_configs['accept_prob'],
            program
        ))

        if success:
            successful += 1
            print(f"✓ Completed successfully in {duration:.2f}s")
        else:
            failed += 1
            print(f"✗ Failed: {error}")

    print(f"\n{'='*80}")
    print(f"Auto-run completed!")
    print(f"Successful: {successful}/{total_configs}")
    print(f"Failed: {failed}/{total_configs}")
    print(f"{'='*80}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-run experiments with different configurations"
    )
    parser.add_argument(
        "--program",
        type=str,
        default=DEFAULT_PROGRAM,
        help=f"Python program to run (default: {DEFAULT_PROGRAM})"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=DEFAULT_PARALLEL_JOBS,
        help=f"Number of parallel jobs (default: {DEFAULT_PARALLEL_JOBS}). Set to 1 for sequential."
    )
    parser.add_argument(
        "--proj-name",
        type=str,
        default=None,
        help=f"Project name for output directory (default: {experiment_configs['proj_name']})"
    )
    parser.add_argument(
        "--accept-prob",
        type=float,
        default=None,
        help=f"Token acceptance probability (default: {experiment_configs['accept_prob']})"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Override config values if provided via command line
    if args.proj_name is not None:
        experiment_configs['proj_name'] = args.proj_name
    if args.accept_prob is not None:
        experiment_configs['accept_prob'] = args.accept_prob

    # Check if program exists
    if not os.path.exists(args.program):
        print(f"Error: Program '{args.program}' not found!")
        sys.exit(1)

    # Run experiments
    if args.parallel > 1:
        run_experiments_parallel(
            program=args.program,
            max_workers=args.parallel,
            save_results=not args.no_save
        )
    else:
        run_experiments_sequential(program=args.program)


if __name__ == "__main__":
    main()
