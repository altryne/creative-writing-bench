# creative_writing_bench.py

"""
Main entry for the Creative Writing Benchmark with iteration-based generation.
"""
import argparse
import sys
import signal
import logging
from dotenv import load_dotenv
from utils.logging_setup import setup_logging, get_verbosity
from core.benchmark import run_eq_bench_creative

load_dotenv()

def signal_handler(signum, frame):
    print(f"\n[DEBUG] Signal {signum} caught! Stopping gracefully.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run Creative Writing Benchmark (with iterations).")
    parser.add_argument("--test-model", required=True, help="The model name or identifier for the test model.")
    parser.add_argument("--judge-model", required=True, help="The model name or identifier for the judge model.")
    parser.add_argument("--runs-file", default="creative_bench_runs.json", help="File where run data is stored.")
    parser.add_argument("--run-id", help="Optional: Resume or create a run with this ID prefix")
    parser.add_argument("--threads", type=int, default=4, help="Number of parallel threads.")
    parser.add_argument("--verbosity", choices=['DEBUG','INFO','WARNING','ERROR','CRITICAL'], default="INFO")
    parser.add_argument("--redo-judging", action="store_true", default=False, help="Re-run the judge step on existing items.")
    parser.add_argument("--creative-prompts-file", default="data/creative_writing_prompts_v3.json")
    parser.add_argument("--criteria-file", default="data/creative_writing_criteria.txt")
    parser.add_argument("--negative-criteria-file", default="data/negative_criteria.txt")
    parser.add_argument("--judge-prompt-file", default="data/creative_writing_judging_prompt.txt")
    parser.add_argument("--save-interval", type=int, default=2, help="How often to save partial progress.")
    parser.add_argument("--iterations", type=int, default=1, help="How many iteration passes to run (one seed per iteration).")

    args = parser.parse_args()
    setup_logging(get_verbosity(args.verbosity))

    # Hook signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    run_key = run_eq_bench_creative(
        test_model=args.test_model,
        judge_model=args.judge_model,
        runs_file=args.runs_file,
        num_threads=args.threads,
        run_id=args.run_id,
        creative_prompts_file=args.creative_prompts_file,
        creative_criteria_file=args.criteria_file,
        negative_criteria_file=args.negative_criteria_file,
        judge_prompt_file=args.judge_prompt_file,
        redo_judging=args.redo_judging,
        save_interval=args.save_interval,
        iterations=args.iterations
    )

    logging.info(f"Creative writing benchmark completed. Run key: {run_key}")
    print(f"\nCreative writing benchmark completed. Run key: {run_key}")

if __name__ == "__main__":
    main()
