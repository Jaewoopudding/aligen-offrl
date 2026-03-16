import os
import json
import glob
import pandas as pd
import numpy as np
import sys

def load_data(base_dir: str) -> pd.DataFrame:
    """Load and parse evaluation data from all experiments."""
    records = []
    pattern = os.path.join(base_dir, "exp", "aligen", "*_adaptive_sweep", "*", "flags.json")
    flags_paths = glob.glob(pattern)
    
    for flag_path in flags_paths:
        run_dir = os.path.dirname(flag_path)
        eval_path = os.path.join(run_dir, "eval.csv")
        
        if not os.path.exists(eval_path):
            continue
            
        with open(flag_path, 'r') as f:
            try:
                flags = json.load(f)
            except json.JSONDecodeError:
                continue
                
        env_name = flags.get("env_name", "unknown")
        seed = flags.get("seed", -1)
        agent_config = flags.get("agent", {})
        alpha = agent_config.get("alpha", None)
        temp_method = agent_config.get("temp_method", "unknown")
        
        try:
            df = pd.read_csv(eval_path)
        except Exception:
            continue
            
        ret_col = "evaluation/episode.normalized_return"
        step_col = "step"
        if ret_col not in df.columns or step_col not in df.columns:
            continue
            
        df = df.dropna(subset=[step_col, ret_col])
        
        for _, row in df.iterrows():
            records.append({
                "temp_method": temp_method,
                "env_name": env_name,
                "alpha": alpha,
                "seed": seed,
                "step": int(row[step_col]),
                "return": row[ret_col]
            })
            
    return pd.DataFrame(records)

def save_and_print_summary(df: pd.DataFrame, out_dir: str):
    """Compute and print mean/std, and save to files."""
    if df.empty:
        print("No valid data found.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Prepare statistics
    grouped = df.groupby(["temp_method", "env_name", "alpha", "step"])["return"]
    stats = grouped.agg(["mean", "std", "count"]).reset_index()
    
    # Final step
    max_steps = stats.groupby(["temp_method", "env_name", "alpha"])["step"].max().reset_index()
    final_stats = pd.merge(stats, max_steps, on=["temp_method", "env_name", "alpha", "step"])
    
    # Save raw stats to CSV
    stats.to_csv(os.path.join(out_dir, "all_steps_stats.csv"), index=False)
    final_stats.to_csv(os.path.join(out_dir, "final_results_stats.csv"), index=False)

    # 2. Print & Save text summary
    summary_txt_path = os.path.join(out_dir, "summary.txt")
    final_best_txt_path = os.path.join(out_dir, "final_results.txt")
    
    with open(summary_txt_path, 'w') as f, open(final_best_txt_path, 'w') as f_best:
        # Define a helper to print both to terminal and file
        def write_out(msg):
            print(msg)
            f.write(msg + '\n')
            
        def write_best(msg):
            f_best.write(msg + '\n')
            
        write_out("="*80)
        write_out("=== FINAL RESULTS SUMMARY (Last Step) ===")
        write_out("="*80)
        
        write_best("="*80)
        write_best("=== BEST HYPERPARAMETER PER ENVIRONMENT (FINAL LAST STEP) ===")
        write_best("="*80)
        
        for method in sorted(final_stats["temp_method"].unique()):
            write_out(f"\n[ temp_method: {method} ]")
            write_out(f"{'Environment':<30} | {'Alpha':<5} | {'Step':<8} | {'Mean Return':<11} | {'Std Dev':<7} | {'Seeds'}")
            write_out("-" * 80)
            
            write_best(f"\n[ temp_method: {method} ]")
            write_best(f"{'Environment':<30} | {'Alpha':<5} | {'Step':<8} | {'Mean Return':<11} | {'Std Dev':<7} | {'Seeds'}")
            write_best("-" * 80)
            
            method_df = final_stats[final_stats["temp_method"] == method]
            for env in sorted(method_df["env_name"].unique()):
                env_df = method_df[method_df["env_name"] == env]
                
                # Find maximum mean return for this environment
                max_mean = env_df["mean"].max()
                
                for _, row in env_df.sort_values("alpha").iterrows():
                    std_str = f"{row['std']:.2f}" if not pd.isna(row['std']) else "N/A"
                    
                    # Highlight if this row contains the maximum mean
                    is_best = row["mean"] == max_mean
                    highlight = " 🌟 (BEST)" if is_best else ""
                    
                    line = f"{env:<30} | {row['alpha']:<5} | {int(row['step']):<8} | {row['mean']:<11.2f} | {std_str:<7} | {int(row['count'])}"
                    write_out(line + highlight)
                    
                    if is_best:
                        write_best(line)

        write_out("\n" + "="*80)
        write_out("=== TRAINING TREND (Mean ± Std computed over seeds) ===")
        write_out("="*80)
        
        for method in sorted(stats["temp_method"].unique()):
            write_out(f"\n[ temp_method: {method} ]")
            method_df = stats[stats["temp_method"] == method]
            for env in sorted(method_df["env_name"].unique()):
                write_out(f"  Env: {env}")
                env_df = method_df[method_df["env_name"] == env]
                for alpha in sorted(env_df["alpha"].unique()):
                    alpha_df = env_df[env_df["alpha"] == alpha]
                    
                    trend_strs = []
                    for _, row in alpha_df.sort_values("step").iterrows():
                        std_val = row['std'] if not pd.isna(row['std']) else 0.0
                        trend_strs.append(f"{int(row['step'])//1000}k({row['mean']:.1f}±{std_val:.1f})")
                    
                    write_out(f"    Alpha: {alpha:<4} -> " + ", ".join(trend_strs))
                    
    print(f"\n✅ All results and CSV summaries have been saved to the '{out_dir}' directory.")

if __name__ == "__main__":
    base_dir = "/home/jaewoo/aligen-offrl"
    out_dir = os.path.join(base_dir, "results")
    df = load_data(base_dir)
    save_and_print_summary(df, out_dir)
