"""
Script 08: Generate Final Report Summary

Consolidates all experiment results into a comprehensive summary.
Generates statistics for the final report notebook.

Run: python scripts/08_generate_final_report.py
"""

import sys
import json
from pathlib import Path

def main():
    print("="*80)
    print("Final Report Summary Generator")
    print("="*80)
    
    root = Path(".")
    
    # 1. Training Info
    print("\n" + "="*80)
    print("1. TRAINING INFORMATION")
    print("="*80)
    
    training_info_file = root / "models/finetuned/training_info.json"
    if training_info_file.exists():
        with open(training_info_file) as f:
            training_info = json.load(f)
        
        print(f"\nModel: {training_info['base_model']}")
        print(f"Batch size: {training_info['batch_size']}")
        print(f"Epochs: {training_info['num_epochs']}")
        print(f"Training pairs: {training_info['training_pairs']:,}")
        print(f"Total steps: {training_info['total_steps']}")
        print(f"Training time: {training_info['training_time_hours']:.2f} hours")
        print(f"\nLoss:")
        print(f"  Initial: {training_info['initial_loss']:.4f}")
        print(f"  Final: {training_info['final_loss']:.4f}")
        print(f"  Reduction: {training_info['loss_reduction']:.4f} ({training_info['loss_reduction']/training_info['initial_loss']*100:.1f}%)")
    else:
        print("⚠ Training info not found")
    
    # 2. Loss History Summary
    print("\n" + "="*80)
    print("2. LOSS HISTORY")
    print("="*80)
    
    loss_file = root / "models/finetuned/loss_history.json"
    if loss_file.exists():
        with open(loss_file) as f:
            loss_history = json.load(f)
        
        print(f"\nTotal recorded steps: {len(loss_history)}")
        
        # Calculate per-epoch statistics
        epochs = {}
        for entry in loss_history:
            epoch = entry['epoch']
            if epoch not in epochs:
                epochs[epoch] = []
            epochs[epoch].append(entry['loss'])
        
        print(f"\nPer-epoch average loss:")
        for epoch_num in sorted(epochs.keys()):
            losses = epochs[epoch_num]
            avg_loss = sum(losses) / len(losses)
            print(f"  Epoch {epoch_num + 1}: {avg_loss:.4f} ({len(losses)} steps)")
    else:
        print("⚠ Loss history not found")
    
    # 3. Baseline Metrics
    print("\n" + "="*80)
    print("3. BASELINE EVALUATION")
    print("="*80)
    
    baseline_file = root / "results/baseline_metrics_test.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        print(f"\nRecall@10: {baseline['recall@10']:.4f}")
        print(f"MRR@10: {baseline['mrr@10']:.4f}")
        print(f"nDCG@10: {baseline['ndcg@10']:.4f}")
    else:
        print("⚠ Baseline metrics not found")
    
    # 4. Fine-tuned Metrics
    print("\n" + "="*80)
    print("4. FINE-TUNED EVALUATION")
    print("="*80)
    
    finetuned_file = root / "results/finetuned_metrics_test.json"
    if finetuned_file.exists():
        with open(finetuned_file) as f:
            finetuned = json.load(f)
        
        print(f"\nRecall@10: {finetuned['recall@10']:.4f}")
        print(f"MRR@10: {finetuned['mrr@10']:.4f}")
        print(f"nDCG@10: {finetuned['ndcg@10']:.4f}")
        
        # Calculate improvements
        if baseline_file.exists():
            print(f"\nImprovements:")
            print(f"  Recall@10: +{(finetuned['recall@10'] - baseline['recall@10']):.4f} ({(finetuned['recall@10'] / baseline['recall@10'] - 1)*100:.1f}%)")
            print(f"  MRR@10: +{(finetuned['mrr@10'] - baseline['mrr@10']):.4f} ({(finetuned['mrr@10'] / baseline['mrr@10'] - 1)*100:.1f}%)")
            print(f"  nDCG@10: +{(finetuned['ndcg@10'] - baseline['ndcg@10']):.4f} ({(finetuned['ndcg@10'] / baseline['ndcg@10'] - 1)*100:.1f}%)")
    else:
        print("⚠ Fine-tuned metrics not found")
    
    # 5. Bonus Experiments
    print("\n" + "="*80)
    print("5. BONUS EXPERIMENTS")
    print("="*80)
    
    bonus_dir = root / "results/bonus"
    if bonus_dir.exists():
        # Experiment 1: Function name impact
        exp1_file = bonus_dir / "experiment1_function_name_impact.json"
        if exp1_file.exists():
            with open(exp1_file) as f:
                exp1 = json.load(f)
            
            print("\nExperiment 1: Function Name Impact")
            if 'with_names' in exp1 and 'without_names' in exp1:
                print(f"  With names: nDCG@10 = {exp1['with_names']['ndcg@10']:.4f}")
                print(f"  Without names: nDCG@10 = {exp1['without_names']['ndcg@10']:.4f}")
                diff = exp1['with_names']['ndcg@10'] - exp1['without_names']['ndcg@10']
                print(f"  Difference: {diff:.4f} ({abs(diff)/exp1['without_names']['ndcg@10']*100:.1f}%)")
        
        # Experiment 2: Query type analysis
        exp2_file = bonus_dir / "experiment2_query_type_analysis.json"
        if exp2_file.exists():
            with open(exp2_file) as f:
                exp2 = json.load(f)
            
            print("\nExperiment 2: Query Type Analysis")
            if 'by_query_type' in exp2:
                for qtype, stats in exp2['by_query_type'].items():
                    print(f"  {qtype}: {stats['found']} found / {stats['total']} total ({stats['found']/stats['total']*100:.1f}%)")
        
        # Experiment 3: Code complexity
        exp3_file = bonus_dir / "experiment3_code_complexity.json"
        if exp3_file.exists():
            with open(exp3_file) as f:
                exp3 = json.load(f)
            
            print("\nExperiment 3: Code Complexity")
            if 'by_code_length' in exp3:
                for length_range, stats in sorted(exp3['by_code_length'].items()):
                    print(f"  {length_range} lines: {stats['found']}/{stats['total']} ({stats['found']/stats['total']*100:.0f}%), avg_rank={stats.get('avg_rank', 'N/A')}")
    else:
        print("⚠ Bonus experiments not found")
    
    # 6. FAISS Hyperparameters
    print("\n" + "="*80)
    print("6. FAISS HYPERPARAMETER EXPERIMENTS")
    print("="*80)
    
    faiss_dir = root / "results/faiss_hyperparams"
    if faiss_dir.exists() and (faiss_dir / "all_experiments.json").exists():
        with open(faiss_dir / "all_experiments.json") as f:
            faiss_exp = json.load(f)
        
        # Experiment 1: Index types
        if 'experiment1_index_types' in faiss_exp:
            print("\nExperiment 1: Index Types")
            for config, metrics in faiss_exp['experiment1_index_types'].items():
                print(f"  {config}: nDCG@10={metrics['ndcg@10']:.4f}, search={metrics['search_time_total']:.4f}s")
        
        # Experiment 2: nlist sweep
        if 'experiment2_nlist_sweep' in faiss_exp:
            print("\nExperiment 2: nlist Parameter Sweep")
            for config, metrics in sorted(faiss_exp['experiment2_nlist_sweep'].items()):
                nlist = config.split('_')[1].replace('nlist', '')
                print(f"  nlist={nlist}: nDCG@10={metrics['ndcg@10']:.4f}, search={metrics['search_time_total']:.4f}s")
        
        # Experiment 3: nprobe sweep
        if 'experiment3_nprobe_sweep' in faiss_exp:
            print("\nExperiment 3: nprobe Parameter Sweep")
            for config, metrics in sorted(faiss_exp['experiment3_nprobe_sweep'].items(), 
                                         key=lambda x: int(x[0].split('_')[2].replace('nprobe', ''))):
                nprobe = config.split('_')[2].replace('nprobe', '')
                print(f"  nprobe={nprobe}: nDCG@10={metrics['ndcg@10']:.4f}, search={metrics['search_time_total']:.4f}s")
    else:
        print("⚠ FAISS hyperparameter experiments not found (may still be running)")
    
    # 7. File Summary
    print("\n" + "="*80)
    print("7. PROJECT FILES")
    print("="*80)
    
    print("\nGenerated files:")
    
    important_files = [
        "models/finetuned/loss_history.json",
        "models/finetuned/training_info.json",
        "results/baseline_metrics_test.json",
        "results/finetuned_metrics_test.json",
        "results/bonus/experiment1_function_name_impact.json",
        "results/bonus/experiment2_query_type_analysis.json",
        "results/bonus/experiment3_code_complexity.json",
        "results/faiss_hyperparams/all_experiments.json",
        "results/faiss_hyperparams/FAISS_EXPERIMENTS_SUMMARY.md",
    ]
    
    for filepath in important_files:
        full_path = root / filepath
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filepath} (missing)")
    
    print("\n" + "="*80)
    print("✓ Report generation complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
