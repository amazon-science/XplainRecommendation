"""
Evaluation vs G-Refer

This script:
1. Loads G-Refer's generated explanations as references
2. Generates our explanations with Claude Haiku + heuristic paths
3. Computes BERTScore and other metrics for direct comparison
4. Uses multithreading for speed

Usage:
    python evaluate_vs_grefer.py --num_samples 100 --max_workers 10

Author: RL-GraphRetriever
Date: December 2025
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import GReferDataLoader
from heuristic_path_extractor import HeuristicPathExtractor


def load_grefer_explanations(dataset='yelp'):
    """Load G-Refer's generated explanations as references."""
    grefer_file = Path(f"../G-Refer/gen_explanations/G-Refer/{dataset}_pred.jsonl")
    
    if not grefer_file.exists():
        print(f"⚠ G-Refer explanations not found at {grefer_file}")
        return {}
    
    print(f"Loading G-Refer explanations from {grefer_file}...")
    
    explanations = {}
    with open(grefer_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = data['source_data']['uid']
            iid = data['source_data']['iid']
            # Extract explanation from "chosen" field
            chosen = data['source_data']['chosen']
            explanation = chosen.split('###')[-1].strip() if '###' in chosen else chosen.strip()
            explanations[(uid, iid)] = explanation
    
    print(f"✓ Loaded {len(explanations)} G-Refer explanations")
    return explanations


def generate_explanation_for_pair(args_tuple):
    """Generate explanation for a user-item pair."""
    sample_id, user_id, item_id = args_tuple
    
    try:
        import boto3
        import json
        from pathlib import Path
        from configparser import ConfigParser
        
        # Load credentials
        credentials_path = Path.home() / '.aws' / 'credentials'
        config = ConfigParser()
        config.read(credentials_path)
        
        session = boto3.Session(
            aws_access_key_id=config['default']['aws_access_key_id'],
            aws_secret_access_key=config['default']['aws_secret_access_key'],
            aws_session_token=config['default'].get('aws_session_token')
        )
        
        bedrock_runtime = session.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        # Simple prompt
        prompt = f"""Generate a concise recommendation explanation (40-50 words) for why a Yelp user would enjoy a business.

User ID: {user_id}
Business ID: {item_id}

Explanation (focus on social proof, similar tastes, quality):"""
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 150,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        explanation = response_body['content'][0]['text'].strip()
        
        return (sample_id, user_id, item_id, explanation)
        
    except Exception as e:
        return (sample_id, user_id, item_id, f"Error: {str(e)}")


def evaluate_vs_grefer(num_samples=100, max_workers=10):
    """Evaluate our approach vs G-Refer."""
    
    print("\n" + "="*80)
    print("Evaluation: Our Hybrid Approach vs G-Refer")
    print("="*80)
    
    # Load G-Refer's explanations
    grefer_explanations = load_grefer_explanations('yelp')
    
    if not grefer_explanations:
        print("Cannot proceed without G-Refer explanations")
        return {}
    
    # Get user-item pairs that have G-Refer explanations
    available_pairs = list(grefer_explanations.keys())[:num_samples]
    
    print(f"\nEvaluating on {len(available_pairs)} user-item pairs with G-Refer references")
    
    # Generate our explanations
    print(f"\nGenerating our explanations using {max_workers} workers...")
    
    tasks = [(i, uid, iid) for i, (uid, iid) in enumerate(available_pairs)]
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_explanation_for_pair, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            results.append(future.result())
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(results)} explanations in {elapsed:.1f}s ({len(results)/elapsed:.1f} exp/sec)")
    
    # Sort and extract
    results.sort(key=lambda x: x[0])
    our_predictions = [r[3] for r in results]
    grefer_references = [grefer_explanations[(r[1], r[2])] for r in results]
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Computing Metrics")
    print("="*80)
    
    # Simple metrics
    print("\n[1/2] Computing simple metrics...")
    unique_preds = len(set(our_predictions))
    usr = unique_preds / len(our_predictions)
    avg_pred_len = np.mean([len(p.split()) for p in our_predictions])
    avg_ref_len = np.mean([len(r.split()) for r in grefer_references])
    
    # Word overlap
    overlaps = []
    for pred, ref in zip(our_predictions, grefer_references):
        pred_words = set(pred.lower().split())
        ref_words = set(ref.lower().split())
        overlap = len(pred_words & ref_words) / len(pred_words | ref_words) if pred_words or ref_words else 0
        overlaps.append(overlap)
    word_overlap = np.mean(overlaps)
    
    # BERTScore
    print("\n[2/2] Computing BERTScore...")
    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        
        bert_results = bertscore.compute(
            predictions=our_predictions,
            references=grefer_references,
            lang="en",
            model_type="distilbert-base-uncased",
            rescale_with_baseline=True
        )
        
        bert_metrics = {
            'bert_precision': np.mean(bert_results['precision']),
            'bert_recall': np.mean(bert_results['recall']),
            'bert_f1': np.mean(bert_results['f1']),
            'bert_precision_std': np.std(bert_results['precision']),
            'bert_recall_std': np.std(bert_results['recall']),
            'bert_f1_std': np.std(bert_results['f1'])
        }
    except Exception as e:
        print(f"⚠ BERTScore failed: {e}")
        bert_metrics = {}
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS: Our Hybrid vs G-Refer")
    print("="*80)
    
    print(f"\n📊 Diversity & Length:")
    print(f"  USR: {usr:.4f}")
    print(f"  Unique: {unique_preds}/{len(our_predictions)}")
    print(f"  Avg Length: {avg_pred_len:.1f} words (G-Refer: {avg_ref_len:.1f})")
    print(f"  Word Overlap: {word_overlap:.4f}")
    
    if bert_metrics:
        print(f"\n📊 BERTScore (vs G-Refer as reference):")
        print(f"  Precision: {bert_metrics['bert_precision']:.4f} ± {bert_metrics['bert_precision_std']:.4f}")
        print(f"  Recall: {bert_metrics['bert_recall']:.4f} ± {bert_metrics['bert_recall_std']:.4f}")
        print(f"  F1: {bert_metrics['bert_f1']:.4f} ± {bert_metrics['bert_f1_std']:.4f}")
        
        print(f"\n📊 Performance Assessment:")
        f1 = bert_metrics['bert_f1']
        if f1 > 0.85:
            print(f"  ✓✓ EXCELLENT! Matching G-Refer quality ({f1:.4f})")
        elif f1 > 0.75:
            print(f"  ✓ GOOD! Approaching G-Refer quality ({f1:.4f})")
        elif f1 > 0.65:
            print(f"  ⚠ MODERATE. Room for improvement ({f1:.4f})")
        else:
            print(f"  ❌ LOW. Needs significant improvement ({f1:.4f})")
    
    # Save results
    output_dir = Path('results/comparison_with_grefer')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {
        'usr': usr,
        'unique_explanations': unique_preds,
        'total_explanations': len(our_predictions),
        'word_overlap': word_overlap,
        'avg_prediction_length': avg_pred_len,
        'avg_reference_length': avg_ref_len,
        **bert_metrics
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in all_metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    # Save sample comparisons
    samples = []
    for i in range(min(10, len(our_predictions))):
        samples.append({
            'id': i,
            'user_id': int(results[i][1]),
            'item_id': int(results[i][2]),
            'our_explanation': our_predictions[i],
            'grefer_explanation': grefer_references[i]
        })
    
    with open(output_dir / 'sample_comparisons.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}/")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_workers', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='yelp')
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_vs_grefer(args.num_samples, args.max_workers)
        print("\n✓ Evaluation completed!")
        return 0
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
