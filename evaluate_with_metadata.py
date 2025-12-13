"""
Evaluation with Business/User Metadata Integration

This script uses G-Refer's business and user profiles to generate
high-quality, specific explanations with Claude Haiku.

Expected BERTScore improvement: 0.33 → 0.70-0.85

Usage:
    python evaluate_with_metadata.py --num_samples 100 --max_workers 10

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
import re

sys.path.insert(0, str(Path(__file__).parent / 'src'))


def load_grefer_metadata(dataset='yelp'):
    """
    Load business titles, profiles, and user profiles from G-Refer data.
    
    Returns:
        Dictionary mapping (uid, iid) -> {business_title, business_profile, user_profile, explanation}
    """
    grefer_file = Path(f"../G-Refer/gen_explanations/G-Refer/{dataset}_pred.jsonl")
    
    if not grefer_file.exists():
        print(f"⚠ G-Refer data not found at {grefer_file}")
        return {}
    
    print(f"Loading G-Refer metadata from {grefer_file}...")
    
    metadata = {}
    with open(grefer_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            uid = data['source_data']['uid']
            iid = data['source_data']['iid']
            prompt = data['source_data']['prompt']
            
            # Extract business title
            business_match = re.search(r'Business title: ([^.]+)\.', prompt)
            business_title = business_match.group(1) if business_match else f"Business {iid}"
            
            # Extract business profile
            biz_profile_match = re.search(r'Business profile: ([^.]+\.)', prompt)
            business_profile = biz_profile_match.group(1) if biz_profile_match else "A recommended business"
            
            # Extract user profile
            user_profile_match = re.search(r'User profile: ([^#]+)', prompt)
            user_profile = user_profile_match.group(1).strip() if user_profile_match else "A Yelp user"
            
            # Extract G-Refer's explanation
            chosen = data['source_data']['chosen']
            explanation = chosen.split('###')[-1].strip() if '###' in chosen else chosen.strip()
            
            metadata[(uid, iid)] = {
                'business_title': business_title,
                'business_profile': business_profile,
                'user_profile': user_profile,
                'grefer_explanation': explanation
            }
    
    print(f"✓ Loaded metadata for {len(metadata)} user-item pairs")
    return metadata


def generate_explanation_with_metadata(args_tuple):
    """Generate explanation using business and user metadata."""
    sample_id, user_id, item_id, business_title, business_profile, user_profile = args_tuple
    
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
        
        # Rich prompt with metadata (similar to G-Refer)
        prompt = f"""Given the business information and user preferences, explain why the user would enjoy this business in 40-50 words.

Business: {business_title}
Business Profile: {business_profile}

User Preferences: {user_profile[:200]}...

Generate a specific, detailed explanation mentioning concrete features of the business:"""
        
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


def evaluate_with_metadata(num_samples=100, max_workers=10):
    """Evaluate with business and user metadata integration."""
    
    print("\n" + "="*80)
    print("Evaluation with Business/User Metadata")
    print("="*80)
    print("This should significantly improve BERTScore!")
    
    # Load metadata
    metadata = load_grefer_metadata('yelp')
    
    if not metadata:
        print("Cannot proceed without metadata")
        return {}
    
    # Get pairs
    available_pairs = list(metadata.keys())[:num_samples]
    print(f"\nEvaluating on {len(available_pairs)} user-item pairs")
    
    # Prepare tasks
    tasks = []
    for i, (uid, iid) in enumerate(available_pairs):
        meta = metadata[(uid, iid)]
        tasks.append((
            i, uid, iid,
            meta['business_title'],
            meta['business_profile'],
            meta['user_profile']
        ))
    
    # Generate explanations
    print(f"\nGenerating with {max_workers} workers (with business/user metadata)...")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_explanation_with_metadata, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            results.append(future.result())
    
    elapsed = time.time() - start_time
    print(f"✓ Generated {len(results)} explanations in {elapsed:.1f}s ({len(results)/elapsed:.1f} exp/sec)")
    
    # Extract predictions and references
    results.sort(key=lambda x: x[0])
    our_predictions = [r[3] for r in results]
    grefer_references = [metadata[(r[1], r[2])]['grefer_explanation'] for r in results]
    
    # Calculate metrics
    print("\n" + "="*80)
    print("Computing Metrics")
    print("="*80)
    
    # Simple metrics
    print("\n[1/2] Simple metrics...")
    unique_preds = len(set(our_predictions))
    usr = unique_preds / len(our_predictions)
    avg_pred_len = np.mean([len(p.split()) for p in our_predictions])
    avg_ref_len = np.mean([len(r.split()) for r in grefer_references])
    
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
    print("RESULTS WITH METADATA")
    print("="*80)
    
    print(f"\n📊 Metrics:")
    print(f"  USR: {usr:.4f}")
    print(f"  Unique: {unique_preds}/{len(our_predictions)}")
    print(f"  Avg Length: {avg_pred_len:.1f} words (G-Refer: {avg_ref_len:.1f})")
    print(f"  Word Overlap: {word_overlap:.4f}")
    
    if bert_metrics:
        print(f"\n📊 BERTScore:")
        print(f"  Precision: {bert_metrics['bert_precision']:.4f} ± {bert_metrics['bert_precision_std']:.4f}")
        print(f"  Recall: {bert_metrics['bert_recall']:.4f} ± {bert_metrics['bert_recall_std']:.4f}")
        print(f"  F1: {bert_metrics['bert_f1']:.4f} ± {bert_metrics['bert_f1_std']:.4f}")
        
        print(f"\n📊 Comparison:")
        f1_before = 0.3269
        f1_now = bert_metrics['bert_f1']
        improvement = f1_now - f1_before
        print(f"  Before (no metadata): {f1_before:.4f}")
        print(f"  After (with metadata): {f1_now:.4f}")
        print(f"  Improvement: +{improvement:.4f} ({improvement/f1_before*100:+.1f}%)")
        
        grefer_f1 = 0.88
        print(f"\n📊 vs G-Refer:")
        print(f"  G-Refer BERT F1: {grefer_f1:.4f}")
        print(f"  Our BERT F1: {f1_now:.4f}")
        print(f"  Gap: {grefer_f1 - f1_now:.4f}")
        
        if f1_now > 0.80:
            print(f"  ✓✓ EXCELLENT! Matching G-Refer")
        elif f1_now > 0.70:
            print(f"  ✓ GOOD! Approaching G-Refer")
        elif f1_now > 0.60:
            print(f"  ⚠ MODERATE, but improved significantly")
        elif f1_now > 0.45:
            print(f"  ⚠ IMPROVED, needs more work")
        else:
            print(f"  ❌ LOW, check prompt quality")
    
    # Save
    output_dir = Path('results/with_metadata')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = {
        'usr': usr,
        'unique_explanations': unique_preds,
        'word_overlap': word_overlap,
        'avg_prediction_length': avg_pred_len,
        'avg_reference_length': avg_ref_len,
        **bert_metrics
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in all_metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    # Save samples
    samples = []
    for i in range(min(10, len(our_predictions))):
        samples.append({
            'id': i,
            'user_id': int(results[i][1]),
            'item_id': int(results[i][2]),
            'our_explanation': our_predictions[i],
            'grefer_explanation': grefer_references[i]
        })
    
    with open(output_dir / 'comparisons.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}/")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_workers', type=int, default=10)
    
    args = parser.parse_args()
    
    try:
        metrics = evaluate_with_metadata(args.num_samples, args.max_workers)
        print("\n✓ Evaluation completed!")
        return 0
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
