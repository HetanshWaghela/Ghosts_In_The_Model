"""Verifying that Gpt-2 actually knows the answers to the prompts in the dataset"""

def verify_dataset(model,tokenizer,dataset,device,threshold=0.05):
    """
    Verifying that the model knows all facts in dataset.
    
    NOTE: threshold=0.05 (5%) matches data_utils.py generation threshold.
    GPT-2 Small often has low confidence even for facts it "knows".
    """

    import torch
    results= []

    for prompt in dataset:
        inputs= tokenizer(prompt['text'],return_tensors='pt').to(device)
        with torch.no_grad():
            outputs= model(**inputs)
            logits = outputs.logits[0,-1,:]
            probs = torch.softmax(logits,dim=-1)

            target_tokens= tokenizer.encode(" " + prompt['target'])
            if target_tokens:
                target_prob= probs[target_tokens[0]].item()
            else:
                target_prob=0.0

            top_prob, top_idx = torch.max(probs,dim=-1)
            top_token= tokenizer.decode([top_idx.item()])

            
            results.append({
            'prompt': prompt['text'],
            'target': prompt['target'],
            'target_prob': target_prob,
            'top_token': top_token.strip(),
            'top_prob': top_prob.item(),
            'is_correct': top_token.strip().lower() == prompt['target'].lower(),
            'above_threshold': target_prob >= threshold
        })
    
    correct= sum(r['is_correct'] for r in results)
    above_thresh= sum(r['above_threshold'] for r in results)

    print(f"dataset verification:")
    print(f"  - Correct top-1: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    print(f"  - Above threshold: {above_thresh}/{len(results)} ({above_thresh/len(results)*100:.1f}%)")
  
    return results

if __name__ == "__main__":
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    import json
    from pathlib import Path
    from datetime import datetime
    import sys
    device= "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer= GPT2Tokenizer.from_pretrained("gpt2")
    model= GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    data_dir = Path("../data/processed")
    results_dir = Path("../results/verification")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    datasets_to_verify = [
        ("forget.json", "FORGET", "Facts to unlearn - GPT-2 MUST know these"),
        ("retain.json", "RETAIN", "Facts to preserve - needed to detect model damage"),
        ("probe_train.json", "PROBE", "Facts for training ghost detectors"),
    ]
    
    all_results = {}
    summary_data = {}
    
    output_lines = []
    
    for filename, name, description in datasets_to_verify:
        filepath = data_dir / filename
        
        if not filepath.exists():
            msg = f"\n WARNING: {name} dataset not found at {filepath}"
            print(msg)
            output_lines.append(msg)
            continue
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        header = f"\n{'='*60}\nVerifying {name} dataset: {description}\n{'='*60}"
        print(header)
        output_lines.append(header)
        
        results = verify_dataset(model, tokenizer, data, device)
        all_results[name] = results
        
        correct = sum(r['is_correct'] for r in results)
        above_thresh = sum(r['above_threshold'] for r in results)
        total = len(results)
        
        summary_data[name] = {
            'correct_top1': correct,
            'total': total,
            'top1_pct': correct/total*100 if total > 0 else 0,
            'above_threshold': above_thresh,
            'above_threshold_pct': above_thresh/total*100 if total > 0 else 0
        }
        
        verify_output = f"dataset verification:\n  - Correct top-1: {correct}/{total} ({correct/total*100:.1f}%)\n  - Above threshold: {above_thresh}/{total} ({above_thresh/total*100:.1f}%)"
        output_lines.append(verify_output)
        
        failed = [r for r in results if not r['above_threshold']]
        if failed:
            msg = f"\n WARNING: {len(failed)} prompts below threshold:"
            print(msg)
            output_lines.append(msg)
            for r in failed[:5]:
                line = f"  '{r['prompt']}' → expected '{r['target']}', got '{r['top_token']}' (p={r['target_prob']:.3f})"
                print(line)
                output_lines.append(line)
            if len(failed) > 5:
                msg = f"  ... and {len(failed)-5} more"
                print(msg)
                output_lines.append(msg)
    
    summary_header = f"\n{'='*60}\nVERIFICATION SUMMARY\n{'='*60}"
    print(summary_header)
    output_lines.append(summary_header)
    
    for name, results in all_results.items():
        valid = sum(r['above_threshold'] for r in results)
        total = len(results)
        pct = valid/total*100 if total > 0 else 0
        status = "PASSED" if pct >= 80 else "WARNING" if pct >= 50 else "FAILED"
        line = f"  {status} {name}: {valid}/{total} ({pct:.1f}%) above threshold"
        print(line)
        output_lines.append(line)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'model': 'gpt2',
        'threshold': 0.05,
        'device': device,
        'summary': summary_data
    }
    
    json_path = results_dir / f"verify_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    txt_path = results_dir / f"verify_{timestamp}.txt"
    with open(txt_path, 'w') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\nResults saved to:")
    print(f"  JSON: {json_path}")
    print(f"  Text: {txt_path}")