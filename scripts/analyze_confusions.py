"""Analyze phoneme confusions from confusion matrix results."""

import json
from pathlib import Path

def analyze_confusions(confusion_json_path):
    """Analyze phoneme confusions from confusion matrix JSON."""
    
    with open(confusion_json_path, 'r') as f:
        data = json.load(f)
    
    cm = data['confusion_matrix']
    phonemes = data['phoneme_order']
    confusions = []
    
    # Find all confusions
    for i in range(len(phonemes)):
        row_sum = sum(cm[i])
        if row_sum == 0:
            continue
            
        for j in range(len(phonemes)):
            if i != j and cm[i][j] > 0:
                error_rate = cm[i][j] / row_sum
                confusions.append({
                    'source': phonemes[i],
                    'target': phonemes[j],
                    'count': cm[i][j],
                    'total': row_sum,
                    'error_rate': error_rate
                })
    
    # Sort by error rate
    confusions.sort(key=lambda x: x['error_rate'], reverse=True)
    
    print('Top Phoneme Confusions by Error Rate:')
    print('Source → Target (errors/total = rate)')
    print('-' * 50)
    
    # Show confusions with 100% error rate
    perfect_confusions = [c for c in confusions if c['error_rate'] == 1.0]
    if perfect_confusions:
        print('\n100% Confusion (always misclassified):')
        for conf in perfect_confusions:
            print(f"{conf['source']:6} → {conf['target']:6} ({conf['count']}/{conf['total']} = 100.0%)")
    
    # Show other high confusions
    high_confusions = [c for c in confusions if 0.5 <= c['error_rate'] < 1.0]
    if high_confusions:
        print('\nHigh Confusion (50-99%):')
        for conf in high_confusions[:10]:
            print(f"{conf['source']:6} → {conf['target']:6} ({conf['count']}/{conf['total']} = {conf['error_rate']*100:.1f}%)")
    
    # Analyze phonetic patterns
    print('\n\nPhonetic Pattern Analysis:')
    
    # Voicing confusions
    voicing_confusions = []
    for conf in confusions:
        src, tgt = conf['source'], conf['target']
        # Check if only voicing differs (p/b, t/d, k/g, f/v, s/z)
        voicing_pairs = [('p', 'b'), ('t', 'd'), ('k', 'g'), ('f', 'v'), ('s', 'z')]
        for v1, v2 in voicing_pairs:
            if (v1 in src and v2 in tgt and src.replace(v1, v2) == tgt) or \
               (v2 in src and v1 in tgt and src.replace(v2, v1) == tgt):
                voicing_confusions.append(conf)
                break
    
    if voicing_confusions:
        print('\nVoicing Confusions:')
        for conf in sorted(voicing_confusions, key=lambda x: x['error_rate'], reverse=True)[:5]:
            print(f"{conf['source']:6} ↔ {conf['target']:6} ({conf['error_rate']*100:.1f}%)")
    
    # Place of articulation confusions
    place_groups = {
        'labial': ['p', 'b', 'f', 'v'],
        'dental': ['t', 'd', 's', 'z'],
        'velar': ['k', 'g']
    }
    
    place_confusions = []
    for conf in confusions:
        src_consonants = [c for c in conf['source'] if c.isalpha() and c not in 'aeiou']
        tgt_consonants = [c for c in conf['target'] if c.isalpha() and c not in 'aeiou']
        
        if src_consonants and tgt_consonants:
            src_place = None
            tgt_place = None
            
            for place, consonants in place_groups.items():
                if any(c in consonants for c in src_consonants):
                    src_place = place
                if any(c in consonants for c in tgt_consonants):
                    tgt_place = place
            
            if src_place and tgt_place and src_place != tgt_place:
                place_confusions.append({**conf, 'src_place': src_place, 'tgt_place': tgt_place})
    
    if place_confusions:
        print('\nPlace of Articulation Confusions:')
        for conf in sorted(place_confusions, key=lambda x: x['error_rate'], reverse=True)[:5]:
            print(f"{conf['source']:6} → {conf['target']:6} ({conf['src_place']} → {conf['tgt_place']}, {conf['error_rate']*100:.1f}%)")
    
    # Summary statistics
    total_correct = sum(cm[i][i] for i in range(len(phonemes)))
    total_samples = sum(sum(row) for row in cm)
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    print(f'\n\nOverall Statistics:')
    print(f'Total samples: {total_samples}')
    print(f'Overall accuracy: {overall_accuracy:.3f}')
    print(f'Number of phoneme classes: {len(phonemes)}')
    print(f'Total confusions: {len(confusions)}')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path('multirun/2025-06-07/14-41-06/complete_analysis/confusion_matrix_results.json')
    
    analyze_confusions(path)