"""Prepare native RULER evaluation and score with TTT-NTP's Full-13 rules."""
import json
import os
from pathlib import Path
import re
import sys

FULL13 = [
    'niah_single_1', 'niah_single_2', 'niah_single_3',
    'niah_multikey_1', 'niah_multikey_2', 'niah_multikey_3',
    'niah_multiquery', 'niah_multivalue', 'vt', 'fwe', 'cwe',
    'qa_squad', 'qa_hotpotqa',
]


def score_sample(task, prediction, refs):
    """Same case-insensitive substring scoring as inference_closed_form_patch.py."""
    hits = sum(ref.lower() in prediction.lower() for ref in refs)
    return float(hits > 0) if task.startswith('qa_') else hits / max(1, len(refs))


def prepare(out, lengths, samples):
    from mmengine.config import Config

    ckpt = str(Path(os.environ['CKPT']).resolve())
    model_abbr = os.environ['MODEL_ABBR']
    config_dir = out / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        if length not in {'4k', '8k', '16k', '32k', '64k'}:
            raise ValueError(f'Unsupported context length: {length}')
        # Resolve the existing OpenCompass dataset recipes, including their
        # prompt templates, answer fields, and default random_seed=42.
        config = Config.fromfile(str(Path(__file__).resolve().parents[1] / 'eval_config' / f'ruler_{length}.py'))
        datasets = config.datasets
        assert len(datasets) == 13
        assert {d['abbr'].removeprefix('ruler_').removesuffix('_' + length) for d in datasets} == set(FULL13)
        for dataset in datasets:
            dataset['tokenizer_model'] = ckpt
            dataset['num_samples'] = samples
        model = dict(
            type="opencompass.models.HuggingFaceBaseModel", abbr=model_abbr, path=ckpt,
            model_kwargs=dict(trust_remote_code=True),
            tokenizer_kwargs=dict(trust_remote_code=True),
            gen_config=dict(do_sample=False, temperature=0, top_p=0.95,
                            max_new_tokens=int(os.environ['MAX_NEW_TOKENS'])),
            max_seq_len=262144, max_out_len=int(os.environ['MAX_NEW_TOKENS']),
            # Native TTT inference updates only the first example in a batch.
            batch_size=1, run_cfg=dict(num_gpus=1),
            pred_postprocessor=dict(type="opencompass.utils.text_postprocessors.extract_non_reasoning_content"),
        )
        config.models = [model]
        config.model_dataset_combinations = [dict(models=[model], datasets=datasets)]
        config.work_dir = str(out / length)
        config.infer.partitioner.num_worker = 1
        config.infer.runner.max_num_workers = 1
        config.eval.runner.max_num_workers = 1
        config.dump(str(config_dir / f'ruler_{length}.py'))
    print(f'Prepared configs: {config_dir}')


def summarize(out, lengths, samples, selected_length=None):
    result = dict(method='inplace_ttt_native', checkpoint=os.environ['CKPT'],
                  num_samples=samples, max_new_tokens=int(os.environ['MAX_NEW_TOKENS']),
                  scoring='TTT-NTP substring matching; equal mean over 13 tasks', lengths={})
    eval_lengths = [selected_length] if selected_length else lengths
    for length in eval_lengths:
        candidates = sorted((out / length).glob('*/predictions/' + os.environ['MODEL_ABBR']))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one prediction run for {length}, found {len(candidates)}. Use a fresh OUT_DIR.')
        per_task = {}
        for task in FULL13:
            pattern = re.compile(rf'ruler_{re.escape(task)}_{length}(?:_(\d+))?\.json$')
            files = [p for p in candidates[0].glob('*.json') if pattern.fullmatch(p.name)]
            # Reject mixed merged/sharded output instead of double counting.
            if len(files) > 1 and any(p.name == f'ruler_{task}_{length}.json' for p in files):
                raise ValueError(f'Ambiguous merged and sharded predictions: {task}')
            rows = []
            for path in sorted(files):
                rows.extend(v for v in json.loads(path.read_text()).values()
                            if isinstance(v, dict) and 'origin_prompt' in v)
            if len(rows) != samples:
                raise ValueError(f'{length}/{task}: expected {samples} predictions, got {len(rows)}')
            for row in rows:
                if not isinstance(row.get('prediction'), str) or not isinstance(row.get('gold'), list) or not row['gold']:
                    raise ValueError(f'Invalid prediction/gold record in {length}/{task}')
            score = 100 * sum(score_sample(task, r['prediction'], r['gold']) for r in rows) / samples
            per_task[task] = dict(n=samples, native=score)
        score = sum(v['native'] for v in per_task.values()) / 13
        result['lengths'][length] = dict(n_tasks=13, native=score, per_task=per_task, missing=[])
        print(f'RULER {length} Full-13 native score: {score:.2f}')
    if selected_length:
        target = out / selected_length / 'summary.json'
        target.write_text(json.dumps(result['lengths'][selected_length], indent=2) + '\n')
        print(f'Summary: {target}')
        return

    all_scores = [v['native'] for v in result['lengths'].values()]
    result['all_lengths_avg'] = dict(native=sum(all_scores) / len(all_scores), lengths=lengths)
    paper_lengths = [length for length in ('4k', '8k', '16k', '32k') if length in result['lengths']]
    if paper_lengths:
        result['full13_avg'] = dict(
            native=sum(result['lengths'][length]['native'] for length in paper_lengths) / len(paper_lengths),
            lengths=paper_lengths,
        )
        print(f"RULER paper Avg (4k-32k): {result['full13_avg']['native']:.2f}")
    print(f"RULER all-length Avg ({', '.join(lengths)}): {result['all_lengths_avg']['native']:.2f}")
    target = out / 'ruler_inplace_summary.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Summary: {target}')


if __name__ == '__main__':
    out = Path(os.environ['OUT_DIR']).resolve()
    lengths = os.environ['LENGTHS'].split()
    samples = int(os.environ['NUM_SAMPLES'])
    if not lengths or len(set(lengths)) != len(lengths) or samples <= 0 or int(os.environ['MAX_NEW_TOKENS']) <= 0:
        raise ValueError('Lengths must be unique and nonempty; sample/token counts must be positive.')
    command = sys.argv[1]
    if command == 'summarize-length':
        if len(sys.argv) != 3 or sys.argv[2] not in lengths:
            raise ValueError('summarize-length requires one context length from LENGTHS.')
        summarize(out, lengths, samples, selected_length=sys.argv[2])
    else:
        {'prepare': prepare, 'summarize': summarize}[command](out, lengths, samples)
