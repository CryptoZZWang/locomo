"""
Build Mem0-lite memories for LoCoMo samples.

Pipeline (Mem0-inspired, fully in-repo):
  1. For each conversation sample, walk sessions in order.
  2. Slide over every adjacent turn pair (user, assistant), giving the LLM
     a small rolling context of the previous turns in the same session
     (default: last 6 turns).
  3. Ask Qwen to extract a short JSON list of speaker-attributed facts.
  4. DRAGON-embed all extracted facts, greedily dedupe by cosine similarity.
  5. Dump a pkl in the same schema prepare_for_rag() expects for dialog /
     observation modes, at: {emb_dir}/{dataset_prefix}_mem0_{sample_id}.pkl

Run on Colab (or any box with a GPU + HF_TOKEN):

    python scripts/build_mem0_memories.py \
        --data-file data/locomo10_smoke.json \
        --emb-dir dragon_emb \
        --model qwen2.5-7b-instruct \
        --use-4bit

Then evaluate with the existing HF path:

    python task_eval/evaluate_qa.py \
        --model qwen2.5-7b-instruct --use-4bit \
        --data-file data/locomo10_smoke.json \
        --out-file outputs/qwen_conv30_mem0_top10_qa.json \
        --use-rag --rag-mode mem0 --retriever dragon --top-k 10 --batch-size 1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import pickle
import re
from argparse import Namespace

import numpy as np
from tqdm import tqdm

from task_eval.hf_llm_utils import init_hf_model
from task_eval.rag_utils import get_embeddings


EXTRACTION_SYSTEM = (
    "You are a memory extraction assistant. Given a recent snippet of a "
    "conversation between {a} and {b}, extract at most 4 short, self-contained "
    "factual statements worth remembering for future conversations. Each fact "
    "must be a single short sentence, must start with the speaker's name, and "
    "must be factual (preferences, events, relationships, plans, feelings, "
    "possessions) rather than greetings, chit-chat, or generic statements. "
    "Return ONLY a JSON list of objects with keys \"speaker\" and \"fact\". "
    "Return [] if nothing is worth remembering."
)

EXTRACTION_USER = (
    "Recent context:\n{context}\n\n"
    "New exchange:\n{pair}\n\n"
    "Return JSON only:"
)


CONSOLIDATE_SYSTEM = (
    "You are a memory manager for an AI assistant. For each new candidate "
    "fact about a person, you decide how it should be integrated relative "
    "to the most similar existing memories already in the store. Be precise "
    "and conservative."
)

CONSOLIDATE_USER = (
    "Existing memories (most similar to the candidate, indexed):\n{neighbors}\n\n"
    "Candidate fact:\n{candidate}\n\n"
    "Choose ONE operation:\n"
    "- ADD:    candidate is new information, not redundant with any existing memory\n"
    "- NOOP:   candidate is already represented by one of the existing memories\n"
    "- UPDATE: candidate refines or augments an existing memory; replace it with one richer fact\n"
    "- DELETE: candidate contradicts an existing memory; the old memory is now incorrect\n\n"
    "Return ONLY a JSON object with these keys:\n"
    "- \"action\":       one of \"ADD\", \"NOOP\", \"UPDATE\", \"DELETE\"\n"
    "- \"target_index\": integer index of the affected existing memory (only for UPDATE / DELETE)\n"
    "- \"merged_fact\":  one short sentence merging candidate with the target memory (only for UPDATE)"
)


def parse_json_list(text):
    """Best-effort JSON list extraction from LLM output."""
    i = text.find('[')
    j = text.rfind(']')
    if i == -1 or j == -1 or j < i:
        return []
    try:
        out = json.loads(text[i:j + 1])
    except Exception:
        return []
    if not isinstance(out, list):
        return []
    clean = []
    for item in out:
        if isinstance(item, dict) and 'fact' in item:
            fact = str(item.get('fact', '')).strip()
            speaker = str(item.get('speaker', '')).strip()
            if fact:
                clean.append({'speaker': speaker, 'fact': fact})
    return clean


def parse_json_obj(text):
    """Best-effort JSON object extraction from LLM output."""
    i = text.find('{')
    j = text.rfind('}')
    if i == -1 or j == -1 or j < i:
        return {}
    try:
        out = json.loads(text[i:j + 1])
        return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def turn_str(turn):
    text = turn.get('text', '').strip()
    spk = turn.get('speaker', '').strip()
    line = '%s: "%s"' % (spk, text)
    if turn.get('blip_caption'):
        line += ' [shared image: %s]' % turn['blip_caption']
    return line


def iter_session_nums(conv):
    nums = []
    for k in conv.keys():
        if k.startswith('session_') and 'date_time' not in k and k.split('_')[1].isdigit():
            nums.append(int(k.split('_')[1]))
    return sorted(set(nums))


def extract_facts_for_sample(data, pipeline, tokenizer, args):
    """Extract memories across all sessions; returns list of dicts:
        {'text': 'Speaker: fact', 'dia_id': 'D10:3,D10:4', 'date_time': '...'}
    """
    conv = data['conversation']
    session_nums = iter_session_nums(conv)

    # names of the two speakers (from session 1)
    first_session = conv.get('session_%d' % session_nums[0], [])
    names = []
    for t in first_session:
        s = t.get('speaker', '').strip()
        if s and s not in names:
            names.append(s)
        if len(names) == 2:
            break
    while len(names) < 2:
        names.append('Speaker_%d' % (len(names) + 1))
    spk_a, spk_b = names[0], names[1]

    sys_msg = EXTRACTION_SYSTEM.format(a=spk_a, b=spk_b)
    out_records = []

    for sn in session_nums:
        turns = conv.get('session_%d' % sn, [])
        if len(turns) < 2:
            continue
        date_time = conv.get('session_%d_date_time' % sn, '')

        for p in tqdm(range(len(turns) - 1),
                      desc='sample=%s session=%d' % (data.get('sample_id', '?'), sn),
                      leave=False):
            t1, t2 = turns[p], turns[p + 1]
            ctx_turns = turns[max(0, p - args.context_window):p]
            context_text = '\n'.join(turn_str(t) for t in ctx_turns) or '(start of session)'
            pair_text = '%s\n%s' % (turn_str(t1), turn_str(t2))

            messages = [
                {'role': 'system', 'content': sys_msg},
                {'role': 'user',   'content': EXTRACTION_USER.format(
                    context=context_text, pair=pair_text)},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                gen = pipeline(
                    prompt,
                    max_new_tokens=256,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                text = gen[0]['generated_text']
            except Exception as e:
                print('  extract error at pair %d: %s' % (p, e))
                continue

            facts = parse_json_list(text)
            dia_id = '%s,%s' % (t1.get('dia_id', ''), t2.get('dia_id', ''))
            for f in facts:
                spk = f['speaker'] or (t1.get('speaker', '') if p % 2 == 0 else t2.get('speaker', ''))
                fact_text = f['fact']
                if not fact_text.lower().startswith(spk.lower()):
                    fact_text = '%s: %s' % (spk, fact_text)
                out_records.append({
                    'text': fact_text,
                    'dia_id': dia_id,
                    'date_time': date_time,
                })

    return out_records


def _text_key(s):
    return ' '.join(str(s).lower().split())


def dedupe_records(records, embeddings, threshold, verbose=True):
    """Two-stage dedupe:
       (1) exact text dedupe (case-folded + whitespace-normalized fact text)
       (2) centered-cosine greedy dedupe.

    Centering (subtract the mean embedding) removes the first-PC anisotropy in
    raw DRAGON CLS vectors, which otherwise makes unrelated facts score cosine
    ~0.93-0.97 and collapses everything under a 0.92 threshold.
    """
    if len(records) == 0:
        d = embeddings.shape[1] if embeddings.ndim == 2 else 0
        return [], np.zeros((0, d))

    # stage 1: exact text dedupe
    seen = set()
    idx_after_text = []
    for i, r in enumerate(records):
        k = _text_key(r['text'])
        if k and k not in seen:
            seen.add(k)
            idx_after_text.append(i)
    records1 = [records[i] for i in idx_after_text]
    emb1 = embeddings[idx_after_text]
    if verbose:
        print('  stage-1 exact-text dedupe: %d -> %d' % (len(records), len(records1)))

    if len(records1) <= 1:
        return records1, emb1

    # stage 2: centered cosine
    mean = emb1.mean(axis=0, keepdims=True)
    centered = emb1 - mean
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
    normed = centered / norms

    if verbose:
        n = min(300, len(normed))
        sim_mat = normed[:n] @ normed[:n].T
        iu = np.triu_indices(n, k=1)
        sims = sim_mat[iu]
        qs = np.quantile(sims, [0.5, 0.9, 0.99, 0.995, 0.999]).round(3).tolist()
        print('  centered-cos quantiles p50/p90/p99/p99.5/p99.9 (n=%d sample): %s' % (n, qs))

    keep_idx = []
    for i in range(len(records1)):
        if not keep_idx:
            keep_idx.append(i)
            continue
        sims = normed[keep_idx] @ normed[i]
        if float(sims.max()) < threshold:
            keep_idx.append(i)
    kept = [records1[i] for i in keep_idx]
    kept_emb = emb1[keep_idx]
    if verbose:
        print('  stage-2 centered-cos dedupe (threshold=%.3f): %d -> %d' % (threshold, len(records1), len(kept)))
    return kept, kept_emb


def _qwen_decide(candidate_text, neighbor_texts, pipeline, tokenizer):
    neighbors_block = '\n'.join('[%d] %s' % (i, t) for i, t in enumerate(neighbor_texts))
    messages = [
        {'role': 'system', 'content': CONSOLIDATE_SYSTEM},
        {'role': 'user',   'content': CONSOLIDATE_USER.format(
            neighbors=neighbors_block, candidate=candidate_text)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    try:
        gen = pipeline(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        return parse_json_obj(gen[0]['generated_text'])
    except Exception:
        return {}


def llm_consolidate_records(records, embeddings, pipeline, tokenizer, args, verbose=True):
    """Mem0 Algorithm 1 (Appendix B), streaming variant.

    For each candidate (in original chronological order):
      - find top-s most similar already-kept memories using centered cosine
      - if max similarity < auto_add_threshold -> auto-ADD (no LLM call)
      - else ask Qwen for one of {ADD, NOOP, UPDATE, DELETE} + target_index + merged_fact
    """
    if len(records) <= 1:
        return list(records), np.array(embeddings)

    mean = embeddings.mean(axis=0)

    def center_norm(e):
        c = e - mean
        n = float(np.linalg.norm(c)) + 1e-8
        return c / n

    kept_records = [records[0]]
    kept_emb = [embeddings[0]]
    kept_normed = [center_norm(embeddings[0])]

    counts = {'auto_add': 1, 'add': 0, 'noop': 0, 'update': 0, 'delete': 0, 'llm_calls': 0}

    for i in tqdm(range(1, len(records)), desc='consolidate', leave=False):
        cand = records[i]
        cand_emb = embeddings[i]
        cand_normed = center_norm(cand_emb)

        sims = np.array(kept_normed) @ cand_normed
        max_sim = float(sims.max())

        if max_sim < args.consolidate_add_threshold:
            kept_records.append(cand)
            kept_emb.append(cand_emb)
            kept_normed.append(cand_normed)
            counts['auto_add'] += 1
            continue

        s = min(args.consolidate_top_s, len(kept_records))
        if s < len(sims):
            top_idx = np.argpartition(-sims, s - 1)[:s]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        else:
            top_idx = np.argsort(-sims)
        neighbor_texts = [kept_records[int(j)]['text'] for j in top_idx]

        op = _qwen_decide(cand['text'], neighbor_texts, pipeline, tokenizer)
        counts['llm_calls'] += 1
        action = str(op.get('action', 'ADD')).strip().upper()
        target = op.get('target_index')
        try:
            target = int(target) if target is not None else None
        except (TypeError, ValueError):
            target = None
        target_ok = target is not None and 0 <= target < len(neighbor_texts)

        if action == 'NOOP':
            counts['noop'] += 1
            continue

        if action == 'UPDATE' and target_ok:
            g = int(top_idx[target])
            merged = str(op.get('merged_fact') or cand['text']).strip()
            new_emb = (kept_emb[g] + cand_emb) / 2.0
            kept_records[g] = {
                'text':      merged,
                'dia_id':    '%s;%s' % (kept_records[g]['dia_id'], cand['dia_id']),
                'date_time': cand['date_time'],
            }
            kept_emb[g] = new_emb
            kept_normed[g] = center_norm(new_emb)
            counts['update'] += 1
            continue

        if action == 'DELETE' and target_ok:
            g = int(top_idx[target])
            del kept_records[g]
            del kept_emb[g]
            del kept_normed[g]
            kept_records.append(cand)
            kept_emb.append(cand_emb)
            kept_normed.append(cand_normed)
            counts['delete'] += 1
            continue

        kept_records.append(cand)
        kept_emb.append(cand_emb)
        kept_normed.append(cand_normed)
        counts['add'] += 1

    if verbose:
        print('  llm-consolidate: auto_add=%(auto_add)d add=%(add)d noop=%(noop)d '
              'update=%(update)d delete=%(delete)d llm_calls=%(llm_calls)d' % counts)
        print('  kept %d facts' % len(kept_records))

    return kept_records, np.stack(kept_emb)


def build_for_sample(data, pipeline, tokenizer, args, out_pkl, raw_pkl):
    if args.from_raw and os.path.exists(raw_pkl):
        print('Loading raw extractions from %s' % raw_pkl)
        raw = pickle.load(open(raw_pkl, 'rb'))
        records = raw['records']
        embeddings = raw['embeddings']
        print('  loaded %d raw facts' % len(records))
    else:
        print('Extracting memories for sample=%s' % data.get('sample_id'))
        records = extract_facts_for_sample(data, pipeline, tokenizer, args)
        print('  extracted %d raw facts' % len(records))
        if len(records) == 0:
            print('  WARNING: no facts extracted; writing empty pkl')
            database = {'embeddings': np.zeros((0, 768)), 'date_time': [], 'dia_id': [], 'context': []}
            with open(out_pkl, 'wb') as f:
                pickle.dump(database, f)
            return

        texts = [r['text'] for r in records]
        embeddings = get_embeddings(args.retriever, texts, 'context')
        assert embeddings.shape[0] == len(records)

        os.makedirs(os.path.dirname(raw_pkl) or '.', exist_ok=True)
        with open(raw_pkl, 'wb') as f:
            pickle.dump({'records': records, 'embeddings': embeddings}, f)
        print('  wrote raw cache to %s' % raw_pkl)

    if args.llm_consolidate:
        seen = set()
        idx_after_text = []
        for i, r in enumerate(records):
            k = _text_key(r['text'])
            if k and k not in seen:
                seen.add(k)
                idx_after_text.append(i)
        records = [records[i] for i in idx_after_text]
        embeddings = embeddings[idx_after_text]
        print('  stage-1 exact-text dedupe: %d records' % len(records))
        kept, kept_emb = llm_consolidate_records(records, embeddings, pipeline, tokenizer, args)
    else:
        kept, kept_emb = dedupe_records(records, embeddings, args.dedupe_threshold)

    database = {
        'embeddings': kept_emb,
        'date_time': [r['date_time'] for r in kept],
        'dia_id':    [r['dia_id']    for r in kept],
        'context':   [r['text']      for r in kept],
    }
    os.makedirs(os.path.dirname(out_pkl) or '.', exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(database, f)
    print('  wrote %s' % out_pkl)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-file', required=True)
    ap.add_argument('--emb-dir', default='dragon_emb')
    ap.add_argument('--sample-id', default=None,
                    help='Optional: build only for this sample_id')
    ap.add_argument('--model', default='qwen2.5-7b-instruct')
    ap.add_argument('--use-4bit', action='store_true')
    ap.add_argument('--retriever', default='dragon')
    ap.add_argument('--context-window', type=int, default=6,
                    help='Number of prior turns from same session used as context')
    ap.add_argument('--dedupe-threshold', type=float, default=0.92,
                    help='Drop a fact if cosine similarity to any kept fact exceeds this')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--from-raw', action='store_true',
                    help='Skip LLM extraction; reuse {prefix}_mem0_raw_{sample}.pkl and only redo dedupe.')
    ap.add_argument('--llm-consolidate', action='store_true',
                    help='Use Mem0-style LLM ADD/UPDATE/DELETE/NOOP consolidation instead of cosine dedupe.')
    ap.add_argument('--consolidate-top-s', type=int, default=5,
                    help='Number of nearest existing memories to show the LLM per candidate.')
    ap.add_argument('--consolidate-add-threshold', type=float, default=0.3,
                    help='If max centered cosine to existing memories < this, auto-ADD without calling the LLM.')
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.data_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    if args.sample_id is not None:
        samples = [s for s in samples if s.get('sample_id') == args.sample_id]
        if not samples:
            raise SystemExit('No sample with sample_id=%s in %s' % (args.sample_id, args.data_file))

    os.makedirs(args.emb_dir, exist_ok=True)
    dataset_prefix = os.path.splitext(os.path.basename(args.data_file))[0]

    pipeline, tokenizer = None, None
    need_model = (not args.from_raw) or args.llm_consolidate
    if need_model:
        init_ns = Namespace(model=args.model, use_4bit=args.use_4bit)
        pipeline, _model_name = init_hf_model(init_ns)
        tokenizer = pipeline.tokenizer

    for data in samples:
        out_pkl = os.path.join(args.emb_dir, '%s_mem0_%s.pkl' % (dataset_prefix, data['sample_id']))
        raw_pkl = os.path.join(args.emb_dir, '%s_mem0_raw_%s.pkl' % (dataset_prefix, data['sample_id']))
        if os.path.exists(out_pkl) and not args.overwrite and not args.from_raw:
            print('skip existing %s (use --overwrite to rebuild, or --from-raw to re-dedupe)' % out_pkl)
            continue
        build_for_sample(data, pipeline, tokenizer, args, out_pkl, raw_pkl)

    print('done.')


if __name__ == '__main__':
    main()
