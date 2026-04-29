"""
Build graph memory layer on top of existing Mem0-lite memories.

Pipeline:
  1. Load existing Mem0-lite memory pkl produced by build_mem0_memories.py:
       dragon_emb/{prefix}_mem0_{sample_id}.pkl
     This pkl already contains LLM-extracted, deduped/consolidated memory
     facts and DRAGON embeddings. We do NOT re-extract facts here.
  2. For each fact, ask Qwen for a JSON list of (subject, relation, object)
     triples (best-effort parsing; bad outputs are skipped, not fatal).
  3. Build a networkx.MultiDiGraph:
       nodes = canonical entity strings (lowercased)
       edges = (subj_node, obj_node) with attrs:
           relation, mem_idx, text, date_time, dia_id
     Also keep a parallel list mem_to_entities[mi] -> [entity_keys] so the
     retriever can do 1-hop expansion at query time.
  4. Save the original pkl fields (embeddings, context, date_time, dia_id)
     plus 'graph' and 'mem_to_entities' to:
       dragon_emb/{prefix}_graph_mem0_{sample_id}.pkl

Run on Colab:

    !pip install -q networkx
    python scripts/build_graph_memories.py \
        --data-file data/locomo10_smoke.json \
        --sample-id conv-30 \
        --emb-dir dragon_emb \
        --model qwen2.5-7b-instruct --use-4bit
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import pickle
from argparse import Namespace

import networkx as nx
from tqdm import tqdm

from task_eval.hf_llm_utils import init_hf_model


EXTRACTION_SYSTEM = (
    "You are an information extraction assistant. Given one short factual "
    "statement about a person, extract zero or more (subject, relation, "
    "object) triples capturing the entities and their relationship.\n"
    "Rules:\n"
    "- Subject is a person, place, organization, or named entity.\n"
    "- Relation is a short verb phrase, 1-4 words, lowercase, "
    "  underscores allowed (e.g. 'lost_job_as', 'plans_to_start').\n"
    "- Object is another entity, a date, a status, or a concrete value.\n"
    "- If the fact contains a date or time, encode it as a SEPARATE triple "
    "  with relation 'occurred_on' or 'since'.\n"
    "- Use the speaker's name as subject when the fact is about that speaker.\n"
    "- Return ONLY a JSON list of objects with keys 'subject', 'relation', "
    "  'object'. No prose, no explanation.\n"
    "- Return at most 4 triples. Return [] if no clear triple is present."
)

EXTRACTION_USER = "Fact: {text}\nReturn JSON only:"


def parse_json_list(text):
    """Best-effort JSON list extraction from LLM output (mirrors Mem0-lite)."""
    if not text:
        return []
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
    triples = []
    for item in out:
        if not isinstance(item, dict):
            continue
        s = str(item.get('subject', '')).strip()
        r = str(item.get('relation', '')).strip()
        o = str(item.get('object', '')).strip()
        if s and r and o and len(s) < 80 and len(o) < 120:
            triples.append({'subject': s, 'relation': r, 'object': o})
    return triples


def extract_triples_from_fact(fact_text, pipeline, tokenizer):
    messages = [
        {'role': 'system', 'content': EXTRACTION_SYSTEM},
        {'role': 'user',   'content': EXTRACTION_USER.format(text=fact_text)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    try:
        gen = pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            return_full_text=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        return parse_json_list(gen[0]['generated_text'])
    except Exception as e:
        print('  triple extract error: %s' % e)
        return []


def _ent_key(s):
    return ' '.join(str(s).strip().lower().split())


def build_graph_for_sample(base_pkl, out_pkl, pipeline, tokenizer, args):
    base = pickle.load(open(base_pkl, 'rb'))
    contexts = base.get('context', [])
    n = len(contexts)
    print('  loaded %d Mem0-lite facts from %s' % (n, base_pkl))

    G = nx.MultiDiGraph()
    mem_to_entities = []
    triple_count = 0
    parse_fail = 0

    for mi in tqdm(range(n), desc='extract triples'):
        text = contexts[mi]
        triples = extract_triples_from_fact(text, pipeline, tokenizer)
        if not triples:
            parse_fail += 1
            mem_to_entities.append([])
            continue

        ents = []
        for t in triples:
            sk = _ent_key(t['subject'])
            ok = _ent_key(t['object'])
            if not sk or not ok or sk == ok:
                continue
            if sk not in G:
                G.add_node(sk, display=t['subject'].strip())
            if ok not in G:
                G.add_node(ok, display=t['object'].strip())
            G.add_edge(sk, ok,
                       relation=t['relation'],
                       mem_idx=mi,
                       text=text,
                       date_time=base.get('date_time', [''] * n)[mi],
                       dia_id=base.get('dia_id', [''] * n)[mi])
            triple_count += 1
            if sk not in ents:
                ents.append(sk)
            if ok not in ents:
                ents.append(ok)
        mem_to_entities.append(ents)

    print('  graph: %d nodes, %d edges (triples)' %
          (G.number_of_nodes(), triple_count))
    print('  facts with no parsed triple: %d / %d' % (parse_fail, n))

    out = dict(base)
    out['graph'] = G
    out['mem_to_entities'] = mem_to_entities
    os.makedirs(os.path.dirname(out_pkl) or '.', exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(out, f)
    print('  wrote %s' % out_pkl)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-file', required=True)
    ap.add_argument('--emb-dir', default='dragon_emb')
    ap.add_argument('--sample-id', default=None,
                    help='Optional: build only for this sample_id')
    ap.add_argument('--model', default='qwen2.5-7b-instruct')
    ap.add_argument('--use-4bit', action='store_true')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--base-suffix', default='mem0',
                    help='Read {prefix}_{base-suffix}_{sample}.pkl as the input '
                         'memory store (default: mem0).')
    return ap.parse_args()


def main():
    args = parse_args()
    samples = json.load(open(args.data_file, 'r', encoding='utf-8'))
    if args.sample_id is not None:
        samples = [s for s in samples if s.get('sample_id') == args.sample_id]
        if not samples:
            raise SystemExit('No sample with sample_id=%s' % args.sample_id)

    dataset_prefix = os.path.splitext(os.path.basename(args.data_file))[0]
    init_ns = Namespace(model=args.model, use_4bit=args.use_4bit)
    pipeline, _ = init_hf_model(init_ns)
    tokenizer = pipeline.tokenizer

    for data in samples:
        sid = data['sample_id']
        base_pkl = os.path.join(args.emb_dir,
                                '%s_%s_%s.pkl' % (dataset_prefix, args.base_suffix, sid))
        out_pkl = os.path.join(args.emb_dir,
                               '%s_graph_%s_%s.pkl' % (dataset_prefix, args.base_suffix, sid))
        if not os.path.exists(base_pkl):
            print('skip sample=%s: base pkl missing %s '
                  '(build it first via scripts/build_mem0_memories.py)' % (sid, base_pkl))
            continue
        if os.path.exists(out_pkl) and not args.overwrite:
            print('skip existing %s (use --overwrite to rebuild)' % out_pkl)
            continue
        print('Building graph memory for sample=%s' % sid)
        build_graph_for_sample(base_pkl, out_pkl, pipeline, tokenizer, args)

    print('done.')


if __name__ == '__main__':
    main()
