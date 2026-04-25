"""
Minimal official-mem0ai evaluation on LoCoMo conv-30.

Backend (default): Google Gemini 2.0 Flash for both LLM and embedder, qdrant
in-memory store. Chosen because previous mem0ai + local-LLM attempts on Colab
crashed on binary deps; Gemini Flash sidesteps that and is free at this scale.

Pipeline:
  1. Load --data-file (default: data/locomo10_smoke.json), pick --sample-id
     (default: conv-30).
  2. Initialize mem0.Memory with Gemini LLM + Gemini embedder.
  3. Per session, ingest the dialog into Mem0 with user_id = sample_id.
     This triggers Mem0's official extract + ADD/UPDATE/DELETE/NOOP under the hood.
  4. Per QA: m.search(question, user_id, limit=top_k), then call Gemini with the
     paper's "Prompt Template for Results Generation (Mem0)" (Appendix A).
  5. Save predictions to --out-file with shape [{sample_id, qa:[...]}], using
     prediction key official_mem0_gemini_prediction. Fully scoreable by
     task_eval/evaluation.py.
  6. Score with eval_question_answering + analyze_aggr_acc and write stats next
     to the predictions file.

Run on Colab (after `pip install mem0ai google-generativeai`):

    export GEMINI_API_KEY=...
    python scripts/run_official_mem0_eval.py \
        --data-file data/locomo10_smoke.json \
        --sample-id conv-30 \
        --out-file outputs/conv30_official_mem0_qa.json \
        --top-k 10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import time

from tqdm import tqdm


PREDICTION_KEY = "official_mem0_gemini_prediction"
MODEL_KEY      = "official_mem0_gemini"


# Paper Appendix A "Prompt Template for Results Generation (Mem0)" -- trimmed.
ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers.
2. Pay special attention to the timestamps to determine the answer.
3. If the question asks about a specific event or fact, look for direct evidence in the memories.
4. If the memories contain contradictory information, prioritize the most recent memory.
5. Always convert relative time references (e.g. "last year", "two months ago") to specific dates, months, or years based on the memory timestamp.
6. The answer should be less than 5-6 words.

Memories:
{memories}

Question: {question}
Answer:"""


def get_session_nums(conv):
    nums = []
    for k in conv.keys():
        if k.startswith('session_') and 'date_time' not in k and k.split('_')[1].isdigit():
            nums.append(int(k.split('_')[1]))
    return sorted(set(nums))


def build_memory(api_key, llm_model, embed_model, collection):
    """Construct an official mem0.Memory backed by Gemini + qdrant in-memory."""
    from mem0 import Memory
    config = {
        "llm": {
            "provider": "gemini",
            "config": {
                "model": llm_model,
                "api_key": api_key,
                "temperature": 0.0,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "gemini",
            "config": {
                "model": embed_model,
                "api_key": api_key,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": 768,
                "on_disk": False,
            },
        },
    }
    return Memory.from_config(config)


def ingest_sample(m, sample, user_id, sleep_per_session=2.0):
    """Feed each session into Mem0. Mem0 internally extracts + consolidates."""
    conv = sample['conversation']
    session_nums = get_session_nums(conv)
    for sn in tqdm(session_nums, desc='ingest sessions'):
        turns = conv.get('session_%d' % sn, [])
        if not turns:
            continue
        date_time = conv.get('session_%d_date_time' % sn, '')

        first_speaker = turns[0]['speaker']
        messages = []
        for t in turns:
            role = 'user' if t['speaker'] == first_speaker else 'assistant'
            content = '[%s] %s: %s' % (date_time, t['speaker'], t.get('text', ''))
            if t.get('blip_caption'):
                content += ' [shared image: %s]' % t['blip_caption']
            messages.append({'role': role, 'content': content})

        try:
            m.add(messages, user_id=user_id)
        except Exception as e:
            print('  ingest error session %d: %s' % (sn, e))

        if sleep_per_session:
            time.sleep(sleep_per_session)


def search_memories(m, query, user_id, k):
    """Mem0's search() return shape varies between releases."""
    try:
        res = m.search(query=query, user_id=user_id, limit=k)
    except TypeError:
        res = m.search(query, user_id=user_id, limit=k)
    if isinstance(res, dict):
        hits = res.get('results', []) or []
    elif isinstance(res, list):
        hits = res
    else:
        hits = []
    out = []
    for h in hits:
        if isinstance(h, dict):
            out.append(h.get('memory') or h.get('text') or str(h))
        else:
            out.append(str(h))
    return out


def gemini_generate(api_key, model_name, prompt, max_output_tokens=80, retries=2):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.0,
                    'max_output_tokens': max_output_tokens,
                },
            )
            return (resp.text or '').strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise last_err


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-file', default='data/locomo10_smoke.json')
    ap.add_argument('--sample-id', default='conv-30')
    ap.add_argument('--out-file',  default='outputs/conv30_official_mem0_qa.json')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--llm-model', default='gemini-2.0-flash')
    ap.add_argument('--embed-model', default='models/text-embedding-004')
    ap.add_argument('--collection', default='locomo_official_mem0')
    ap.add_argument('--skip-ingest', action='store_true',
                    help='Reuse an existing Mem0 collection if it persists; otherwise ingest.')
    ap.add_argument('--ingest-sleep', type=float, default=2.0,
                    help='Seconds to sleep between sessions to stay under Gemini RPM.')
    return ap.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise SystemExit('Set GEMINI_API_KEY in the environment.')

    samples = json.load(open(args.data_file, 'r', encoding='utf-8'))
    sample = next((s for s in samples if s.get('sample_id') == args.sample_id), None)
    if sample is None:
        raise SystemExit('sample_id=%s not found in %s' % (args.sample_id, args.data_file))

    print('initializing mem0...')
    m = build_memory(api_key, args.llm_model, args.embed_model, args.collection)

    if not args.skip_ingest:
        print('ingesting sample=%s into Mem0...' % args.sample_id)
        ingest_sample(m, sample, user_id=args.sample_id, sleep_per_session=args.ingest_sleep)

    print('generating answers for %d QAs...' % len(sample['qa']))
    qa_out = []
    os.makedirs(os.path.dirname(args.out_file) or '.', exist_ok=True)

    for i, qa in enumerate(tqdm(sample['qa'], desc='qa')):
        question = qa['question']
        try:
            mems = search_memories(m, question, args.sample_id, args.top_k)
        except Exception as e:
            print('  search error qa[%d]: %s' % (i, e))
            mems = []
        mem_text = '\n'.join('- %s' % t for t in mems) if mems else '(no memories retrieved)'

        prompt = ANSWER_PROMPT.format(memories=mem_text, question=question)
        try:
            answer = gemini_generate(api_key, args.llm_model, prompt)
        except Exception as e:
            print('  generate error qa[%d]: %s' % (i, e))
            answer = ''

        rec = dict(qa)
        rec[PREDICTION_KEY] = answer
        rec[PREDICTION_KEY + '_context'] = mems
        qa_out.append(rec)

        # incremental save every 10 QAs (cheap insurance)
        if (i + 1) % 10 == 0:
            with open(args.out_file, 'w') as f:
                json.dump([{'sample_id': args.sample_id, 'qa': qa_out}], f, indent=2)

    out = [{'sample_id': args.sample_id, 'qa': qa_out}]
    with open(args.out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print('wrote %s' % args.out_file)

    # score with the repo's own evaluator
    print('scoring...')
    from task_eval.evaluation import eval_question_answering
    from task_eval.evaluation_stats import analyze_aggr_acc

    f1s, lengths, recall = eval_question_answering(qa_out, PREDICTION_KEY)
    for j in range(len(qa_out)):
        qa_out[j][MODEL_KEY + '_f1'] = round(f1s[j], 3)
    with open(args.out_file, 'w') as f:
        json.dump([{'sample_id': args.sample_id, 'qa': qa_out}], f, indent=2)

    stats_file = args.out_file.replace('.json', '_stats.json')
    analyze_aggr_acc(args.data_file, args.out_file, stats_file,
                     MODEL_KEY, MODEL_KEY + '_f1', rag=True)
    print('wrote %s' % stats_file)


if __name__ == '__main__':
    main()
