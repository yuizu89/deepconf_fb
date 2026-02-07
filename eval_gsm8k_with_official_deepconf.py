from __future__ import annotations

import argparse
import json
import random
import re
from typing import Dict, List, Optional

from datasets import load_dataset

from deepconf import DeepThinkLLM  # facebookresearch/deepconf
from vllm import SamplingParams


_HASH_RE = re.compile(r"####\s*([^\n\r]*)")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOX_RE = re.compile(r"\\boxed\s*[\{\s]")  # existence check only


def extract_last_number(text: str) -> Optional[str]:
    nums = _NUM_RE.findall(text)
    return nums[-1] if nums else None


def extract_after_hashes(text: str) -> Optional[str]:
    m = _HASH_RE.findall(text)
    if not m:
        return None
    s = m[-1].strip()
    return s if s else None


def normalize_gsm8k_answer(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = x.strip()
    n = extract_last_number(x)
    return n if n is not None else x


def extract_gsm8k_gold(answer_field: str) -> Optional[str]:
    return normalize_gsm8k_answer(extract_after_hashes(answer_field))


def build_messages_boxed(question: str) -> List[Dict[str, str]]:
    # 公式deepconfの extractor は \boxed{} を期待するため、boxed形式で指示するのが安全 :contentReference[oaicite:2]{index=2}
    user = (
        "Solve the following grade-school math problem.\n\n"
        f"{question}\n\n"
        "Show your reasoning briefly, then give the final answer as:\n"
        "\\boxed{<number>}\n"
    )
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user},
    ]


def maybe_patch_deepconf_extract_answer(enable: bool):
    if not enable:
        return
    import deepconf.utils as dcu

    orig = dcu.extract_answer

    def extract_answer_patched(text: str):
        # 1) #### を優先（あなたのgsm8k抽出互換）
        s = extract_after_hashes(text)
        if s is not None:
            return s
        # 2) 既存 \boxed{}
        return orig(text)

    dcu.extract_answer = extract_answer_patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--max_examples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)

    ap.add_argument("--window_size", type=int, default=2048)  # deepconf の window_size :contentReference[oaicite:3]{index=3}
    ap.add_argument("--patch_extract_hashes", type=int, default=0)  # 1で #### 対応を追加
    ap.add_argument("--out_jsonl", type=str, default="gsm8k_deepconf_official.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)

    # optional: #### を deepconf 側が拾えるようにする
    maybe_patch_deepconf_extract_answer(bool(args.patch_extract_hashes))

    # DeepThinkLLM: vLLM wrapper（trust_remote_code=True など既定） :contentReference[oaicite:4]{index=4}
    deep_llm = DeepThinkLLM(args.model)

    ds = load_dataset("gsm8k", "main", split=args.split)
    n = min(len(ds), int(args.max_examples))

    # vLLM sampling params（deepthink に渡す） :contentReference[oaicite:5]{index=5}
    sp = SamplingParams(
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        max_tokens=int(args.max_tokens),
        repetition_penalty=args.repetition_penalty,
        seed=int(args.seed),
        logprobs=20,  # DeepConfはtop-k logprobsを使う前提 :contentReference[oaicite:6]{index=6}
    )

    # 投票method名（公式出力） :contentReference[oaicite:7]{index=7}
    methods = [
        "majority",
        "mean_confidence_weighted",
        "tail_confidence_weighted",
        "bottom_window_weighted",
        "min_window_weighted",
        "top10_tail_filtered",
        "top10_bottom_window_filtered",
    ]

    correct = {m: 0 for m in methods}
    total = 0

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i in range(n):
            ex = ds[i]
            q = ex["question"]
            gold = extract_gsm8k_gold(ex["answer"])

            messages = build_messages_boxed(q)

            # deepconf は「chat template適用済みの prompt string」を要求 :contentReference[oaicite:8]{index=8}
            prompt = deep_llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # ← 明示的にON（Qwen3のハードスイッチ）
            )

            out = deep_llm.deepthink(
                prompt=prompt,
                mode="offline",
                budget=int(args.K),
                window_size=int(args.window_size),
                sampling_params=sp,
                compute_multiple_voting=True,
            )

            print("voting keys:", sorted(out.voting_results.keys()))

            rec = {"idx": i, "gold": gold, "question": q, "K": int(args.K), "results": {}}

            for m in methods:
                r = out.voting_results.get(m, None) or {}
                ans = r.get("answer", None)
                pred = normalize_gsm8k_answer(ans) if ans is not None else None
                ok = (pred is not None) and (gold is not None) and (str(pred) == str(gold))
                correct[m] += int(ok)
                rec["results"][m] = {"answer": ans, "pred": pred, "correct": bool(ok), "meta": r}

            total += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                msg = [f"[{i+1}/{n}]"]
                msg += [f"{m}={correct[m]/max(1,total):.4f}" for m in methods[:3]]
                msg += [f"... bottom={correct['bottom_window_confidence_weighted']/max(1,total):.4f}"]
                print(" ".join(msg))

    print("Done.")
    for m in methods:
        print(f"{m}: {correct[m]/max(1,total):.4f} ({correct[m]}/{total})")
    print(f"Wrote: {args.out_jsonl}")


if __name__ == "__main__":
    main()
