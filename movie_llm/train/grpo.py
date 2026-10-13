"""動画 QA 用の最小限の GRPO（LoRA）。

TRL の GRPOTrainer は動画入力（フレーム列 + 時刻）に対応していないので、推論と同じ前処理を使って自前で回す。

1 ステップの流れ（prompts_per_step 問ぶんを勾配累積）:
  1. 1 問につき G 本の回答をサンプリング（generate、num_return_sequences）
  2. 正誤（規則採点）と形式で報酬を付け、グループ内で平均を引いて advantage にする
  3. 現在の方針（LoRA あり）で各回答トークンの対数確率を計算し、-advantage × logp をトークン平均したものを損失にする
     （生成直後に 1 回だけ更新する on-policy なので、確率比は 1 でクリップは不要）
  4. 参照方針（LoRA を外した元のモデル）との KL を k3 推定量で加える（報酬ハッキング対策）
  5. 勾配を累積して 1 回更新。LoRA の重みだけが更新される
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from movie_llm.eval.rules import extract_choice, gold_letter, normalize
from movie_llm.infer.fetch import video_path
from movie_llm.infer.qwen3vl import build_prompt, pick_device
from movie_llm.infer.video import sample_frames_cached


@dataclass
class GRPOConfig:
    model_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    out_dir: str = "results/grpo"
    video_dir: str = "data/jvqa/videos"
    frames_dir: str | None = "data/jvqa/frames"
    num_frames: int = 16
    max_side: int = 640
    max_pixels: int = 256 * 28 * 28
    # 生成
    group_size: int = 8          # G: 1 問あたりの回答数
    max_new_tokens: int = 16     # 記号 1 文字で答えさせるので短くてよい
    temperature: float = 1.0
    # 学習
    prompts_per_step: int = 4    # 1 回の更新に使う問題数
    micro_batch: int = 4         # 対数確率の計算で一度に流す回答数（メモリに応じて）
    lr: float = 2e-5
    beta: float = 0.02           # KL の係数
    std_normalize: bool = False  # advantage を標準偏差で割るか（Dr. GRPO に倣い既定は割らない）
    max_steps: int = 100
    save_every: int = 20
    seed: int = 0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    # 報酬
    format_bonus: float = 0.1    # 記号を読み取れる形で答えたら加点
    gradient_checkpointing: bool = True


def reward_fn(pred: str, item: dict, format_bonus: float) -> tuple[float, bool, bool]:
    """(報酬, 正解か, 形式が読めたか)。正解 1.0 + 形式 format_bonus。"""
    letter = extract_choice(pred, item["choices"])
    gold = gold_letter(item["answer"], item["choices"])
    parsable = letter is not None and len(normalize(pred)) <= 40
    correct = letter is not None and letter == gold
    return float(correct) + (format_bonus if parsable else 0.0), correct, parsable


class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig, items: list[dict]) -> None:
        self.cfg = cfg
        self.items = items
        self.device = pick_device()
        self.dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        if self.device == "cuda" and torch.cuda.get_device_capability(0)[0] < 8:
            self.dtype = torch.float16
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.processor = AutoProcessor.from_pretrained(cfg.model_id)
        self.processor.video_processor.size = {"shortest_edge": 128 * 32 * 32, "longest_edge": cfg.max_pixels * cfg.num_frames}
        base = AutoModelForImageTextToText.from_pretrained(cfg.model_id, dtype=self.dtype)
        base.to(self.device)
        if cfg.gradient_checkpointing:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.config.use_cache = False
        # 視覚側は固定（LoRA の対象名は言語モデル側だけに当たる。名前は q_proj など）
        for p in base.parameters():
            p.requires_grad_(False)
        lcfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=cfg.lora_targets, task_type="CAUSAL_LM")
        self.model: PeftModel = get_peft_model(base, lcfg)
        if cfg.gradient_checkpointing:
            self.model.enable_input_require_grads()
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"device={self.device} dtype={self.dtype} LoRA params={n_train / 1e6:.1f}M")
        self.opt = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=0.0)
        self.step = 0
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_path = self.out / "train_log.jsonl"
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))

    # ---- 入力の組み立て -------------------------------------------------
    def _encode_prompt(self, item: dict) -> dict:
        vp = video_path(item["video_id"], Path(self.cfg.video_dir))
        if vp is None:
            raise FileNotFoundError(item["video_id"])
        frames, meta = sample_frames_cached(str(vp), self.cfg.num_frames, self.cfg.max_side, self.cfg.frames_dir)
        prompt = build_prompt(item["question"], item["type"], item["choices"])
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self.processor(text=[text], videos=[frames], video_metadata=[meta], do_sample_frames=False, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def _sample(self, enc: dict) -> list[torch.Tensor]:
        """G 本の回答トークン列（EOS まで、パディング無し）を返す。"""
        self.model.eval()
        self.model.config.use_cache = True
        out = self.model.generate(
            **enc, do_sample=True, temperature=self.cfg.temperature, top_p=1.0,
            max_new_tokens=self.cfg.max_new_tokens, num_return_sequences=self.cfg.group_size,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        self.model.config.use_cache = False
        start = enc["input_ids"].shape[1]
        eos_ids = set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list) else [self.model.generation_config.eos_token_id])
        comps = []
        for row in out[:, start:]:
            ids = row.tolist()
            cut = len(ids)
            for i, t in enumerate(ids):
                if t in eos_ids:
                    cut = i + 1  # EOS も学習対象に含める（止め方を学ぶ）
                    break
            comps.append(row[:cut])
        return comps

    def _completion_logprobs(self, enc: dict, comps: list[torch.Tensor], with_grad: bool) -> list[torch.Tensor]:
        """各回答のトークンごとの対数確率。prompt + 回答 を micro_batch ずつまとめて流す。"""
        pid = enc["input_ids"][0]
        plen = pid.shape[0]
        pad = self.processor.tokenizer.pad_token_id
        results: list[torch.Tensor] = []
        for s in range(0, len(comps), self.cfg.micro_batch):
            chunk = comps[s : s + self.cfg.micro_batch]
            B = len(chunk)
            L = plen + max(c.shape[0] for c in chunk)
            ids = torch.full((B, L), pad, dtype=torch.long, device=self.device)
            att = torch.zeros((B, L), dtype=torch.long, device=self.device)
            # どのトークンが動画かを示す列（M-RoPE の位置計算に必要）。回答部分はすべて文章 (0)
            mm = torch.zeros((B, L), dtype=torch.long, device=self.device)
            for i, c in enumerate(chunk):
                ids[i, :plen] = pid
                ids[i, plen : plen + c.shape[0]] = c
                att[i, : plen + c.shape[0]] = 1
                if "mm_token_type_ids" in enc:
                    mm[i, :plen] = enc["mm_token_type_ids"][0]
            kw = {
                "input_ids": ids,
                "attention_mask": att,
                "pixel_values_videos": enc["pixel_values_videos"].repeat(B, 1),
                "video_grid_thw": enc["video_grid_thw"].repeat(B, 1),
            }
            if "mm_token_type_ids" in enc:
                kw["mm_token_type_ids"] = mm
            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                logits = self.model(**kw, use_cache=False).logits  # (B, L, V)
                # 位置 t のロジットは t+1 のトークンを予測する
                lp = torch.log_softmax(logits[:, plen - 1 : L - 1].float(), dim=-1)
                tgt = ids[:, plen:L]
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-plen)
            for i, c in enumerate(chunk):
                results.append(tok_lp[i, : c.shape[0]])
            del logits, lp
        return results

    # ---- 1 ステップ -------------------------------------------------------
    def train_step(self, batch: list[dict]) -> dict:
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        stats = {"reward": [], "acc": [], "parsable": [], "kl": [], "loss": [], "len": [], "zero_adv_groups": 0}
        for item in batch:
            enc = self._encode_prompt(item)
            comps = self._sample(enc)
            preds = [self.processor.decode(c, skip_special_tokens=True).strip() for c in comps]
            rewards, corrects, parsables = zip(*[reward_fn(p, item, cfg.format_bonus) for p in preds])
            r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            adv = r - r.mean()
            if cfg.std_normalize:
                adv = adv / (r.std() + 1e-4)
            stats["reward"].append(float(r.mean())); stats["acc"].append(sum(corrects) / len(corrects))
            stats["parsable"].append(sum(parsables) / len(parsables)); stats["len"].append(sum(c.shape[0] for c in comps) / len(comps))
            if float(adv.abs().sum()) == 0.0:
                stats["zero_adv_groups"] += 1
                continue  # 全員同じ報酬なら学習信号なし（この問題は飛ばす）

            # 参照方針（LoRA を外す）の対数確率
            self.model.eval()
            with self.model.disable_adapter():
                ref_lps = self._completion_logprobs(enc, comps, with_grad=False)
            # 現在の方針
            self.model.train()
            cur_lps = self._completion_logprobs(enc, comps, with_grad=True)

            total_tokens = sum(c.shape[0] for c in comps)
            loss = torch.zeros((), device=self.device)
            kl_sum = 0.0
            for a, cur, ref in zip(adv, cur_lps, ref_lps):
                # 方針勾配: -A × logp（トークンごと）。KL: k3 推定量 exp(ref-cur) - (ref-cur) - 1
                diff = ref.detach() - cur
                kl = torch.exp(diff) - diff - 1
                loss = loss + (-(a * cur) + cfg.beta * kl).sum()
                kl_sum += float(kl.detach().sum())
            loss = loss / total_tokens / len(batch)  # 問題数で割って勾配累積
            loss.backward()
            stats["loss"].append(float(loss) * len(batch)); stats["kl"].append(kl_sum / total_tokens)
            del cur_lps, ref_lps, loss

        torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1.0)
        self.opt.step()
        self.step += 1
        return {k: (sum(v) / len(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else None)) for k, v in stats.items()}

    def save(self) -> None:
        d = self.out / f"step_{self.step:05d}"
        self.model.save_pretrained(str(d))
        (self.out / "latest.txt").write_text(str(d))

    def resume(self) -> None:
        latest = self.out / "latest.txt"
        if latest.exists():
            d = Path(latest.read_text().strip())
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            sd = load_file(str(d / "adapter_model.safetensors"))
            set_peft_model_state_dict(self.model, sd)
            self.step = int(d.name.split("_")[1])
            print(f"再開: {d}（step {self.step}）")

    def train(self) -> None:
        cfg = self.cfg
        self.resume()
        rng = random.Random(cfg.seed + self.step)
        t0 = time.time()
        while self.step < cfg.max_steps:
            batch = rng.sample(self.items, cfg.prompts_per_step)
            try:
                st = self.train_step(batch)
            except torch.cuda.OutOfMemoryError:
                print("OOM: この step を飛ばす（micro_batch か group_size を下げる）")
                torch.cuda.empty_cache()
                self.opt.zero_grad(set_to_none=True)
                continue
            st.update({"step": self.step, "elapsed": round(time.time() - t0)})
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(st, ensure_ascii=False) + "\n")
            print(f"step {self.step:4d}  reward {st['reward']:.3f}  acc {st['acc']:.3f}  parsable {st['parsable']:.2f}  kl {st['kl'] if st['kl'] is not None else float('nan'):.4f}  loss {st['loss'] if st['loss'] is not None else float('nan'):.4f}  len {st['len']:.1f}  skip {st['zero_adv_groups']}  {st['elapsed']}s")
            if self.step % cfg.save_every == 0 or self.step >= cfg.max_steps:
                self.save()
