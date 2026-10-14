"""動画 QA 用の最小限の GRPO（LoRA）。

TRL の GRPOTrainer は動画入力（フレーム列 + 時刻）に対応していないので、推論と同じ前処理を使って自前で回す。

1 ステップの流れ（prompts_per_step 問ぶんを勾配累積）:
  1. 1 問につき G 本の回答をサンプリング（generate、num_return_sequences）
  2. 正誤（規則採点）と形式で報酬を付け、グループ内で平均を引いて advantage にする
  3. 現在の方針（LoRA あり）で各回答トークンの対数確率を計算し、-advantage × logp をトークン平均したものを損失にする
     （生成直後に 1 回だけ更新する on-policy なので、確率比は 1 でクリップは不要）
  4. 参照方針（LoRA を外した元のモデル）との KL を k3 推定量で加える（報酬ハッキング対策）
  5. 勾配を累積して 1 回更新。LoRA の重みだけが更新される

報酬の形式は問題の type で切り替える:
  - multi_choice: 記号が正解なら 1.0
  - numeric（数を答える）: 一致で 1.0、差 1 で numeric_partial（既定 0.5）、それ以外 0
  - time（場面の時刻を当てる）: 答えの時刻が start〜end ± 許容幅に入れば 1.0、外れた秒数で線形に減る（movie_llm.eval.timeloc.score_time）。
    問題には start / end / video_duration（秒）が要る。指示文は評価と同じ build_time_prompt、根拠を書いてから答えるので max_new_tokens は 96 程度
  - いずれも、読み取れる形で答えたら format_bonus を加える

動的サンプリング（DAPO）: 全回答が同じ報酬の問題は学習信号が無いので飛ばし、別の問題を引いて
prompts_per_step 問ぶんの信号が集まるまで続ける（上限 max_draws 問）。
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


def _ignore_old_torchao() -> None:
    """Colab 既定の古い torchao (0.10) が入っていると peft が LoRA を当てる途中で例外を出す。
    量子化は使わないので、古い版のときは「torchao は無い」と peft に思わせる。"""
    try:
        from importlib.metadata import version

        major, minor = (int(x) for x in version("torchao").split(".")[:2])
        if (major, minor) < (0, 16):
            import peft.import_utils as piu
            import peft.tuners.lora.torchao as plt

            piu.is_torchao_available = lambda: False
            plt.is_torchao_available = lambda: False
    except Exception:
        pass


_ignore_old_torchao()
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor

from movie_llm.eval.rules import extract_choice, gold_letter, normalize, score_numeric
from movie_llm.eval.timeloc import build_time_prompt, extract_time, score_time, tolerance
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
    format_bonus: float = 0.1    # 記号（または数字）を読み取れる形で答えたら加点
    numeric_partial: float = 0.5 # 数を答える問題で、正解との差が 1 のときの報酬
    # 動的サンプリング: 学習信号の無い問題を飛ばして別の問題を引く。1 ステップで引く問題数の上限
    max_draws: int = 12
    gradient_checkpointing: bool = True


def reward_fn(pred: str, item: dict, format_bonus: float, numeric_partial: float = 0.5, num_frames: int = 32) -> tuple[float, bool, bool]:
    """(報酬, 正解か, 形式が読めたか)。正誤の報酬（0〜1）+ 形式 format_bonus。"""
    short = len(normalize(pred)) <= 40
    if item["type"] == "time":
        t = extract_time(pred)
        score, hit, _ = score_time(t, float(item["start"]), float(item["end"]), tolerance(float(item["video_duration"]), num_frames))
        parsable = t is not None and "答え" in pred and len(normalize(pred)) <= 300
        return score + (format_bonus if parsable else 0.0), hit, parsable
    if item["type"] == "numeric":
        score, correct, parsed = score_numeric(item["answer"], pred, numeric_partial)
        parsable = parsed and short
        return score + (format_bonus if parsable else 0.0), correct, parsable
    letter = extract_choice(pred, item["choices"])
    gold = gold_letter(item["answer"], item["choices"])
    parsable = letter is not None and short
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
        if item["type"] == "time":
            prompt = build_time_prompt(item["question"])
            item.setdefault("video_duration", float(meta.duration))
        else:
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
                # 末尾 k 位置のロジットだけ計算する（prompt 部分の語彙サイズ×長さの巨大なテンソルを作らない）
                k = L - plen + 1
                logits = self.model(**kw, use_cache=False, logits_to_keep=k).logits  # (B, k, V) = 位置 plen-1 .. L-1
                # 位置 t のロジットは t+1 のトークンを予測する
                lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)  # 位置 plen-1 .. L-2
                tgt = ids[:, plen:L]
                tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-plen)
            for i, c in enumerate(chunk):
                results.append(tok_lp[i, : c.shape[0]])
            del logits, lp
        return results

    # ---- 1 ステップ -------------------------------------------------------
    def train_step(self, draw) -> dict:
        """draw() で問題を 1 つずつ引き、学習信号のある問題が prompts_per_step 問集まるまで続けて 1 回更新する。"""
        cfg = self.cfg
        self.opt.zero_grad(set_to_none=True)
        stats = {"reward": [], "acc": [], "parsable": [], "kl": [], "loss": [], "len": [], "zero_adv_groups": 0, "draws": 0}
        used = 0
        pending: list[tuple[dict, dict, list[torch.Tensor], torch.Tensor]] = []
        # 1) 信号のあるグループが揃うまでサンプリング（勾配計算はまだしない）
        while used < cfg.prompts_per_step and stats["draws"] < cfg.max_draws:
            item = draw()
            stats["draws"] += 1
            enc = self._encode_prompt(item)
            comps = self._sample(enc)
            preds = [self.processor.decode(c, skip_special_tokens=True).strip() for c in comps]
            rewards, corrects, parsables = zip(*[reward_fn(p, item, cfg.format_bonus, cfg.numeric_partial, cfg.num_frames) for p in preds])
            r = torch.tensor(rewards, dtype=torch.float32, device=self.device)
            adv = r - r.mean()
            if cfg.std_normalize:
                adv = adv / (r.std() + 1e-4)
            stats["reward"].append(float(r.mean())); stats["acc"].append(sum(corrects) / len(corrects))
            stats["parsable"].append(sum(parsables) / len(parsables)); stats["len"].append(sum(c.shape[0] for c in comps) / len(comps))
            if float(adv.abs().sum()) < 1e-6:
                stats["zero_adv_groups"] += 1
                continue  # 全員同じ報酬なら学習信号なし（この問題は飛ばし、次を引く）
            pending.append((item, enc, comps, adv))
            used += 1
        if not pending:
            self.step += 1
            return {k: (sum(v) / len(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else None)) for k, v in stats.items()}

        # 2) 集まったグループで勾配を累積
        for item, enc, comps, adv in pending:
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
            loss = loss / total_tokens / len(pending)  # 問題数で割って勾配累積
            loss.backward()
            stats["loss"].append(float(loss.detach()) * len(pending)); stats["kl"].append(kl_sum / total_tokens)
            del cur_lps, ref_lps, loss
        del pending

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
        order: list[dict] = []

        def draw() -> dict:
            # 全問を使い切るまで重複なく引く（使い切ったら並べ直す）
            if not order:
                order.extend(rng.sample(self.items, len(self.items)))
            return order.pop()

        while self.step < cfg.max_steps:
            try:
                st = self.train_step(draw)
            except torch.cuda.OutOfMemoryError:
                print("OOM: この step を飛ばす（micro_batch か group_size を下げる）")
                torch.cuda.empty_cache()
                self.opt.zero_grad(set_to_none=True)
                continue
            st.update({"step": self.step, "elapsed": round(time.time() - t0)})
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(st, ensure_ascii=False) + "\n")
            print(f"step {self.step:4d}  reward {st['reward']:.3f}  acc {st['acc']:.3f}  parsable {st['parsable']:.2f}  kl {st['kl'] if st['kl'] is not None else float('nan'):.4f}  loss {st['loss'] if st['loss'] is not None else float('nan'):.4f}  len {st['len']:.1f}  draws {st['draws']}  skip {st['zero_adv_groups']}  {st['elapsed']}s")
            if self.step % cfg.save_every == 0 or self.step >= cfg.max_steps:
                self.save()
