"""おもちゃモデルで GRPO + LoRA を自分で組み立てる（穴埋め版）。

学習の中身にあたる 4 か所が `raise NotImplementedError` になっている。toy_grpo.py（完成版）を見ずに書いてみて、
`--check` で答え合わせをしてから学習を回す。

    uv run python learn/toy_grpo_exercise.py --check     # 4 か所を完成版と突き合わせる
    uv run python learn/toy_grpo_exercise.py             # 学習を回す（toy_grpo.py と同じ引数が使える）

穴埋めの順番（易しい順）:
    課題 1  reward_fn      : 距離から報酬を決める（許容幅の中は 1.0、外は線形に減って 0）
    課題 2  advantage      : グループ平均を引く。全員同じなら「学習信号なし」と判定する
    課題 3  LoRALinear     : y = W x + (α/r) B A x。B は 0 で初期化
    課題 4  grpo_loss      : −advantage × logp + β × KL（k3 推定量 exp(d) − d − 1、d = logp_ref − logp）
"""

from __future__ import annotations

import argparse
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import learn.toy_grpo as ref  # データ・モデルの土台・評価は完成版のものをそのまま使う
from learn.toy_grpo import K, T, Policy, evaluate, make_batch, pretrain  # noqa: F401


# ---------------------------------------------------------------------------
# 課題 1: 報酬
# ---------------------------------------------------------------------------
def reward_fn(pred: torch.Tensor, gold: torch.Tensor, tol: int, partial_window: float) -> torch.Tensor:
    """pred: (G,) サンプリングした答えの位置、gold: 正解の位置（スカラー）。
    返り値: (G,) の報酬。|pred − gold| ≤ tol なら 1.0。外れた距離 over = |pred − gold| − tol に対して
    1 − over / partial_window を 0 で下から切る。partial_window ≤ 0 のときは 0/1 の報酬（許容幅の中だけ 1.0）。
    ヒント: (pred - gold).abs().float()、.clamp(min=0)、(x == 0).float()
    """
    raise NotImplementedError("課題 1")


# ---------------------------------------------------------------------------
# 課題 2: advantage
# ---------------------------------------------------------------------------
def advantage(r: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """r: (G,) 報酬。返り値: (advantage (G,), 学習信号があるか)。
    advantage = r − r の平均（標準偏差では割らない）。全員同じ報酬（advantage の絶対値の和が 1e-6 未満）なら信号なし。
    """
    raise NotImplementedError("課題 2")


# ---------------------------------------------------------------------------
# 課題 3: LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """元の線形層 base（固定）に、低ランクの差分 (α/r) · B A を足す。
    A: (r, in_features) を小さな乱数で、B: (out_features, r) を 0 で初期化する（最初は元の層と同じ出力になる）。
    enabled が False のときは差分を足さない（参照方針 = 元モデル）。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.enabled = True
        # TODO: self.A と self.B を nn.Parameter で作る
        raise NotImplementedError("課題 3: __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: y = base(x)、enabled なら + scale * (x A^T) B^T。ヒント: F.linear(x, self.A) は x A^T
        raise NotImplementedError("課題 3: forward")


# ---------------------------------------------------------------------------
# 課題 4: 損失
# ---------------------------------------------------------------------------
def grpo_loss(adv: torch.Tensor, cur_lp: torch.Tensor, ref_lp: torch.Tensor, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """adv: (G,)、cur_lp: 現在の方針での各答えの対数確率 (G,)（勾配あり）、ref_lp: 参照方針での対数確率 (G,)（勾配なし）。
    返り値: (損失（スカラー）, KL（G,）)。
    d = ref_lp − cur_lp、KL = exp(d) − d − 1（k3 推定量。常に 0 以上）、損失 = mean(−adv × cur_lp + beta × KL)。
    ref_lp は .detach() しておく。
    """
    raise NotImplementedError("課題 4")


# ---------------------------------------------------------------------------
# ここから下は完成版と同じ（組み立て）
# ---------------------------------------------------------------------------
def add_lora(model: Policy, r: int, alpha: float) -> list[LoRALinear]:
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc1 = LoRALinear(model.fc1, r, alpha)
    model.fc2 = LoRALinear(model.fc2, r, alpha)
    return [model.fc1, model.fc2]


def set_adapter(loras: list[LoRALinear], enabled: bool) -> None:
    for m in loras:
        m.enabled = enabled


def grpo_step(model: Policy, loras: list[LoRALinear], opt, rng: random.Random, cfg) -> dict:
    stats = {"reward": [], "acc": [], "kl": [], "entropy": [], "skipped": 0, "draws": 0}
    opt.zero_grad()
    used = 0
    while used < cfg.prompts_per_step and stats["draws"] < cfg.max_draws:
        stats["draws"] += 1
        x, q, y = make_batch(rng, 1, cfg.p_noise)
        with torch.no_grad():
            probs = F.softmax(model(x, q)[0] / cfg.temperature, dim=-1)
            answers = torch.multinomial(probs, cfg.group_size, replacement=True)
            stats["entropy"].append(float(-(probs * probs.clamp_min(1e-9).log()).sum()))
        r = reward_fn(answers, y[0], cfg.tol, cfg.partial_window)
        adv, has_signal = advantage(r)
        stats["reward"].append(float(r.mean())); stats["acc"].append(float(((answers - y[0]).abs() <= cfg.tol).float().mean()))
        if not has_signal:
            stats["skipped"] += 1
            continue
        used += 1
        set_adapter(loras, False)
        with torch.no_grad():
            ref_lp = F.log_softmax(model(x, q)[0], dim=-1)[answers]
        set_adapter(loras, True)
        cur_lp = F.log_softmax(model(x, q)[0], dim=-1)[answers]
        loss, kl = grpo_loss(adv, cur_lp, ref_lp, cfg.beta)
        (loss / cfg.prompts_per_step).backward()
        stats["kl"].append(float(kl.detach().mean()))
    if used:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return {k: (sum(v) / len(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else float("nan"))) for k, v in stats.items()}


def check() -> None:
    """4 つの穴埋めを完成版と突き合わせる。"""
    g = torch.Generator().manual_seed(0)
    ok = True

    def report(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= passed
        print(f"  {'OK ' if passed else 'NG '} {name} {detail}")

    try:
        pred = torch.randint(0, T, (8,), generator=g); gold = torch.tensor(7)
        for pw in (3.0, 0.0):
            a, b = reward_fn(pred, gold, 1, pw), ref.reward_fn(pred, gold, 1, pw)
            report(f"課題 1 reward_fn (partial={pw})", torch.allclose(a, b), f"あなた {a.tolist()} / 正解 {b.tolist()}" if not torch.allclose(a, b) else "")
    except NotImplementedError as e:
        report("課題 1", False, f"未実装 ({e})")
    try:
        r = torch.tensor([1.0, 0.1, 1.1, 0.1, 0.5, 1.1, 0.1, 0.1])
        a, sig = advantage(r)
        report("課題 2 advantage", torch.allclose(a, r - r.mean()) and sig is True)
        a0, sig0 = advantage(torch.full((8,), 1.1))
        report("課題 2 全員同じ報酬 → 信号なし", sig0 is False and float(a0.abs().sum()) < 1e-6)
    except NotImplementedError as e:
        report("課題 2", False, f"未実装 ({e})")
    try:
        torch.manual_seed(1); base = nn.Linear(6, 4); x = torch.randn(3, 6)
        torch.manual_seed(2); mine = LoRALinear(base, 2, 4.0)
        torch.manual_seed(2); theirs = ref.LoRALinear(base, 2, 4.0)
        report("課題 3 初期状態で元の層と同じ出力", torch.allclose(mine(x), base(x)))
        with torch.no_grad():
            mine.B.copy_(torch.randn(4, 2)); theirs.B.copy_(mine.B); theirs.A.copy_(mine.A)
        report("課題 3 差分を足した出力", torch.allclose(mine(x), theirs(x)))
        mine.enabled = False
        report("課題 3 enabled=False で元の層に戻る", torch.allclose(mine(x), base(x)))
        report("課題 3 学習するのは A と B だけ", {n for n, p in mine.named_parameters() if p.requires_grad} == {"A", "B"})
    except NotImplementedError as e:
        report("課題 3", False, f"未実装 ({e})")
    try:
        adv = torch.tensor([0.5, -0.5, 0.5, -0.5]); cur = torch.tensor([-1.0, -2.0, -0.5, -3.0], requires_grad=True); refl = torch.tensor([-1.2, -1.5, -0.7, -2.5])
        loss, kl = grpo_loss(adv, cur, refl, 0.02)
        d = refl - cur.detach(); kl_ref = torch.exp(d) - d - 1; loss_ref = (-(adv * cur.detach()) + 0.02 * kl_ref).mean()
        report("課題 4 KL(k3)", torch.allclose(kl.detach(), kl_ref))
        report("課題 4 損失", torch.allclose(loss.detach(), loss_ref))
        loss.backward(); report("課題 4 cur_lp に勾配が流れる", cur.grad is not None)
    except NotImplementedError as e:
        report("課題 4", False, f"未実装 ({e})")
    print("\nすべて OK" if ok else "\nNG があります。ヒントは各関数の docstring")
    sys.exit(0 if ok else 1)


def main() -> None:
    if "--check" in sys.argv:
        check()
        return
    # 完成版の main と同じ流れ（引数も同じ）
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4); ap.add_argument("--max-draws", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-3); ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--tol", type=int, default=1)
    ap.add_argument("--partial", type=float, default=3.0, dest="partial_window"); ap.add_argument("--p-noise", type=float, default=0.2)
    ap.add_argument("--pretrain-steps", type=int, default=200); ap.add_argument("--pretrain-max-pos", type=int, default=T // 2)
    ap.add_argument("--lora-r", type=int, default=2); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=50)
    cfg = ap.parse_args()
    rng = random.Random(cfg.seed); torch.manual_seed(cfg.seed)
    model = Policy(); pretrain(model, rng, cfg.pretrain_steps, p_noise=0.1, max_pos=cfg.pretrain_max_pos)
    loras = add_lora(model, cfg.lora_r, alpha=2 * cfg.lora_r)
    before = evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol)
    print(f"学習前の正解率: {before * 100:.1f}%")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    for step in range(1, cfg.steps + 1):
        st = grpo_step(model, loras, opt, rng, cfg)
        if step % cfg.log_every == 0 or step == 1:
            print(f"step {step:4d} reward {st['reward']:.3f} acc {st['acc']:.3f} kl {st['kl']:.4f} entropy {st['entropy']:.3f} draws {st['draws']} skip {st['skipped']}  貪欲法 {evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol) * 100:.1f}%")
    print(f"学習後の正解率: {evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol) * 100:.1f}%（学習前 {before * 100:.1f}%）")


if __name__ == "__main__":
    main()
