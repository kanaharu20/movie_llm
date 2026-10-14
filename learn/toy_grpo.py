"""おもちゃモデルで GRPO + LoRA を一から回す（完成版）。CPU で 1 分以内。

本物（movie_llm/train/grpo.py）と同じ構造を、次の置き換えで小さくしたもの。

| 本物 | おもちゃ |
|---|---|
| 動画 32 フレーム（画像） | 長さ T=16 の「出来事の列」（各フレームに 0〜K-1 の出来事番号が 1 つ） |
| 問い「〇〇の場面は何分何秒か」 | 問い「出来事 X はどのフレームか」 |
| Qwen3-VL-4B（視覚 + 言語） | 埋め込み + 小さな MLP。答えは T 個の位置の確率分布 |
| LoRA（言語モデル側の射影に低ランク差分） | MLP の重みに W + B A（r=2）の差分。B は 0 で初期化 |
| 参照方針 = LoRA を外した元モデル | 差分の係数を 0 にした同じモデル |
| 規則採点 + 許容幅 + 部分点 | |答え − 正解| ≤ tol で 1.0、外れた距離で線形に減る |
| 1 問 8 回答、advantage = r − 平均、k3 KL、動的サンプリング | 同じ |

使い方:
    uv run python learn/toy_grpo.py                       # 既定の設定で学習し、前後の正解率を表示
    uv run python learn/toy_grpo.py --beta 0 --temperature 1.0 --steps 300   # KL なしで多様性の消失を観察
    uv run python learn/toy_grpo.py --partial 0           # 部分点なし（0/1 の報酬）で学習信号がどう変わるか
    uv run python learn/toy_grpo.py --help

読み方: 各ステップに「報酬の平均」「正解率（サンプリング）」「KL」「飛ばしたグループ数」「エントロピー」が出る。
エントロピーは答えの分布のばらつきで、0 に近づくほど「毎回同じ答え」になっている（本物の 2 回目で起きた現象）。
"""

from __future__ import annotations

import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1. データ: 「動画」と「問い」と「正解の時刻」
# ---------------------------------------------------------------------------
T = 16   # フレーム数
K = 6    # 出来事の種類
NOISE = K  # 「ぼやけて見えない」フレームを表す特別な番号


def make_example(rng: random.Random, p_noise: float, max_pos: int = T) -> tuple[list[int], int, int]:
    """(フレーム列, 問い=探す出来事, 正解の位置) を 1 つ作る。探す出来事は列の中に 1 回だけ出る。
    max_pos を小さくすると、正解が前半にしか現れない偏ったデータになる（学習前モデルの「題材の偏り」を作るのに使う）。"""
    target = rng.randrange(K)
    pos = rng.randrange(max_pos)
    frames = []
    for t in range(T):
        if t == pos:
            ev = target
        else:
            ev = rng.choice([e for e in range(K) if e != target])
        # 一部のフレームはぼやけて見えない（正解の位置も含む。見えないときは周りから推測するしかない）
        frames.append(NOISE if rng.random() < p_noise else ev)
    return frames, target, pos


def make_batch(rng: random.Random, n: int, p_noise: float, max_pos: int = T) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, qs, ys = zip(*[make_example(rng, p_noise, max_pos) for _ in range(n)])
    return torch.tensor(xs), torch.tensor(qs), torch.tensor(ys)


# ---------------------------------------------------------------------------
# 2. モデル: 元モデル（固定）+ LoRA（学習する差分）
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """y = x W^T + (α/r) · x A^T B^T。W は固定、A と B だけ学習する。B を 0 で初期化するので、最初は元の層と同じ出力。"""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.1)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.enabled = True  # False にすると参照方針（元モデル）になる

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.enabled:
            y = y + self.scale * F.linear(F.linear(x, self.A), self.B)
        return y


class Policy(nn.Module):
    """各フレームと問いを見て、「正解はこのフレーム」という T 個の位置のロジットを出す。"""

    def __init__(self, d: int = 16) -> None:
        super().__init__()
        self.frame_emb = nn.Embedding(K + 1, d)
        self.query_emb = nn.Embedding(K, d)
        self.pos_emb = nn.Embedding(T, d)
        self.fc1 = nn.Linear(2 * d, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, frames: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        # frames: (B, T), query: (B,) -> logits (B, T)
        f = self.frame_emb(frames) + self.pos_emb(torch.arange(T))  # (B, T, d)
        q = self.query_emb(query).unsqueeze(1).expand(-1, T, -1)      # (B, T, d)
        h = torch.tanh(self.fc1(torch.cat([f, q], dim=-1)))           # (B, T, 32)
        return self.fc2(h).squeeze(-1)                                # (B, T)


def add_lora(model: Policy, r: int, alpha: float) -> list[LoRALinear]:
    """fc1 と fc2 を LoRA 付きに差し替え、元の重みをすべて固定する（本物: q/k/v/o/gate/up/down に r=16）。"""
    for p in model.parameters():
        p.requires_grad_(False)
    model.fc1 = LoRALinear(model.fc1, r, alpha)
    model.fc2 = LoRALinear(model.fc2, r, alpha)
    return [model.fc1, model.fc2]


def set_adapter(loras: list[LoRALinear], enabled: bool) -> None:
    for m in loras:
        m.enabled = enabled


# ---------------------------------------------------------------------------
# 3. 「学習前のモデル」を作る: 偏ったデータで教師あり学習（本物では Qwen の事前学習に相当）
#    正解が前半（0〜7）にしか無いデータで学ぶので、後半に正解がある問題が苦手なモデルになる。
#    本物で「室内の演技映像で学んだことが日本の YouTube 映像に移らない」のと同じ種類のずれを、小さく作る。
# ---------------------------------------------------------------------------
def pretrain(model: Policy, rng: random.Random, steps: int, p_noise: float, max_pos: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x, q, y = make_batch(rng, 64, p_noise, max_pos)
        loss = F.cross_entropy(model(x, q), y)
        opt.zero_grad()
        loss.backward()
        opt.step()


# ---------------------------------------------------------------------------
# 4. 報酬: 本物の score_time と同じ形（許容幅の中は 1.0、外は距離で線形に減る）
# ---------------------------------------------------------------------------
def reward_fn(pred: torch.Tensor, gold: torch.Tensor, tol: int, partial_window: float) -> torch.Tensor:
    dist = (pred - gold).abs().float()
    over = (dist - tol).clamp(min=0)
    if partial_window <= 0:
        return (over == 0).float()
    return (1.0 - over / partial_window).clamp(min=0)


# ---------------------------------------------------------------------------
# 5. GRPO の 1 ステップ
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: Policy, rng: random.Random, n: int, p_noise: float, tol: int) -> float:
    """貪欲法（いちばん確率の高い位置）の正解率。学習前後の比較用（本物の dev / ベンチマーク）。"""
    x, q, y = make_batch(rng, n, p_noise)
    pred = model(x, q).argmax(-1)
    return float(((pred - y).abs() <= tol).float().mean())


def grpo_step(model: Policy, loras: list[LoRALinear], opt, rng: random.Random, cfg) -> dict:
    stats = {"reward": [], "acc": [], "kl": [], "entropy": [], "skipped": 0, "draws": 0}
    opt.zero_grad()
    used = 0
    while used < cfg.prompts_per_step and stats["draws"] < cfg.max_draws:
        stats["draws"] += 1
        x, q, y = make_batch(rng, 1, cfg.p_noise)

        # (a) 現在の方針から G 本サンプリング（本物: model.generate(num_return_sequences=G)）
        with torch.no_grad():
            logits = model(x, q)[0] / cfg.temperature                 # (T,)
            probs = F.softmax(logits, dim=-1)
            answers = torch.multinomial(probs, cfg.group_size, replacement=True)  # (G,)
            ent = float(-(probs * probs.clamp_min(1e-9).log()).sum())
        # (b) 報酬と advantage（グループ平均を引く）
        r = reward_fn(answers, y[0], cfg.tol, cfg.partial_window)
        adv = r - r.mean()
        stats["reward"].append(float(r.mean())); stats["acc"].append(float(((answers - y[0]).abs() <= cfg.tol).float().mean()))
        stats["entropy"].append(ent)
        if float(adv.abs().sum()) < 1e-6:
            stats["skipped"] += 1   # 全員同じ報酬 → 学習信号なし → 別の問題を引く（動的サンプリング）
            continue
        used += 1
        # (c) 参照方針（LoRA を外す）の対数確率。勾配は不要
        set_adapter(loras, False)
        with torch.no_grad():
            ref_lp = F.log_softmax(model(x, q)[0], dim=-1)[answers]
        set_adapter(loras, True)
        # (d) 現在の方針の対数確率。勾配あり
        cur_lp = F.log_softmax(model(x, q)[0], dim=-1)[answers]
        # (e) 損失: −advantage × logp + β × KL(k3 推定量)
        diff = ref_lp.detach() - cur_lp
        kl = torch.exp(diff) - diff - 1
        loss = (-(adv * cur_lp) + cfg.beta * kl).mean() / cfg.prompts_per_step
        loss.backward()
        stats["kl"].append(float(kl.detach().mean()))
    if used:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
    return {k: (sum(v) / len(v) if isinstance(v, list) and v else (v if not isinstance(v, list) else float("nan"))) for k, v in stats.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=400, help="GRPO のステップ数")
    ap.add_argument("--group-size", type=int, default=8, help="G: 1 問あたりの回答数")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-draws", type=int, default=12, help="動的サンプリングで 1 ステップに引く問題数の上限")
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--beta", type=float, default=0.02, help="KL の係数")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tol", type=int, default=1, help="許容幅（フレーム数）")
    ap.add_argument("--partial", type=float, default=3.0, dest="partial_window", help="部分点が 0 になる距離。0 なら 0/1 の報酬")
    ap.add_argument("--p-noise", type=float, default=0.2, help="ぼやけて見えないフレームの割合（難しさ）")
    ap.add_argument("--pretrain-steps", type=int, default=200, help="学習前モデルを作る教師あり学習の回数")
    ap.add_argument("--pretrain-max-pos", type=int, default=T // 2, help="学習前モデルが見た正解位置の範囲（既定: 前半だけ。T にすると偏りなし）")
    ap.add_argument("--lora-r", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    cfg = ap.parse_args()

    rng = random.Random(cfg.seed)
    torch.manual_seed(cfg.seed)
    model = Policy()
    pretrain(model, rng, cfg.pretrain_steps, p_noise=0.1, max_pos=cfg.pretrain_max_pos)
    loras = add_lora(model, cfg.lora_r, alpha=2 * cfg.lora_r)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    before = evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol)
    print(f"パラメータ {n_all}、学習するのは LoRA の {n_train}（{n_train / n_all * 100:.0f}%）")
    print(f"学習前の正解率（貪欲法、2000 問、許容幅 {cfg.tol}）: {before * 100:.1f}%   当て推量: {(2 * cfg.tol + 1) / T * 100:.1f}%\n")
    print(f"{'step':>5} {'報酬':>6} {'正解率':>6} {'KL':>7} {'entropy':>7} {'draws':>5} {'skip':>4} {'貪欲法':>6}")
    print("（報酬・正解率はサンプリングした回答の平均。貪欲法は 2000 問の評価で、学習前後の比較に使う値）")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    for step in range(1, cfg.steps + 1):
        st = grpo_step(model, loras, opt, rng, cfg)
        if step % cfg.log_every == 0 or step == 1:
            greedy = evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol)
            print(f"{step:5d} {st['reward']:6.3f} {st['acc']:6.3f} {st['kl']:7.4f} {st['entropy']:7.3f} {st['draws']:5d} {st['skipped']:4d} {greedy * 100:5.1f}%")

    after = evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol)
    set_adapter(loras, False)
    ref = evaluate(model, random.Random(123), 2000, cfg.p_noise, cfg.tol)
    print(f"\n学習後の正解率: {after * 100:.1f}%（学習前 {before * 100:.1f}%、LoRA を外すと {ref * 100:.1f}% に戻る）")
    print(f"最終ステップのエントロピー {st['entropy']:.3f}（最大 {math.log(T):.3f}）。0 に近いほど毎回同じ答え")


if __name__ == "__main__":
    main()
