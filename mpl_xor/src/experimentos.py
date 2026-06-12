"""
experimentos.py
Reproduz os experimentos do artigo + variações do roteiro.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from mlp import MLP, sigmoid_binary, sigmoid_bipolar, tanh

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# paleta
COLORS = ["#5B4FCF", "#E05A3A", "#3AAE8A", "#C9A227", "#A259CC"]
plt.rcParams.update({
    "figure.facecolor": "#0F0F0F",
    "axes.facecolor":   "#1A1A1A",
    "axes.edgecolor":   "#333333",
    "axes.labelcolor":  "#CCCCCC",
    "xtick.color":      "#888888",
    "ytick.color":      "#888888",
    "text.color":       "#CCCCCC",
    "grid.color":       "#2A2A2A",
    "grid.linewidth":   0.6,
    "font.family":      "monospace",
    "font.size":        9,
    "legend.framealpha": 0.15,
})

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]],         dtype=float)


# Subclasse com critério de convergência: todos os outputs dentro de 0.1 do alvo
class MLPExp(MLP):
    def train(self, X, y, tolerance=0.001, max_epochs=10_000):
        history = []
        for epoch in range(1, max_epochs + 1):
            self.forward(X)
            self.backward(X, y)
            total_error = np.sum(np.abs(y - self.a2))
            history.append(total_error)
            if np.all(np.abs(y - self.a2) < 0.1):      # convergência real
                return epoch, history
        return max_epochs, history


# ── PARTE 1 — plano cartesiano ────────────────────────────────────────────────
def plot_xor_space():
    fig, ax = plt.subplots(figsize=(4.5, 4))
    cores = [COLORS[0] if l == 0 else COLORS[1] for l in y.ravel().astype(int)]
    ax.scatter(X[:,0], X[:,1], c=cores, s=140, zorder=3,
               edgecolors="#FFFFFF", linewidths=0.6)
    for (xi, yi_), l in zip(X, y.ravel().astype(int)):
        ax.annotate(f"({int(xi)},{int(yi_)}) → {l}",
                    (xi, yi_), textcoords="offset points", xytext=(8,5),
                    fontsize=8, color="#AAAAAA")
    ax.set_xlim(-0.4, 1.6); ax.set_ylim(-0.4, 1.6)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
    ax.set_title("XOR — espaço de entrada", pad=10)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/xor_espaco.png", dpi=150)
    plt.close(fig)
    print("  ✓ xor_espaco.png")


# ── PARTE 3 — funções de ativação ─────────────────────────────────────────────
def plot_activation_functions():
    x = np.linspace(-6, 6, 400)
    funcs = {
        "Sigmóide binária":   sigmoid_binary(x),
        "Sigmóide bipolar":   sigmoid_bipolar(x),
        "Tangente hiperbólica": tanh(x),
    }
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (nome, fx), cor in zip(axes, funcs.items(), COLORS):
        ax.plot(x, fx, color=cor, linewidth=1.8)
        ax.axhline(0, color="#444", lw=0.6); ax.axvline(0, color="#444", lw=0.6)
        ax.set_title(nome, fontsize=8); ax.grid(True)
    fig.suptitle("Funções de ativação", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/funcoes_ativacao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ funcoes_ativacao.png")


# ── PARTE 6 — experimento principal ──────────────────────────────────────────
def experimento_principal():
    rede = MLPExp(2, 4, 1, learning_rate=0.2, activation="bipolar", seed=42)
    epocas, hist = rede.train(X, y, max_epochs=10_000)

    print(f"\n  Épocas até convergência: {epocas}")
    print("  Saídas após treinamento:")
    saidas = rede.predict(X)
    for xi, si in zip(X, saidas):
        print(f"    {xi.astype(int)} → {si[0]:.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(hist, color=COLORS[0], lw=1.1, alpha=0.9)
    ax.set_xlabel("Época"); ax.set_ylabel("Erro absoluto total")
    ax.set_title(f"Convergência XOR  —  {epocas} épocas  (lr=0.2 | n=4)", pad=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/convergencia_principal.png", dpi=150)
    plt.close(fig)
    print("  ✓ convergencia_principal.png")


# ── EXPERIMENTO A — neurônios ─────────────────────────────────────────────────
def experimento_A():
    configs = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), sharey=False)
    print("\n  Exp A — neurônios na camada intermediária:")
    for ax, n, cor in zip(axes, configs, COLORS):
        rede = MLPExp(2, n, 1, learning_rate=0.5, activation="bipolar", seed=42)
        ep, hist = rede.train(X, y, max_epochs=10_000)
        ax.plot(hist, color=cor, lw=1.1)
        ax.set_title(f"n = {n}\n{ep} épocas", fontsize=8)
        ax.grid(True)
        if ax == axes[0]: ax.set_ylabel("Erro")
        print(f"    n={n}: {ep} épocas  |  erro final={hist[-1]:.4f}")
    fig.suptitle("Exp. A — quantidade de neurônios na camada intermediária  (lr=0.5)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/exp_A_neuronios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ exp_A_neuronios.png")


# ── EXPERIMENTO B — taxa de aprendizagem ──────────────────────────────────────
def experimento_B():
    taxas = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    print("\n  Exp B — taxa de aprendizagem  (n=4):")
    for taxa, cor in zip(taxas, COLORS):
        rede = MLPExp(2, 4, 1, learning_rate=taxa, activation="bipolar", seed=42)
        ep, hist = rede.train(X, y, max_epochs=10_000)
        ax.plot(hist, color=cor, lw=1.1, label=f"lr={taxa}  ({ep} ép.)")
        print(f"    lr={taxa}: {ep} épocas  |  erro={hist[-1]:.4f}")
    ax.set_xlabel("Época"); ax.set_ylabel("Erro absoluto total")
    ax.set_title("Exp. B — taxa de aprendizagem  (n=4, seed=42)", pad=9)
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/exp_B_taxa.png", dpi=150)
    plt.close(fig)
    print("  ✓ exp_B_taxa.png")


# ── EXPERIMENTO C — seeds ─────────────────────────────────────────────────────
def experimento_C():
    seeds = [0, 1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    print("\n  Exp C — inicialização dos pesos  (n=4, lr=0.2):")
    for seed, cor in zip(seeds, COLORS):
        rede = MLPExp(2, 4, 1, learning_rate=0.2, activation="bipolar", seed=seed)
        ep, hist = rede.train(X, y, max_epochs=10_000)
        ax.plot(hist, color=cor, lw=1.0, alpha=0.85, label=f"seed={seed}  ({ep} ép.)")
        print(f"    seed={seed}: {ep} épocas  |  erro={hist[-1]:.4f}")
    ax.set_xlabel("Época"); ax.set_ylabel("Erro absoluto total")
    ax.set_title("Exp. C — inicialização dos pesos  (n=4, lr=0.2)", pad=9)
    ax.legend(fontsize=8); ax.grid(True)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/exp_C_seeds.png", dpi=150)
    plt.close(fig)
    print("  ✓ exp_C_seeds.png")


# ── DESAFIO — AND / OR / NAND / NOR / XOR ────────────────────────────────────
def experimento_outras_funcoes():
    funcoes = {
        "AND":  np.array([[0],[0],[0],[1]], dtype=float),
        "OR":   np.array([[0],[1],[1],[1]], dtype=float),
        "NAND": np.array([[1],[1],[1],[0]], dtype=float),
        "NOR":  np.array([[1],[0],[0],[0]], dtype=float),
        "XOR":  y,
    }
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.2), sharey=False)
    print("\n  Desafio — outras funções lógicas  (n=4, lr=0.5, seed=42):")
    for ax, (nome, yi), cor in zip(axes, funcoes.items(), COLORS):
        rede = MLPExp(2, 4, 1, learning_rate=0.5, activation="bipolar", seed=42)
        ep, hist = rede.train(X, yi, max_epochs=10_000)
        ax.plot(hist, color=cor, lw=1.1)
        ax.set_title(f"{nome}\n{ep} épocas", fontsize=8)
        ax.grid(True)
        if ax == axes[0]: ax.set_ylabel("Erro")
        print(f"    {nome}: {ep} épocas")
    fig.suptitle("Desafio — AND / OR / NAND / NOR / XOR", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/exp_funcoes_logicas.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ exp_funcoes_logicas.png")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n── Gerando gráficos ──────────────────────────────────")
    plot_xor_space()
    plot_activation_functions()

    print("\n── Experimento principal (reprodução do artigo) ──────")
    experimento_principal()

    print("\n── Experimentos do roteiro ───────────────────────────")
    experimento_A()
    experimento_B()
    experimento_C()
    experimento_outras_funcoes()

    print("\n✓ Todos os gráficos salvos em /results\n")
