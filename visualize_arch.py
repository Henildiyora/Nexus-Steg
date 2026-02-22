"""
Generate computational graph visualizations of the Nexus-Steg architecture.

Usage:
    pip install torchviz graphviz
    python visualize_arch.py

On macOS:  brew install graphviz
On Ubuntu: sudo apt-get install graphviz
On Colab:  !apt-get install graphviz && pip install torchviz graphviz
"""

import torch
from torchviz import make_dot
from src.models.hybrid_transformer import HidingNetwork, RevealNetwork
from src.models.discriminator import SteganalysisDiscriminator
import os

os.makedirs("results", exist_ok=True)

cover = torch.randn(1, 3, 256, 256)
secret = torch.randn(1, 3, 256, 256)

print("Generating HidingNetwork graph...")
hiding = HidingNetwork()
stego = hiding(cover, secret)
dot = make_dot(
    stego,
    params=dict(hiding.named_parameters()),
    show_attrs=False,
    show_saved=False,
)
dot.attr(rankdir="TB")
dot.render("results/hiding_network_graph", format="png", cleanup=True)
print(f"  Saved: results/hiding_network_graph.png")

print("Generating RevealNetwork graph...")
reveal = RevealNetwork()
revealed = reveal(stego.detach())
dot = make_dot(
    revealed,
    params=dict(reveal.named_parameters()),
    show_attrs=False,
    show_saved=False,
)
dot.attr(rankdir="TB")
dot.render("results/reveal_network_graph", format="png", cleanup=True)
print(f"  Saved: results/reveal_network_graph.png")

print("Generating Discriminator graph...")
disc = SteganalysisDiscriminator()
d_out = disc(stego.detach())
dot = make_dot(
    d_out,
    params=dict(disc.named_parameters()),
    show_attrs=False,
    show_saved=False,
)
dot.attr(rankdir="TB")
dot.render("results/discriminator_graph", format="png", cleanup=True)
print(f"  Saved: results/discriminator_graph.png")

# Print model summaries
print("\n--- Model Parameter Counts ---")
for name, model in [("HidingNetwork", hiding), ("RevealNetwork", reveal), ("Discriminator", disc)]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {name}: {total:,} total  ({trainable:,} trainable)")
