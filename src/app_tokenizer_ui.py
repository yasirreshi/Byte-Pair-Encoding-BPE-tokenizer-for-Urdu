"""SOTA-ish Web UI for Urdu BPE Tokenization (Open-source stack)

Features
- Model picker: auto-discovers tokenizers (*.json) in the models folder
- Urdu text input with RTL support
- Tokenization with colored spans, token IDs, and counts
- Compression metrics: chars, tokens, chars/token
- Dark/light theme toggle, modern layout (Gradio Blocks)

Run
  python src/app_tokenizer_ui.py --models-dir models --server-address 127.0.0.1 --server-port 7860
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
import json
from typing import List, Tuple

import gradio as gr
from tokenizers import Tokenizer


@dataclass
class ModelInfo:
    name: str
    path: str


def discover_models(models_dir: str) -> List[ModelInfo]:
    paths = sorted(glob.glob(os.path.join(models_dir, "*.json")))
    return [ModelInfo(name=os.path.basename(p), path=p) for p in paths]


def load_tokenizer(model_path: str) -> Tokenizer:
    return Tokenizer.from_file(model_path)


def tokenize_text(model_path: str, text: str) -> Tuple[str, List[str], List[int], dict]:
    tok = load_tokenizer(model_path)
    enc = tok.encode(text or "")
    tokens = enc.tokens
    ids = enc.ids
    chars = len(text)
    n_tokens = len(tokens)
    compression = (chars / n_tokens) if n_tokens > 0 else 0.0
    stats = {"chars": chars, "tokens": n_tokens, "chars_per_token": round(compression, 3), "vocab_size": tok.get_vocab_size()}

    # Build colored HTML spans for tokens
    # Soft color wheel
    palette = [
        "#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99",
        "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a",
        "#ffff99", "#b15928",
    ]
    spans = []
    for i, t in enumerate(tokens):
        color = palette[i % len(palette)]
        spans.append(f'<span style="background:{color}; padding:2px 4px; margin:2px; border-radius:6px; display:inline-block;">{t}</span>')
    html_tokens = f'<div dir="rtl" style="font-size:18px; line-height:2;">{" ".join(spans)}</div>'

    return html_tokens, tokens, ids, stats


def build_ui(models_dir: str, config_path: str | None = None):
    cfg = {}
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception as e:
            print(f"Warning: failed to read config {config_path}: {e}")
            cfg = {}

    models = discover_models(models_dir)
    model_choices = [m.name for m in models]
    model_map = {m.name: m.path for m in models}
    # Try to match config model_prefix -> tokenizer file
    default_model = None
    if cfg.get("model_prefix"):
        target = os.path.basename(cfg["model_prefix"] + ".json") if not cfg["model_prefix"].endswith(".json") else os.path.basename(cfg["model_prefix"])
        for m in model_choices:
            if m == os.path.basename(target):
                default_model = m
                break
    if not default_model:
        default_model = model_choices[-1] if model_choices else None

    with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 1100px} .token-box {min-height: 80px}") as demo:
        gr.Markdown("""
        # Tiktokenizer
        Provide Urdu text, pick a tokenizer model, and visualize tokens. Models are discovered from the models folder.
        """)
        if cfg:
            gr.Markdown(
                f"**Config Loaded** corpus: `{cfg.get('input','N/A')}` | requested vocab: `{cfg.get('vocab_size','N/A')}` | min compression: `{cfg.get('min_compression','N/A')}` | auto_tune: `{cfg.get('auto_tune','N/A')}`"
            )

        with gr.Row():
            model_dd = gr.Dropdown(
                label="Select tokenizer model (.json)",
                choices=model_choices,
                value=default_model,
                interactive=True,
            )
            vocab_size = gr.Number(label="Vocab size", interactive=False)

        txt = gr.Textbox(
            label="Urdu input (RTL)",
            placeholder="یہاں اردو متن درج کریں…",
            lines=6,
            elem_id="urdu_input",
        )

        with gr.Row():
            run_btn = gr.Button("Tokenize", variant="primary")
            clear_btn = gr.Button("Clear")

        with gr.Row():
            html_out = gr.HTML(label="Token visualization", elem_classes=["token-box"]) 

        with gr.Row():
            tokens_out = gr.Dataframe(headers=["Tokens"], datatype=["str"], row_count=(1, "dynamic"), col_count=1)
            ids_out = gr.Dataframe(headers=["IDs"], datatype=["number"], row_count=(1, "dynamic"), col_count=1)

        with gr.Row():
            chars_box = gr.Number(label="Characters", interactive=False)
            tok_count_box = gr.Number(label="Tokens", interactive=False)
            cpt_box = gr.Number(label="Chars / Token", interactive=False)

        def on_model_change(name: str):
            path = model_map.get(name)
            if not path:
                return gr.update(), 0
            tok = load_tokenizer(path)
            return gr.update(), tok.get_vocab_size()

        def on_tokenize(name: str, text: str):
            path = model_map.get(name)
            if not path:
                return "<i>No model selected</i>", [], [], 0, 0, 0
            html, tokens, ids, stats = tokenize_text(path, text)
            tokens_df = [[t] for t in tokens]
            ids_df = [[i] for i in ids]
            return html, tokens_df, ids_df, stats["chars"], stats["tokens"], stats["chars_per_token"], stats["vocab_size"]

        model_dd.change(fn=on_model_change, inputs=[model_dd], outputs=[html_out, vocab_size])
        run_btn.click(
            fn=on_tokenize,
            inputs=[model_dd, txt],
            outputs=[html_out, tokens_out, ids_out, chars_box, tok_count_box, cpt_box, vocab_size],
        )
        clear_btn.click(lambda: ("", "", "", 0, 0, 0, 0), None, [html_out, tokens_out, ids_out, chars_box, tok_count_box, cpt_box, vocab_size])

    return demo


def parse_args():
    ap = argparse.ArgumentParser(description="Urdu BPE Tokenizer UI")
    # Always use ../models relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_models_dir = os.path.abspath(os.path.join(script_dir, "..", "models"))
    ap.add_argument("--models-dir", default=default_models_dir, help="Directory to scan for *.json tokenizers (default: ../models relative to script)")
    ap.add_argument("--config", help="Optional JSON config for defaults used in training")
    ap.add_argument("--server-address", default="127.0.0.1")
    ap.add_argument("--server-port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return ap.parse_args()


def main():
    args = parse_args()
    demo = build_ui(args.models_dir, config_path=args.config)
    demo.queue()
    demo.launch(server_name=args.server_address, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
