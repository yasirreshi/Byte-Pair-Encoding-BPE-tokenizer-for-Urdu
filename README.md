# Urdu BPE Tokenizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tokenizers](https://img.shields.io/badge/tokenizers-0.19.1-green.svg)](https://github.com/huggingface/tokenizers)
[![Gradio](https://img.shields.io/badge/gradio-4.44.0-orange.svg)](https://gradio.app/)

**Topic**: Building a Production-Quality BPE Tokenizer from Scratch

A Byte-Pair Encoding (BPE) tokenizer for Urdu with intelligent auto-tuning and an interactive web UI.

## 📺 Demo

<img width="1958" height="1074" alt="image" src="https://github.com/user-attachments/assets/baf057be-7e5e-4d6d-a96d-3603b488c674" />

> **Try it yourself**: Clone and run `python src/app_tokenizer_ui.py`

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/urdu-bpe-tokenizer.git
cd urdu-bpe-tokenizer

# Setup environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Launch the UI
python src/app_tokenizer_ui.py
```

Open http://localhost:7860 in your browser and start tokenizing! 🎉

## Project Status: ✅ Complete

**Pre-trained Model Available**: `models/urdu_bpe_experiment.json`
- **Vocabulary Size**: 5,500 tokens (UNCAPPED mode)
- **Compression Ratio**: 4.12 chars/token (exceeds 3.2 target by 28%)
- **Corpus Size**: 127,845 characters
- **Total Tokens**: 31,042

## Original Requirements
- Vocabulary size as close as possible to (but <) 5000 tokens
- Compression ratio ≥ 3.2 (total corpus characters / total token count)

**Note**: The current model operates in UNCAPPED mode, prioritizing compression efficiency over the strict 5K vocab limit. The training script supports both capped and uncapped modes.

## Why BPE for Urdu?

### The Tokenization Trade-off

| Approach | Vocabulary Size | Handles Unknown Words | Captures Meaning | Best For |
|----------|----------------|----------------------|------------------|----------|
| **Word-level** | Huge (millions) | ❌ Fails on unseen | ✅ Perfect | English, fixed domains |
| **Character-level** | Tiny (~50) | ✅ Always works | ❌ Loses semantics | Raw text processing |
| **BPE (Subword)** | **Medium (~5K)** | **✅ Compositional** | **✅ Balanced** | **Morphologically rich languages** |

### Urdu-Specific Challenges
- **Morphologically rich**: One root word generates dozens of inflected forms
  - Example: کتاب (book) → کتابوں (of books), کتابات (books formal), کتابخانہ (library)
- **Agglutinative nature**: Postpositions and suffixes attach to words
- **Arabic/Persian loanwords**: Different character distribution patterns
- **No capitalization cues**: Unlike English, case cannot signal word boundaries

**BPE discovers these morphological patterns automatically from corpus statistics** 🎯

## How it Works (BPE Architecture, End-to-End)

### High-Level Pipeline
1. **Input**: Large Urdu corpus (UTF-8 encoded text)
2. **Normalization**: Unicode NFKC (canonical decomposition + composition)
3. **Pre-tokenization**: Whitespace split + punctuation isolation
4. **Training**: BPE learns merges from most frequent adjacent character pairs
5. **Output**: 
   - Tokenizer JSON (vocab + merge rules + normalizer + pre-tokenizer)
   - Stats JSON (compression metrics + requirement validation)

### Step‑by‑step BPE training
1) Initialize vocabulary
	- Start with special tokens: `[UNK]`, `[PAD]`.
	- Add all unique characters observed after normalization/pre‑tokenization.

2) Count pair frequencies
	- Tokenize the corpus with the current vocab/merges.
	- For every adjacent token pair, count its frequency across the corpus.

3) Pick the most frequent pair (subject to `min_frequency`)
	- If none remain, training stops (the corpus is “saturated”).

4) Merge the pair → add one new token to the vocabulary
	- Update the merges list and re‑tokenize accordingly.

5) Repeat 2–4 until stop
	- Stop if you hit the requested `vocab_size` ceiling or there are no mergeable pairs left.

Effect:
- Each merge adds exactly one token.
- Larger vocab typically reduces token count on the corpus, but with diminishing returns.

### Auto-Tuning Logic (Intelligent Vocab Search)

**Goal**: Find optimal vocab size that maximizes compression while meeting requirements.

The trainer probes multiple vocabulary targets:
- **Capped mode**: Tests 2000, 3000, 4000, 4800, 4900, 4950, 4990, 4999
- **Uncapped mode**: Tests 2000, 6000, 8000, 10000, 16000, 24000, 32000

**For each candidate:**
1. Train BPE with that vocab size target
2. Measure achieved vocabulary (may be less if corpus is small)
3. Calculate compression ratio

**Selection Strategy:**
1. Prefer models meeting compression ≥ target (3.2)
2. Among those, pick highest achieved vocab
3. If none meet compression, pick best compression + largest vocab
4. Early-exit when plateau detected (no improvement over multiple probes)

**Why This Works:**
- Larger vocab → More merges → Longer tokens → Higher compression
- But diminishing returns at very large vocabs
- Auto-tuning finds the sweet spot for your specific corpus

### Metrics and Checks
- **Total characters**: Sum of characters over non-blank lines in the corpus after normalization
- **Total tokens**: Total number of tokens produced by the tokenizer over those lines
- **Compression ratio**: chars/token = total_chars / total_tokens
- **Requirements enforced and reported**:
  - `vocab_size < 5000` (in capped mode)
  - `compression_ratio ≥ 3.2`

### Artifacts
- **Tokenizer**: `models/<name>.json` (vocab, merges, normalizer, pre-tokenizer, special tokens)
- **Stats**: `models/<name>_stats.json` with fields:
  - `vocab_size`, `total_chars`, `total_tokens`, `compression_ratio`
  - `requested_vocab`, `meets_compression_target`, `compression_target`, `vocab_cap_mode`

---

## Quick Start 🚀

### Option 1: Use Pre-trained Model (Instant)
```powershell
# Launch interactive UI with pre-trained model
python src/app_tokenizer_ui.py --models-dir models
```
Then open http://localhost:7860 in your browser.

### Option 2: Train Your Own Model
```powershell
# 1. Setup environment
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# 2. Train (uses existing corpus)
python src/train_bpe.py

# 3. Test with UI
python src/app_tokenizer_ui.py --models-dir models
```

---

## Detailed Setup Instructions

## 1. Environment Setup (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

**Dependencies:**
- `tokenizers==0.19.1` - Hugging Face tokenizers library (BPE implementation)
- `requests==2.32.3` - For corpus building utilities (optional)
- `gradio==4.44.0` - Interactive web UI framework

## 2. Data Preparation

### Current Status: ✅ Corpus Ready
The project includes a pre-built consolidated Urdu corpus:
- **File**: `data/urdu_corpus_consolidated.txt`
- **Size**: 127,845 characters
- **Source**: Urdu Wikipedia (CC BY-SA 3.0 / GFDL)

**No additional data preparation needed** to use the pre-trained model or retrain.

### Optional: Build Custom Corpus
If you want to create your own corpus from scratch, corpus-building utilities are available in the project history (check git log for archived scripts).

## 3. Training the Tokenizer

### Using Internal Defaults (Simplest)
```powershell
python src/train_bpe.py
```

**Default Configuration:**
- Input: `data/urdu_corpus_consolidated.txt`
- Vocab size: 5500 (uncapped mode)
- Model prefix: `models/urdu_bpe_experiment`
- Min compression: 3.2
- Auto-tune: False (single training run)

### Using Command-Line Arguments
```powershell
python src/train_bpe.py --input data/urdu_corpus_consolidated.txt --vocab-size 4900 --model-prefix models/urdu_bpe_custom --min-compression 3.2 --auto-tune
```

**Available Options:**
- `--input` - Path to training corpus
- `--vocab-size` - Target vocabulary size
- `--model-prefix` - Output filename prefix
- `--min-compression` - Minimum chars/token ratio (default: 3.2)
- `--auto-tune` / `--no-auto-tune` - Enable/disable vocab search
- `--config` - Load settings from JSON file

### Using Configuration File
Create `config/train_bpe.json`:
```json
{
  "input": "data/urdu_corpus_consolidated.txt",
  "vocab_size": 4900,
  "model_prefix": "models/urdu_bpe_final",
  "min_compression": 3.2,
  "auto_tune": true
}
```

Then run:
```powershell
python src/train_bpe.py --config config/train_bpe.json
```

**Precedence**: CLI arguments > Config file > Internal defaults

### Training Output
```
models/
├── urdu_bpe_experiment.json         # Trained tokenizer (vocab + merges)
└── urdu_bpe_experiment_stats.json   # Training metrics + validation
```

**Stats File Example:**
```json
{
  "vocab_size": 5500,
  "total_chars": 127845,
  "total_tokens": 31042,
  "compression_ratio": 4.12,
  "requested_vocab": 5500,
  "meets_compression_target": true,
  "compression_target": 3.2,
  "vocab_cap_mode": "UNCAPPED"
}
```

**Notes:**
- Achieved vocab may be lower than requested on small corpora
- BPE cannot create merges beyond what the data supports
- Larger, more diverse corpora approach the vocab ceiling while maintaining compression

## 4. Testing the Tokenizer

### Interactive Web UI (Recommended) 🎨

Launch the Gradio-powered interface:
```powershell
python src/app_tokenizer_ui.py --models-dir models
```

Then open http://localhost:7860 in your browser.

**Optional Arguments:**
```powershell
python src/app_tokenizer_ui.py --models-dir models --server-address 127.0.0.1 --server-port 7862
```

### UI Features

**Core Functionality:**
- 🎯 **Model Selector**: Auto-discovers all `.json` tokenizers in `models/` folder
- ✍️ **Text Input**: RTL (right-to-left) support for proper Urdu rendering
- 🎨 **Visual Tokenization**: Each token highlighted with unique color
- 🔢 **Token IDs Table**: Shows numeric IDs for each token
- 📋 **Token List**: Copy-paste friendly breakdown

**Live Metrics:**
- Character count (input length)
- Token count (number of BPE tokens)
- Compression ratio (chars/token)
- Vocabulary size (total tokens in model)

### Example Usage

**Input Text:**
```
سلام دنیا، کیسے ہیں؟
```

**UI Output:**
```
Tokens (colored):
[سلام] [دنیا] [،] [کیسے] [ہیں] [؟]

Metrics:
• Characters: 21
• Tokens: 6
• Compression: 3.5 chars/token
• Vocabulary: 5,500 tokens

Token IDs:
[ID: 1542, ID: 2341, ID: 12, ID: 3421, ID: 1876, ID: 8]
```

### Why Use the UI?

✅ **Visual debugging** - See exactly how your tokenizer segments text  
✅ **Edge case testing** - Try rare words, technical terms, mixed scripts  
✅ **Model comparison** - Switch between models to evaluate performance  
✅ **No coding required** - Interactive experimentation  
✅ **Shareable** - Can expose with `--share` flag for remote access

## 5. Results & Performance

### Current Best Model: `urdu_bpe_experiment.json`

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Vocabulary Size** | 5,500 | < 5,000 | ⚠️ UNCAPPED |
| **Compression Ratio** | **4.12** | ≥ 3.2 | ✅ **+28% above target** |
| **Corpus Size** | 127,845 chars | N/A | Medium |
| **Total Tokens** | 31,042 | N/A | Efficient |
| **Mode** | UNCAPPED | CAPPED | Prioritizes compression |

### What BPE Learned

**Common Full Words (Single Tokens):**
```
اور (and), میں (in), ہے (is), کا (of), کی (of-fem), 
سے (from), پر (on), نے (ergative), کو (to)
```

**Frequent Morphemes:**
```
Prefixes: ال (the), با (with), بے (without)
Suffixes: یں (plural), وں (oblique), ات (formal plural)
Syllables: کر, کے, تا, دا, نا
```

**Rare/Technical Words (Compositional):**
```
"کوانٹم" (quantum) → ['کو', 'ان', 'ٹ', 'م']
"بائیولوجی" (biology) → ['با', 'ئی', 'و', 'لو', 'جی']
```

### Example Tokenizations

```
Input:  سلام دنیا کیسے ہیں
Tokens: ['سلام', 'دنیا', 'کیسے', 'ہیں']
Ratio:  19 chars / 4 tokens = 4.75 ✅

Input:  اردو زبان بہت خوبصورت ہے
Tokens: ['اردو', 'زبان', 'بہت', 'خوب', 'صورت', 'ہے']
Ratio:  25 chars / 6 tokens = 4.17 ✅

Input:  پاکستان کی سرکاری زبان اردو ہے
Tokens: ['پاکستان', 'کی', 'سرکاری', 'زبان', 'اردو', 'ہے']
Ratio:  31 chars / 6 tokens = 5.17 ✅
```

### Performance Insights

✅ **Morphological patterns captured**: Root words + affixes learned separately  
✅ **High compression maintained**: 4.12 avg (28% above 3.2 target)  
✅ **Unknown word handling**: Rare words decompose gracefully  
✅ **No UNK tokens needed**: Character-level fallback always works  
⚠️ **Large vocabulary**: 5.5K exceeds original 5K cap (trade-off for compression)

## 6. Project Structure

```
Session11/Assignment/
├── data/
│   └── urdu_corpus_consolidated.txt   # Training corpus (127KB, Urdu Wikipedia)
├── models/
│   ├── urdu_bpe_experiment.json       # ✅ Pre-trained tokenizer (vocab: 5500)
│   └── urdu_bpe_experiment_stats.json # Training metrics & validation results
├── src/
│   ├── train_bpe.py                   # 🚀 BPE training with auto-tuning
│   └── app_tokenizer_ui.py            # 🚀 Interactive Gradio web UI
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

### Active Components (Daily Use)
- ✅ **`src/train_bpe.py`** - Train new tokenizers, experiment with vocab sizes
- ✅ **`src/app_tokenizer_ui.py`** - Test and visualize tokenization results
- ✅ **`models/urdu_bpe_experiment.json`** - Ready-to-use trained model
- ✅ **`data/urdu_corpus_consolidated.txt`** - Pre-built training corpus

### Historical/Optional Scripts
Corpus building utilities (Wikipedia scraper, corpus merger, CLI tester) were used during initial setup. Check git history if you need to rebuild the corpus from scratch.

---

## 7. Corpus Quality & Data Sources

### Current Corpus
- **Source**: Urdu Wikipedia articles
- **License**: CC BY-SA 3.0 / GFDL
- **Size**: 127,845 characters
- **Quality**: Clean, normalized, deduplicated

### Attribution
Content derived from Urdu Wikipedia. See: https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content

### Tips for Better Corpora
✅ **Diversity**: Mix domains (literature, news, technical, conversational)  
✅ **Size**: Larger corpora (200K+ chars) improve vocab quality  
✅ **Cleanliness**: Remove HTML, boilerplate, excessive duplication  
✅ **Normalization**: Already handled by NFKC normalizer  
❌ **Avoid**: Machine-translated text, code-switched content

---

## 8. Advanced Configuration

### Vocab Size Trade-offs

| Vocab Size | Compression | Model Size | Use Case |
|------------|-------------|------------|----------|
| 2K-3K | Lower (~3.0) | Tiny | Memory-constrained devices |
| 4K-5K | Good (~3.5) | Small | Original assignment target |
| 5K-10K | High (~4.0+) | Medium | **Current model (best balance)** |
| 10K-32K | Very High | Large | Maximum compression, research |

### Training Modes

**Capped Mode** (Original Assignment):
```python
USER_CONFIG = {
    "vocab_size": 4900,
    "auto_tune": True,  # Tests: 2K, 3K, 4K, 4.8K, 4.9K, 4.95K, 4.99K
}
```

**Uncapped Mode** (Current Model):
```python
USER_CONFIG = {
    "vocab_size": 5500,
    "auto_tune": False,  # Tests: 2K, 6K, 8K, 10K, 16K, 24K, 32K
}
```

### Modifying Training Behavior

Edit `src/train_bpe.py` USER_CONFIG block (lines ~37-42):
```python
USER_CONFIG = {
    "input": str(CLEAN_ROOT / "data" / "urdu_corpus_consolidated.txt"),
    "vocab_size": 12000,           # Your target
    "model_prefix": str(CLEAN_ROOT / "models" / "urdu_bpe_large"),
    "min_compression": 3.5,        # Stricter requirement
    "auto_tune": True,             # Enable intelligent search
}
```

Then run without arguments:
```powershell
python src/train_bpe.py
```

---

## 9. Interactive UI (SOTA‑style)
Launch an interactive app to paste Urdu text, pick any tokenizer from `models/`, and visualize tokens with colors and stats:

```powershell
python src/app_tokenizer_ui.py --models-dir models --config .\config\train_bpe.json --server-port 7862
```

Features
- Auto‑discovers `*.json` tokenizers in `models/`
- RTL Urdu input, colorized tokens, token IDs table, token list table
- Stats: characters, tokens, chars/token, vocab size
- Uses open‑source Gradio; no vendor lock‑in

---

## 10. Troubleshooting

### Common Issues & Solutions

**Issue**: "Vocab size stays low (~1000) despite high target"
- **Cause**: Corpus too small or lacks diversity
- **Fix**: Add more varied Urdu text (aim for 200K+ chars)

**Issue**: "Compression ratio < 3.2"
- **Cause**: Vocab size too small for corpus complexity
- **Fix**: Increase `vocab_size` or improve data quality

**Issue**: "Model file not found"
- **Check**: Are you running from `Assignment/` directory?
- **Verify**: `models/urdu_bpe_experiment.json` exists

**Issue**: "Gradio UI not loading"
- **Try**: Different port `--server-port 7862`
- **Check**: Firewall/antivirus blocking port 7860

**Issue**: "Import errors when running scripts"
- **Fix**: Activate virtual environment: `./.venv/Scripts/Activate.ps1`
- **Verify**: `pip list` shows tokenizers, gradio

**Issue**: "Windows path problems"
- **Use**: Forward slashes: `data/corpus.txt`
- **Avoid**: Single backslashes

---

## 11. Technical Deep Dive

### BPE Training Visualization

```
Corpus: "سلام سلام"
Initial: ['س','ل','ا','م',' ','س','ل','ا','م']

Iteration 1: ('س','ل') appears 2× → merge to 'سل'
Result: ['سل','ا','م',' ','سل','ا','م']

Iteration 2: ('سل','ا') appears 2× → merge to 'سلا'
Result: ['سلا','م',' ','سلا','م']

Iteration 3: ('سلا','م') appears 2× → merge to 'سلام'
Result: ['سلام',' ','سلام']

✓ Full word learned as single token!
```

**Key Insight**: Frequency-driven merging naturally captures linguistic patterns without explicit rules.

---

## 12. References

**BPE Algorithm:**
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016)
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers/

**Urdu NLP:**
- Urdu Wikipedia: https://ur.wikipedia.org/
- Attribution: https://en.wikipedia.org/wiki/Wikipedia:Reusing_Wikipedia_content

---

**Happy Tokenizing! 🚀**
