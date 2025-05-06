# CodeMixBench: Evaluating Large Language Models on Code Generation with Code-Mixed Prompts

This repository is the official implementation of [CodeMixBench: Evaluating Large Language Models
on Code Generation with Code-Mixed Prompts]().

Built on top of [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench), it introduces 6,840 prompts in three multilingual combinations:
- **Hinglish** (Hindi-English)
- **Spanglish** (Spanish-English)
- **Pinyin-English** (Chinese-English)

Each task is augmented at two **Code-Mixing Degree (CMD)** levels: `CMD = 0.6` (light mixing) and `CMD = 0.9` (heavy mixing). The benchmark is designed to reflect real-world multilingual developer interactions and assesses model robustness beyond English-only settings.

CodeMixBench is a multilingual benchmark for evaluating large language models on code generation tasks using code-mixed prompts. Built on top of BigCodeBench, it includes 6,840 prompts across Hinglish, Spanish-English, and Chinese Pinyin-English with controlled code-mixing levels (CMD = 0.6, 0.9).
>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## 📦 Requirements

We strongly recommend using an isolated environment to avoid conflicts with dependencies.

```bash
pip install -I -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
pip install bigcodebench --upgrade
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.



>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).


## 📈 Evaluation

To evaluate models on CodeMixBench using the BigCodeBench evaluation pipeline, follow the steps below:

### 1. Set Dataset Path

Override the default BigCodeBench dataset path using an environment variable:

```python
import os
os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = "dataset/hin-eng-cmd0.6-instruct.jsonl"  # example
```

Available variants include:
- `hin-eng-cmd0.6-instruct.jsonl`
- `hin-eng-cmd0.9-instruct.jsonl`
- `spanglish-cmd0.6-complete.jsonl` (and others)

### 2. Generate Model Predictions

Use the `generate` command to produce code completions for the selected dataset split:

```bash
bigcodebench.generate \
    --model <model_name_or_path> \
    --split complete \
    --subset full \
    --max_new_tokens=3700
```

> Replace `<model_name_or_path>` with any Hugging Face model path (e.g., `microsoft/Phi-4-multimodal-instruct`).

### 3. Run Evaluation

After generation, evaluate the model’s outputs using the `evaluate` command:

```bash
bigcodebench.evaluate \
    --execution local \
    --split complete \
    --subset full \
    --samples <path_to_generated_predictions.jsonl> \
    --save_pass_rate
```

The evaluation will generate:
- `*_eval_results.json` (functional execution outcomes)
- `*_pass_at_k.json` (pass@1 metric)
- Optionally, `*_sanitized_calibrated_pass_at_k.json` (post-processed scores)

### 📁 Output Directory

Results are stored in:
```
codemixbench/hindi-eng/bcb_results_cmd<cmd_value>/<model_id>/results/
```

Each folder contains calibrated JSON results that can be aggregated and plotted using `code-eval.ipynb`.


## ✅ Pre-evaluated Model Results

Previously generated and evaluated results can be found under evals folder, only for hindi-english code-mix combination results are generated and evaluated under both cmd 0.6 and 0.9 levels.


## 📊 Results

### Key Metrics
- **pass@1**: Execution-based metric evaluating whether a generated code sample passes the unit test on the first try.
- **CMD 0.6 / 0.9**: Refers to the level of code-mixing applied to prompts — 0.6 being light and 0.9 being heavy.
- **Original (English)**: Scores on standard BigCodeBench prompts without code-mixing.


| Model                             |   Original (English) %|   CMD 0.6 % |CMD 0.9 %|
|:------------------                |---------------------:|----------:|----------:|
| DeepSeek-R1-Distill-Qwen-1.5B     |                 7.9 |        4.1 |       4.8 |
| StarCoder2-3B                     |                 21.4 |        9.3|       9.1 |
| Llama-3.2-1B                      |                 11.3 |         5 |       4.5 |
| Qwen2.5-Coder-1.5B-Instruct       |                 32.7 |      31.8 |      22.2 |
| Gemma-3-4b-it                     |                    ? |         ? |      28.4 |
| OpenCoder-8B-Instruct             |                 50.9 |      51.3 |         ? |
| Phi-4-multimodal-instruct         |                    ? |      46.4 |      33.1 |
| DeepSeek-R1-Distill-Llama-8B      |                 15.3 |      15.9 |      11.1 |
| Hermes-2-Theta-Llama-3-8B         |                    ? |      36.4 |         ? |
| CodeLlama-7b-Instruct-hf          |                    ? |         ? |      17.7 |
| Qwen2.5-Coder-7B-Instruct         |                 48.8 |      41.2 |      41.3 |
| Llama-3.1-8B-Instruct             |                 40.5 |      38.8 |      31.4 |
|StarCoder2-7b| 27.7| ?| 8.9|
|Phi-4| 55.4| 46.7| 47.1|
|DeepSeek-R1-Distill-Qwen-14B| 48.4| 38.8| 40.3|
|DeepSeek-Coder-V2-Lite-Instruct| 47.6| 37.7| 38.1|
|StarCoder2-15b-instruct-v0.1| 45.1| 33.1| 32.7|

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 