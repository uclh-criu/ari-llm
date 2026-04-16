# ari-llm

Simple local scripts to:
- read prompt `.txt` files from `prompts/`
- run them with either Hugging Face or Ollama
- append results to a CSV
- print saved results in the terminal

## Files

- `script.py`: main runner (generation + CSV logging)
- `results.py`: prints `prompt_file` and `output` from CSV
- `requirements.txt`: Python dependencies
- `prompts/`: input prompt text files

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit the variables at the top of `script.py`:

- `PROVIDER`: `"hf"` or `"ollama"`
- `MODEL_ID`: model name for the selected provider
- `USE_STRUCTURED_OUTPUT`: `True`/`False` (used by Ollama path)
- `PROMPTS_DIR`: input directory (default `prompts`)
- `OUTPUT_CSV`: output file (default `ari_llm_outputs.csv`)
- `MAX_NEW_TOKENS`: generation budget
- `TEMPERATURE`: sampling temperature (`0.0` is forced automatically for structured Ollama)
- `OLLAMA_URL`: Ollama server URL

## Run generation

```bash
python script.py
```

## View results

```bash
python results.py
```