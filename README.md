# Break-Fix First

Code and archived results for productive failure in cloud debugging.

Use Python 3.11 or 3.12.

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests
python replay_llm.py --out replay-check --check
python make_figures.py
```

Replay checks the archived LLM summaries. Figures 2–6 are written to `figs/`. For Figure 1, use pdfLaTeX with standalone, TikZ and Libertine:

```sh
mkdir -p figs
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figs figure1.tex
```

To regenerate simulation results, run `python run_analyses.py`, `python robustness.py`, then `python trial_sim.py` in a working copy. Each writes to `results/`. A fresh LLM benchmark uses `llm_bench.py` with a `host.llm` adapter.

[Code licence](LICENSE) · [Data release terms](LICENSE-DATA)
