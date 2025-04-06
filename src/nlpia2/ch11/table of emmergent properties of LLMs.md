# emmergent properties of LLMs

#### LLM benchmarks

- ["Measuring Massive Multitask Lanugage Understanding" by Dan Hendrycks et al.](https://arxiv.org/pdf/2009.03300.pdf)

```asciidoc
- footnote:["Measuring Massive Multitask Lanugage Understanding" by Dan Hendrycks et al. (https://arxiv.org/pdf/2009.03300.pdf)]
- footnote:["Emergent Abilities of Large Language Models" by Jason Wei et al (https://arxiv.org/abs/2206.07682)]
```

FIXME: use pydoxtools to extract this table?

```csv
flops, params, model, prompt type, problem type
2e22, 13B, GPT-3, few-shot, 3-digit addition and subtraction
2e22, 13B, GPT-3, few-shot, 4-digit addition and subtraction 
3e23, 175B, GPT-3, few-shot, Massive Multitask Language Understanding (MMLU) e.g. math, history, law, CS 
1e22, Toxicity classification (CivilComments benchmark)
```

```text
Few-shot, Addition/subtraction (3 digit), 2.3E+22, 13B, GPT-3, Brown et al., (2020)
Few-shot, Addition/subtraction (4-5 digit), 3.1E+23, 175B, , 
Few-shot, MMLU Benchmark (57 topic avg.), 3.1E+23, 175B, GPT-3, Hendrycks et al., (2021a)
Few-shot, Toxicity classification (CivilComments), 1.3E+22, 7.1B, Gopher, Rae et al., (2021)
Few-shot, Truthfulness (Truthful QA), 5.0E+23, 280B, , 
Few-shot, MMLU Benchmark (26 topics), 5.0E+23, 280B, , 
Few-shot, Grounded conceptual mappings, 3.1E+23, 175B, GPT-3, Patel & Pavlick, (2022)
Few-shot, MMLU Benchmark (30 topics), 5.0E+23, 70B, Chinchilla, Hoffmann et al., (2022)
Few-shot, Word in Context (WiC) benchmark, 2.5E+24, 540B, PaLM, Chowdhery et al., (2022)
Few-shot, Many BIG-Bench tasks (see Appendix E), , ,Many Many Many BIG-Bench, (2022)
Augmented, Instruction following (finetuning), 1.3E+23, 68B, FLAN, Wei et al., (2022a)
Augmented, Scratchpad: 8-digit addition (finetuning), 8.9E+19, 40M, LaMDA, Nye et al., (2021)
Augmented, Using open-book knowledge for fact checking, 1.3E+22, 7.1B, Gopher, Rae et al., (2021)
Augmented, Chain-of-thought: Math word problems, 1.3E+23, 68B, LaMDA, Wei et al., (2022b)
Augmented, Chain-of-thought: StrategyQA, 2.9E+23, 62B, PaLM, Chowdhery et al., (2022)
Augmented, Differentiable search index, 3.3E+22, 11B, T5, Tay et al., (2022b)
Augmented, Self-consistency decoding, 1.3E+23, 68B, LaMDA, Wang et al., (2022b)
Augmented, Leveraging explanations in prompting, 5.0E+23, 280B Gopher, Lampinen et al., (2022)
Augmented, Least-to-most prompting, 3.1E+23, 175B, GPT-3, Zhou et al., (2022)
Augmented, Zero-shot chain-of-thought reasoning, 3.1E+23, 175B, GPT-3, Kojima et al., (2022)
Augmented, Calibration via P(True), 2.6E+23, 52B, Anthropic, Kadavath et al., (2022)
Augmented, Multilingual chain-of-thought reasoning, 2.9E+23, 62B, PaLM, Shi et al., (2022)
Augmented, Ask me anything prompting, 1.4E+22, 6B, EleutherAI, Arora et al., (2022)
```