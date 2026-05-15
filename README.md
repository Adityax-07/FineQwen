title: FinQwen — Financial LLM Benchmark
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Qwen2.5-7B LoRA fine-tuned & benchmarked on finance NLP
FinQwen — Financial LLM Benchmark Dashboard
Qwen2.5-7B fine-tuned with LoRA (Unsloth) on SEC 10-K filings and earnings call sentiment data. Evaluated on a 50-question benchmark across 4 financial NLP task types.

Results Summary
Metric	FinQwen (ft)	Base LLM (8B)	Strong LLM (70B)
Sentiment F1 ↑	0.767	0.324	0.398
ROUGE-L ↑	0.000	0.340	0.318
QA Accuracy ↑	0.867	1.000	1.000
Groundedness ↑	0.387	0.687	0.833
Hallucination Rate ↓	66.7%	20.0%	6.7%
Training Setup
Base: unsloth/Qwen2.5-7B-Instruct-bnb-4bit
Method: LoRA (r=16, RSLoRA, α=16)
Steps: 300 | LR: 2e-4 cosine | Batch: 8
Data: AdaptLLM/finance-tasks + FinGPT/fingpt-sentiment-train (~8k examples)
Hardware: T4 GPU (Google Colab)
Key Finding
Domain fine-tuning significantly improves sentiment classification (+137% F1) but hurts hallucination calibration at 300 steps — a known LoRA tradeoff
