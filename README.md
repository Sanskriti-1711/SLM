# Banking SLM Notebook Project

This repository contains a notebook-first implementation of a small language model (SLM) workflow focused on banking/compliance-style use cases, with long-term memory and controlled memory writes.

Main notebook: `Small_Language_Model_(1).ipynb`

## Problem Framing

The target was not only to train a compact model, but to make it usable in a practical assistant setting where:

- responses should be grounded in known memory,
- high-risk memory writes should be controlled,
- the pipeline should run on limited hardware (low VRAM GPU),
- intermediate artifacts should be auditable.

## Thought Process and Approach

The implementation was designed as a layered system instead of a plain LM training script.

- Layer 1: build a compact generative model that can learn domain patterns.
- Layer 2: add a retrieval memory layer for factual grounding.
- Layer 3: gate critical memory writes through human approval.
- Layer 4: keep logging and persistence so behavior is inspectable and recoverable.

This separation keeps the model simple while handling safety and factuality outside the model weights.

## Core Components

- `CriticalityScorer`
- `ApprovalGate`
- `AuditLog`
- `MemoryBank` (partitioned FAISS)
- `generate(...)` with optional memory grounding

## Data and Tokenization Strategy

The notebook combines multiple sources and formats them into a chat-style structure:

- user turn
- assistant turn
- end token marker

Tokenizer was moved to a SentencePiece-style setup with a smaller practical vocabulary target for SLM training efficiency.

## Training Strategy (Resource-Constrained)

The training loop is step-based and tuned for low VRAM.

- micro-batching (`gpu_micro_batch_size`)
- mixed precision autocast on CUDA
- gradient scaling
- gradient clipping
- periodic validation
- checkpointing
- patience-based early stopping on validation loss

### Steps vs Epochs

- `max_steps` means optimizer update count.
- epoch means one full pass through the training loader.
- approximate epochs covered = `max_steps / len(train_loader)`.

## Memory Management in Training and Runtime

### GPU/Runtime Memory Management

- controlled micro-batch execution to avoid OOM
- OOM handler that clears CUDA cache and continues
- periodic checkpointing to preserve progress

### Artifact Management

Large training artifacts are intentionally excluded from Git tracking via `.gitignore`.

## Partitioned FAISS Memory and Retrieval

Long-term memory is split into semantic partitions:

- Partition `A`: critical policy memory
- Partition `B`: other critical memory (fraud/legal/transaction)
- Partition `C`: medium-importance operational memory

### Why partitioning

- separates high-risk and lower-risk memory types
- allows partition-aware retrieval
- keeps downstream controls explicit

### Retrieval flow

- embed query
- retrieve from `A`, `B`, and `C`
- merge and similarity-rank
- deduplicate snippets
- inject selected memory context into the model prompt

## Approval Gate for Critical Writes

Critical writes are not auto-added blindly.

- input is scored for criticality/subtype
- critical entries trigger approval path
- decision is logged
- only approved critical content is written to memory

This keeps memory evolution human-supervised for sensitive categories.

## Auditability

Every approval action can be recorded with:

- timestamp
- text preview
- level and subtype
- partition
- mode (live or batch)
- decision

This supports traceability and review.

## End-to-End Inference Flow

- user prompt received
- optional memory retrieval performed
- retrieved context prefixed as system guidance
- model generates response with top-k + temperature sampling
- output can be inspected with or without memory grounding

## Key Functionalities Implemented

- compact SLM training loop
- low-VRAM stabilization controls
- checkpoint save/load
- perplexity-style validation pass
- criticality scoring and routing
- approval-gated critical memory writes
- partitioned FAISS storage and retrieval
- memory-grounded generation
- interactive chat test cell in notebook

## Repository Notes

- this repo is intentionally lightweight for GitHub publishing
- large binaries (`.pt`, `.bin`, FAISS index files) are excluded
- generated artifacts should be rebuilt locally from notebook cells

## Suggested Next Improvements

1. Add a scripted runner (non-notebook) for reproducibility.

2. Add an automated banking QA evaluation set.
Create `data/eval/banking_qa.jsonl` with `question`, `expected_answer`, and `topic` fields, then compute exact match/F1 on a fixed eval script so model quality is trackable across runs.

3. Separate demo config from heavy training config.
Keep two explicit profiles:
- `demo`: low steps, low memory footprint, fast turnaround,
- `train`: longer runs, larger token budget, stricter evaluation/checkpointing.

## Current Model Status and Limitation

Current model quality is still below target for coherent, domain-reliable answers.
The main reason is insufficient training budget relative to model/data complexity:

- too few effective training updates/tokens seen,
- limited GPU memory forcing small micro-batches and conservative settings,
- restricted compute time.

In short: the approach and architecture are in place, but the model needs more training time and stronger compute to converge to stable banking-grade performance.
