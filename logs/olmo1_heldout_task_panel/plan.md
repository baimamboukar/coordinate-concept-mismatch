# OLMo 1 Held-Out Task Panel

[Hugging Face](https://huggingface.co/buckets/baimamboukar/coordinate-concept-mismatch/tree/studies) | [Weights & Biases](https://wandb.ai/JinesisLab/coordinate-concept-mismatch)

## Objective

Test whether the SST-2/WildGuardMix activation map generalizes to concepts absent from map fitting. The preceding pooled-map experiment showed that one affine map can serve both included tasks; it did not establish zero-shot map transport. Two held-out tasks are fixed before activation extraction, and neither will be replaced after observing results.

## Tasks and fixed protocol

- **AG News topic:** pinned `fancyzhx/ag_news`; World versus Business, with the official 3,800-row binary test subset protected. Per seed, 12,000 balanced training and 2,000 validation rows are drawn from the training split.
- **MNLI relation:** pinned `nyu-mll/glue` (`mnli`); entailment versus contradiction, formatted as `Premise: ...\nHypothesis: ...`, with all 6,692 eligible matched-validation examples protected for testing. Training and validation sizes match AG News.
- **Models:** pinned Ai2 and AMD OLMo 1B checkpoints, their shared tokenizer, seeds 42 and 137, both directions, 75% residual depth, and last-non-padding activations.
- **Probes:** linear is primary; degree-2 CP and MLP probes are secondary. AUROC is primary, while the complete classification, calibration, low-FPR, threshold, confusion, and row-level contract is retained.

Each task first qualifies independently. All 4/4 linear frozen transfers must satisfy the existing failure rule: both within-model AUROCs at least 0.75, gap at least 0.10, and paired-bootstrap lower bound above zero. Its same-task affine map must then recover at least 75% of the median gap, with substantial recovery in at least 3/4 comparisons. A failed task is reported as non-qualifying; its held-out maps are not interpreted.

## Held-out map test

For every qualifying task $T$, maps are fitted without its activations or labels under four fixed conditions: SST-2 only (12,000 rows), WildGuardMix only (12,000), an equal-budget pool (6,000 plus 6,000), and the full pool (12,000 plus 12,000). For fit distribution $F$,

$$
R_F^T=\frac{\operatorname{AUROC}_T(A_F)-\operatorname{AUROC}_T(\mathrm{frozen})}{\operatorname{AUROC}_T(\mathrm{oracle})-\operatorname{AUROC}_T(\mathrm{frozen})}.
$$

The primary pooled criterion requires median recovery at least 50%, median same-task improvement retention at least 75%, substantial recovery in at least 3/4 comparisons, and 0/4 substantial shuffled controls. Strong generalization requires at least 75% recovery with 4/4 substantial comparisons. A coverage advantage requires the equal-budget pooled map to exceed both single-task maps by at least ten median-recovery points and in at least 3/4 paired comparisons. A ten-point full-versus-equal gain indicates material sample-budget sensitivity.

Broad zero-shot support requires both tasks to qualify and the full pooled map to pass on both. One qualifying success yields bounded support only. All bulky data and results move directly between workers and Hugging Face; W&B tracks probe training.
