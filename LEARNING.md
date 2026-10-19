# Learning log

Entries use the global LEARNING.md format (see ~/.claude/CLAUDE.md, Learning protocol).

## 2026-10-19 — Transfer learning and domain shift
- What it is: pretrain a model on a large related dataset, then fine-tune it on your
  small target set. It helps only as far as the source data resembles the target (domain
  shift), and it only applies to *learned* parts.
- Where it shows up here: the student segmenter (P4-T10, ADR 0017). Not the geometry
  modules (`recon/`, `seg/fusion3d.py`, `risk/`), which have no trainable parameters.
- Gap I had: assumed pretraining on existing video would raise accuracy across the whole
  system. Most of VSCS is geometry or already-pretrained detectors, and the evaluation
  must stay on my own footage.
- Go deeper: Yosinski et al., "How transferable are features in deep neural networks?"
  (NeurIPS 2014), https://arxiv.org/abs/1411.1792

## 2026-10-19 — What limits a scan: capture conditions, not camera detail; geometry is per-vehicle
- What it is: 4K iPhone video has far more resolution than a ±2 cm scan or segmentation
  needs. Reconstruction quality is limited by reflective paint and glass, light and blur
  (R-01). Multi-vehicle data can improve a *learned* segmenter's generalisation (domain
  generalisation), but the collision model must be the driven vehicle's own measured
  geometry.
- Where it shows up here: ADR 0017 (hybrid student training); `recon/` and `model/`
  (vehicle-specific geometry); R-19.
- Gap I had: thought the iPhone's detail was the bottleneck, and that other vehicles'
  datasets could stand in for the RAV4 everywhere.
- Go deeper: Hartley & Zisserman, *Multiple View Geometry*, ch. 1 (what a reconstruction
  recovers); Gulrajani & Lopez-Paz, "In Search of Lost Domain Generalization" (2020),
  https://arxiv.org/abs/2007.01434
