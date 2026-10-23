# The Informational Cost of Spatial Prompts in SAM: A Controlled Study on Polyp Segmentation

**Document status:** complete scientific draft for internal review, 18 October 2026; revised 22 October 2026 (label-efficiency budget, negative-control post-processing, references). It is not a submitted manuscript. Author names, order, affiliations, and corresponding author remain pending formal confirmation.

## Extended abstract

Segment Anything (SAM) is often described as a general segmentation model, yet its mask decoder is conditioned on spatial prompts that communicate where the target lies and, indirectly, which object should be segmented. This work studies the empirical informational cost of removing or degrading those prompts in polyp segmentation. We separate three questions that are frequently conflated: whether useful mask candidates exist, whether SAM selects the right candidate, and whether an automatic localization system can replace a human prompt at acceptable quality and computational cost.

Using fixed Kvasir-SEG development images, frozen SAM ViT-B nearly collapses with an empty spatial input (Dice 0.0089), whereas oracle points and boxes reach 0.6831 and 0.9352. Static learned tokens recover 0.3633 and image-conditioned tokens 0.5013, but a context-permutation control shows that only a small part of the conditioned gain depends on correctly paired image context. A linear probe over frozen SAM features reaches 0.7489; converting its largest component into a box and interior point for SAM raises Dice to 0.8179. A train-derived fixed spatial prior reaches only 0.3582, showing that image-specific localization, rather than the mere presence of a prompt, explains most of the recovery.

The best automatic SAM pipeline does not dominate a mature direct segmenter on the same development set: U-Net at 256 pixels reaches 0.8267 and the paired SAM−U-Net difference is −0.0088 (95% CI −0.0472 to 0.0281). SAM is also roughly 309 times slower in the measured CPU prototype (8.37 s versus 27.08 ms). Its value appears in regimes where pretraining matters. Under fixed update counts, automatic SAM exceeds U-Net trained from scratch by 0.2718, 0.1863, and 0.1226 Dice with 25, 100, and 400 masks; the 400-mask gap reflects that short budget, because the longer U-Net recipe reaches 0.8267 with the same masks. Under frozen transfer to CVC-ClinicDB, SAM reaches 0.4587 versus 0.3405 for U-Net, a sequence-clustered difference of 0.1182 (0.0672 to 0.1684), although both scores are too low to support clinical adequacy.

The external negative control changes the interpretation. On all 1,009 HyperKvasir cecum-class images, automatic SAM marks a mean 5.21% of image area as polyp, compared with 17.33% for U-Net; the paired difference is −12.12 percentage points (−12.67 to −11.59). Nevertheless, SAM produces a non-empty mask in every image and U-Net in 99.83%. Thus, the SAM pipeline marks less false area on these out-of-class frames, but neither system reliably represents target absence: the probe fires on every image, so the always-prompt pipeline always emits a mask. The most defensible contribution is a controlled diagnosis of prompt dependence, candidate availability, selection, label efficiency, transfer, specificity proxy, and end-to-end cost—not a new prompting architecture or a state-of-the-art claim. The results motivate an explicit presence or abstention gate before automatic prompt generation.

![Experimental summary. Error bars are 95% bootstrap intervals. In the label-efficiency panel, U-Net uses the fixed 1,500-update budget and the open marker is the longer recipe with the same 400 masks. HyperKvasir annotations report the percentage of images with at least 1% false area.](results/paper_20261018/figure_experimental_summary.png)

## 1. Research question and scope

The central question is:

> How much of SAM's performance in polyp segmentation depends on spatial information supplied by a prompt, where does performance fail when this information is absent, and how much can an automatic replacement recover?

The phrase *prompt cost* is used in two explicit senses:

1. **Empirical informational cost:** the change in segmentation quality when spatial prompts are removed, perturbed, reduced, replaced with a fixed prior, or predicted from the image.
2. **Operational system cost:** trainable and total parameters, full RGB-to-mask latency, training budget, and number of local masks required.

The project does **not** estimate Shannon bits, human annotation time, energy consumption, or clinical workflow savings. Simulated clicks and boxes are not treated as measured human effort.

## 2. Data and evaluation partitions

### 2.1 Kvasir-SEG

Kvasir-SEG contains 1,000 colonoscopy images with pixel-level polyp masks. The repository preserves the official 120-image MediaEval validation set as the historical E1 test partition and divides the remaining 880 images into 780 training and 100 development images. Phase B uses a fixed 400-image training subset and the same 100-image development set.

- Phase B model development: `train400` and `dev100`.
- Three training seeds: 0, 1, and 2 where applicable.
- The historical 120-image E1 test results are retained for provenance but were not re-evaluated after Phase B decisions.
- Model selection, thresholds, learning rates, and epoch choices use development data; Phase B development intervals are exploratory rather than held-out generalization estimates.

### 2.2 CVC-ClinicDB

Frozen models and thresholds selected on Kvasir-SEG are evaluated without adaptation on all 612 frames from 29 sequences. Confidence intervals resample complete sequences to avoid treating adjacent frames as independent. All frames contain annotated polyps, so this dataset measures positive-domain transfer but cannot estimate false activation on negatives.

### 2.3 HyperKvasir negative proxy

The negative control uses all 1,009 images in the official HyperKvasir `lower-gi-tract/anatomical-landmarks/cecum` folder. Every file is downloaded from OSF and verified against its published SHA-256. The audit finds 1,009 decodable files and 1,009 unique pixel arrays.

The class is a negative proxy at image level: it is an anatomical-landmark label rather than a pixel-by-pixel confirmation that no polyp-like region exists. Public patient and video identifiers are unavailable for these labeled still images. Consequently, bootstrap intervals resample images and must not be interpreted as patient-level clinical specificity. HyperKvasir and Kvasir-SEG were both collected at Vestre Viken Health Trust (Norway), so this control tests out-of-class frames rather than a new acquisition domain, and patient overlap with the Kvasir-SEG training images cannot be excluded.

## 3. Model conditions

### 3.1 Frozen SAM prompt conditions

SAM ViT-B remains frozen and receives images at a longest side of 1,024 pixels. The controlled conditions are:

- empty spatial input and a fixed output token;
- oracle interior point per ground-truth component;
- oracle bounding box per component;
- controlled displacement or enlargement of oracle prompts;
- four learned static prompt tokens;
- four learned image-conditioned tokens;
- a fixed point/box prior estimated only from normalized training masks;
- a 257-parameter linear probe on frozen SAM image embeddings;
- probe-derived box or box-plus-point followed by the frozen SAM prompt encoder and mask decoder.

Oracle prompts may reveal component count and are quality ceilings, not deployable automatic systems.

### 3.2 Candidate availability and selection

SAM can emit several candidate masks for an ambiguous prompt. The analysis separates:

- the mask selected by SAM's predicted-IoU score;
- a fixed output token;
- the best available candidate selected using ground truth as a diagnostic oracle.

With empty spatial input, even oracle selection of the best candidate remains poor. With a correct point, strong candidates exist but predicted-IoU selection leaves a large gap. Candidate generation and candidate selection therefore fail in different regimes.

### 3.3 Direct RGB comparators

FCN-Lite, depthwise-separable U-Net, and compact U-Net are local declared variants trained without pretraining on the same training IDs. The mature U-Net continuation starts from identical epoch-30 weights, Adam states, and RNG states and receives 15 additional fixed epochs at either 128 or 256 pixels. These are practical system controls, not exact reproductions of the original architecture papers or specialist polyp state-of-the-art systems.

## 4. Statistical design

- Dice and IoU are calculated after restoration to original image resolution.
- Development summaries average the three training seeds per image before 10,000 image bootstrap resamples.
- Seed means and standard deviations are retained separately.
- CVC-ClinicDB intervals resample its 29 complete sequences.
- HyperKvasir negative metrics are false area fraction, largest-component fraction, non-empty rate, and rates above 0.5%, 1%, and 5% image area.
- Paired differences are computed per image or per sequence before resampling.
- An interval containing zero is reported as unresolved; it is not evidence of equivalence.

## 5. Results

### 5.1 Prompt dependence and automatic recovery

| Condition | Development Dice | 95% CI |
|---|---:|---:|
| Empty spatial input, single output | 0.0089 | [0.0007, 0.0227] |
| Oracle GT point | 0.6831 | [0.6155, 0.7463] |
| Oracle GT box | 0.9352 | [0.9279, 0.9420] |
| Static learned tokens | 0.3633 | [0.2913, 0.4375] |
| Image-conditioned tokens | 0.5013 | [0.4260, 0.5762] |
| Linear probe | 0.7489 | [0.7067, 0.7879] |
| Probe → box → SAM | 0.7738 | [0.7271, 0.8170] |
| Probe → box+point → SAM | **0.8179** | **[0.7707, 0.8600]** |

The box-plus-point decoder stage improves over the probe by 0.0690 Dice [0.0471, 0.0919]. The fixed train-derived box-plus-point prior reaches only 0.3582, so the gain cannot be explained by a generic central-location prior. Static tokens and the fixed point prior have nearly identical mean Dice, but their paired interval does not establish equivalence.

### 5.2 What fails without a prompt

The selection diagnostic produces two distinct findings:

- With an empty prompt, realizable candidate selection reaches approximately 0.1504 Dice and a ground-truth best-candidate oracle only 0.1824. Better ranking alone cannot recover a good mask if useful candidates are absent.
- With an oracle point, predicted-IoU selection reaches 0.5401 while the best of three candidates reaches 0.8600. Here, useful candidates exist and selection is a major bottleneck.

Prompt perturbations further show that location precision matters. A point shifted by two lesion radii falls to 0.1529 Dice, while a box enlarged to twice its size reaches 0.3216. These are controlled sensitivity results, not human-time measurements.

### 5.3 Direct-model comparison and resolution

| Automatic system | Development Dice | Trainable parameters | Median CPU latency |
|---|---:|---:|---:|
| FCN-Lite, RGB 128 | 0.6002 | 18,355 | 6.86 ms |
| DS-U-Net, RGB 128 | 0.6548 | 18,665 | 14.08 ms |
| Compact U-Net, RGB 128 | 0.7814 | 487,289 | 12.11 ms |
| Mature U-Net endpoint, RGB 256 | **0.8267** | 487,289 | **27.08 ms** |
| Probe → box+point → SAM | 0.8179 | 257 plus frozen SAM | **8.37 s** |

The mature U-Net trains 30 epochs at 128 pixels and 15 more at 256 pixels (2,250 updates). SAM minus mature U-Net is −0.0088 [−0.0472, 0.0281]. This interval demonstrates neither superiority nor equivalence. Input resolution, pretraining, and training recipes differ, so this is a practical system comparison rather than an isolated causal architecture comparison. The measured SAM pipeline has 93,735,729 total unique parameters including the probe, and its image encoder dominates latency.

### 5.4 Label efficiency

All budgets use nested training subsets and fixed update counts: 300 for the probe and 1,500 for U-Net.

| Method | 25 masks | 100 masks | 400 masks |
|---|---:|---:|---:|
| Linear probe | 0.7156 | 0.7416 | 0.7488 |
| Probe → box+point → SAM | **0.7821** | **0.8061** | **0.8190** |
| U-Net 256, fixed endpoint | 0.5103 | 0.6198 | 0.6964 |

Automatic SAM exceeds U-Net by 0.2718 [0.2185, 0.3243], 0.1863 [0.1333, 0.2390], and 0.1226 [0.0757, 0.1692]. The decreasing gap shows that the direct model benefits more from additional local labels, while the frozen representation provides a strong low-label starting point. This fixed-compute study does not claim final convergence or universal label efficiency.

The fixed budget matters at 400 masks. With the same 400 masks and architecture, the two-stage recipe of Section 5.3 (2,250 updates, 128 then 256 pixels) reaches 0.8267 and closes the 0.1226 gap. The curve therefore shows that SAM gives a stronger start when labels or optimization are limited, not a higher ceiling; whether longer U-Net training also narrows the 25- and 100-mask gaps remains untested.

### 5.5 Frozen positive-domain transfer

| Frozen method | CVC-ClinicDB frame Dice | Sequence-clustered 95% CI |
|---|---:|---:|
| Linear probe | 0.4243 | [0.3624, 0.4892] |
| Probe → box+point → SAM | **0.4587** | **[0.3793, 0.5405]** |
| Mature U-Net 256 | 0.3405 | [0.2841, 0.4002] |

SAM exceeds U-Net by 0.1182 [0.0672, 0.1684] and the probe by 0.0344 [0.0110, 0.0576]. Relative transfer favors SAM, but the absolute decrease from development is severe. U-Net also has substantial external seed variability. These results support robustness relative to the local control, not clinical adequacy.

### 5.6 External negative control

| Frozen method | Mean false area | 95% CI | Rate with ≥1% false area | Non-empty output |
|---|---:|---:|---:|---:|
| Linear probe | 5.81% | [5.52%, 6.11%] | 87.94% | 100.00% |
| Probe → box+point → SAM | **5.21%** | **[4.88%, 5.54%]** | **79.42%** | 100.00% |
| Mature U-Net 256 | 17.33% | [16.90%, 17.76%] | 99.31% | 99.83% |

Automatic SAM reduces mean false area relative to U-Net by 12.12 percentage points [11.59, 12.67] and the ≥1% rate by 19.89 points [17.67, 22.17]. It also reduces average false area and ≥1% rate relative to the probe, but that reduction comes from prompting only the probe's largest component, not from the decoder. In a post hoc check ([`scripts/34_negative_postprocessing_check.py`](scripts/34_negative_postprocessing_check.py)), the decoder enlarges that component by 1.36 percentage points [1.23, 1.50] and grows it in 83.6% of image–seed pairs. The advantage over U-Net survives the same single-component rule: U-Net's largest component alone covers 15.58% of the image, 10.37 points [9.85, 10.90] more than SAM. SAM's 95th-percentile largest component is 16.28% of the image, and because the probe fires on every image, the always-prompt pipeline never emits an empty output.

## 6. Scientific contribution

The contribution is the combined controlled evidence:

1. **Operational definition of prompt absence.** Empty spatial input, fixed tokens, multiple candidates, and oracle selection are not treated as interchangeable “no-prompt” settings.
2. **Failure decomposition.** The study separates candidate availability from candidate selection and shows that their importance changes with the prompt regime.
3. **Localization controls.** A train-derived fixed prior and a context permutation test prevent gains from being attributed automatically to image-specific conditioning.
4. **Mitigation with an honest comparator.** Automatic localization recovers much of the prompt gap, while a strengthened direct U-Net tests whether retaining SAM adds value.
5. **Utility beyond one development score.** Label budgets, frozen positive transfer, a negative proxy, and full RGB latency reveal when pretraining helps and when computational cost dominates.
6. **Reproducible execution.** Plans, split IDs, hashes, checkpoints, per-image outputs, resumable state, and paired uncertainty are preserved. The current suite passes 31 tests and 2 subtests.

Automatic prompt generation itself is not claimed as novel. Self-prompting, AutoSAM, learned prompt embeddings, and prompt perturbation (PP-SAM, Stable-SAM) have prior art, and Mazurowski et al. already report the gap between SAM's chosen mask and the best of its three candidates on medical data. The defensible novelty candidate is the integrated diagnostic and utility evaluation under shared IDs and frozen external transfer.

## 7. Limitations

- Phase B scores are development-selected, and the historical Kvasir test was not reopened after these choices.
- Kvasir-SEG contains only positive images and no public patient/video grouping.
- HyperKvasir `cecum` is a class-level negative proxy, not a pixel-reviewed negative examination set.
- CVC-ClinicDB contains correlated frames; sequence bootstrap reduces but does not remove all clinical dependence concerns.
- SAM and U-Net differ in pretraining, input size, compute, and recipe. The comparison evaluates systems, not a single isolated architecture variable.
- U-Net training has not established a universal convergence ceiling, and no specialist modern polyp model or SAM2 model has yet been executed locally.
- Label-efficiency budgets are short (1,500 U-Net updates); the 400-mask gap disappears under the longer recipe, so low-label gaps may also be partly budget-driven.
- CPU latency is implementation- and hardware-specific; it is not a GPU, energy, or deployment benchmark.
- The project does not measure annotator time, reader variability, calibration, sensitivity at a fixed false-positive rate, or patient-level outcomes.

## 8. Recommended next experiment

The immediate next experiment should add a **presence or abstention gate** before producing a box and point. It should be trained only on source training data with explicit positive and negative labels, freeze its operating threshold before external evaluation, and compare:

1. always-prompt automatic SAM;
2. gated automatic SAM with an empty-output option;
3. gated U-Net or a direct segmenter with the same decision threshold;
4. oracle presence as a diagnostic ceiling.

Primary endpoints should include sensitivity, frame-level false-positive rate, area-weighted false activation, AUROC/AUPRC, expected calibration error, and patient/video-clustered intervals. The dataset should expose patient or video groups and contain consecutive negative frames. A modern self-prompting SAM/SAM2 method and a specialist polyp model are useful only after this rejection problem is addressed.

## 9. Reproducibility map

- Core Phase B report: [`results/phase_b/full_dev_20261012/report.md`](results/phase_b/full_dev_20261012/report.md)
- Complete experimental report: [`docs/resultados_experimentacion_20261017.md`](docs/resultados_experimentacion_20261017.md)
- Label-efficiency protocol and results: [`docs/protocolo_eficiencia_etiquetas_20261017.md`](docs/protocolo_eficiencia_etiquetas_20261017.md), [`results/model_comparison/label_efficiency_20261017/`](results/model_comparison/label_efficiency_20261017/)
- External positive protocol and results: [`docs/protocolo_validacion_externa_20261017.md`](docs/protocolo_validacion_externa_20261017.md), [`results/model_comparison/external_clinicdb_20261017/`](results/model_comparison/external_clinicdb_20261017/)
- External negative protocol and results: [`docs/protocolo_especificidad_negativos_20261017.md`](docs/protocolo_especificidad_negativos_20261017.md), [`results/model_comparison/negative_specificity_20261017/`](results/model_comparison/negative_specificity_20261017/)
- Figure-generation script: [`scripts/33_make_paper_figures.py`](scripts/33_make_paper_figures.py)
- WAIMLAp template generator: [`docs/waimlap_2026/generar_borrador.py`](docs/waimlap_2026/generar_borrador.py)

## References

1. A. Kirillov et al. “Segment Anything.” *ICCV*, 2023.
2. D. Jha et al. “Kvasir-SEG: A Segmented Polyp Dataset.” *MMM*, 2020.
3. T. Zhou et al. “Can SAM Segment Polyps?” arXiv:2304.07583, 2023.
4. Q. Wu, Y. Zhang, and M. Elbatel. “Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation.” arXiv:2308.07624, 2023.
5. T. Shaharabany et al. “AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder.” arXiv:2306.06370, 2023.
6. O. Ronneberger, P. Fischer, and T. Brox. “U-Net: Convolutional Networks for Biomedical Image Segmentation.” *MICCAI*, 2015.
7. J. Bernal et al. “WM-DOVA Maps for Accurate Polyp Highlighting in Colonoscopy.” *Computerized Medical Imaging and Graphics*, 2015.
8. H. Borgli et al. “HyperKvasir, a Comprehensive Multi-Class Image and Video Dataset for Gastrointestinal Endoscopy.” *Scientific Data*, 2020.
9. M. A. Mazurowski et al. “Segment Anything Model for Medical Image Analysis: An Experimental Study.” *Medical Image Analysis*, 102918, 2023.
10. M. M. Rahman et al. “PP-SAM: Perturbed Prompts for Robust Adaptation of Segment Anything Model for Polyp Segmentation.” *CVPR Workshops*, 2024.
11. Q. Fan et al. “Stable Segment Anything Model.” arXiv:2311.15776, 2023.

## Submission fields still required

- Confirmed author names and order.
- Affiliations and corresponding-author email.
- Conflict-of-interest, funding, and AI-assistance statements required by the event.
- Final human review of claims, references, figure legibility, and template compliance.
- Explicit decision on whether the event accepts work-in-progress results and whether a separate poster is required.
