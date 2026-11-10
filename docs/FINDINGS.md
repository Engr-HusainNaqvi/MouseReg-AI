# MouseReg-AI: multimodal 3D MRI ↔ CT registration for orthotopic breast-cancer mouse models (findings)

**Task:** deformable registration of T2-weighted MRI to whole-body micro-CT in mice bearing orthotopic mammary-fat-pad tumours.

**Data:** a virtual cohort built on **real micro-CT of 30 mice (221 scans)**. Each scan has an implanted orthotopic tumour, a synthetic T2w MRI and an exactly known misalignment (section 2).

**Evaluation:** **58 frozen test pairs** from 29 scans of **4 held-out mice**. The classical CPU baselines were run on a paired subset of 30 of these pairs.

**Hardware:** a Colab A100 GPU ran training, learned inference and instance optimisation. The classical methods ran on CPU, 2 threads per job.

---

## 1. Problem summary

**Domain:** preclinical cancer imaging. The underlying problem is image registration, a computer-vision inverse problem.

Orthotopic breast-cancer models implant tumour cells (e.g. 4T1, E0771) into the 4th (inguinal) mammary fat pad of the mouse. Two modalities are usually acquired:
- **MRI** (T2-weighted) shows the tumour, its necrotic core and the surrounding soft tissue.
- **Micro-CT** shows bone, lung metastases and the attenuation map needed for PET/SPECT quantification and small-animal radiotherapy planning.

Fusing them means mapping every MRI voxel onto the CT voxel of the same tissue. Three things make this hard:
- **The animal moves between scanners on a transfer bed.** This gives a **rigid offset** of several mm and up to about 10°.
- **The soft abdomen and fat pad deform with posture**, typically by 1–2.5 mm, while bone stays rigid.
- **The two modalities share no intensity relationship.** Fat is bright on T2w MRI and dark on CT, bone is dark on both, and lung is dark on MRI.

Both established approaches fall short:
- **Classical mutual-information registration** is slow (minutes per pair on CPU) and fragile at the tumour, which is the structure the biologist cares about.
- **Deep registration networks** are fast but need training data. No public dataset of paired mouse MRI–CT with orthotopic tumours exists, so ground-truth accuracy cannot be measured on real pairs.

**Research question:** can a learned, anatomy-aware registration pipeline, trained only on *simulated* MRI–CT pairs built from real mouse micro-CT, register a held-out cohort more accurately and faster than established multimodal methods? The comparison puts particular weight on accuracy at the tumour, and the deformation must stay diffeomorphic (no tissue folding).

## 2. Data: a virtual orthotopic cohort with known ground truth

| Step | What is done |
|---|---|
| Real anatomy | The source is the Rosenhain et al. database (*Sci. Data* 2018, CC BY 4.0), native and contrast-enhanced sets: 224 longitudinal whole-body micro-CT scans of 30 female BALB/c nu/nu mice. Images are in Hounsfield units at 0.28 mm, with up to 14 time points per animal (−1 h to 240 h) and manual organ labels. 221 scans were usable (137 native, 84 contrast-enhanced); 3 had labels only at a mismatched resolution and were dropped. |
| Grid | Resampled to 0.4 mm isotropic, 160 × 80 × 112 voxels (head–tail × dorsal–ventral × left–right), body-centred. |
| Orthotopic tumour | Implanted into the left or right inguinal (4th) mammary fat pad. It is a lobulated, heterogeneous mass that bulges the skin and displaces local tissue. Median volume is 190 mm³ (5th–95th percentile 33–432 mm³). 54% have a necrotic core. Left and right sides are balanced (53% / 47%). |
| Synthetic T2w MRI | Built from the same anatomy, with tissue-specific T2w signal and per-case protocol jitter. Real CT micro-texture modulates the signal, so there is no shortcut pattern. The simulator adds a smooth B1/coil bias field, 0.8 mm anisotropic slice blur and Rician noise. The CT→MRI intensity mapping is non-monotonic, which defeats simple correlation metrics. |
| Misalignment | Generated on the fly on the GPU: a rigid bed transfer (up to ±10°/±6°/±6° and ±3 mm) composed with a diffeomorphic posture deformation. The deformation is a smooth stationary velocity field of 0.6–2.4 mm plus a localised bump at the tumour. Its inverse is exact, so the **true correspondence of every voxel is known**. |
| Split | Split by **mouse**, never by scan: 152 scans from 21 mice for training, 40 scans from 5 mice for validation, and 29 scans from 4 held-out mice for testing. Each test scan gets 2 frozen, seeded misalignments, giving **58 test pairs**. |

During training the models see *unlimited* fresh misalignments of the training mice. The test set was frozen before any model was trained.

### Metrics
- **Dense TRE:** the mean, median and 95th-percentile distance between the estimated and true MRI location of every body voxel, in mm.
- **Tumour TRE:** the same distance restricted to tumour voxels.
- **Tumour centroid error.**
- **Dice:** over 10 structures (bone, lung, heart, liver, intestine, bladder, spleen, stomach, kidneys, tumour) after the MRI labels are warped.
- **Tumour HD95.**
- **Folding:** the percentage of body voxels with det J ≤ 0.
- **Runtime per pair.**

The paired Wilcoxon tests use scan-level means (29 scans), so the two misalignments of the same scan are not counted as independent samples.

## 3. Five candidate AI solutions (all implemented and benchmarked)

| # | Candidate | Family | Why it is a candidate |
|---|---|---|---|
| 1 | **Rigid mutual information**: SimpleITK Euler3D, Mattes MI, 3-level pyramid | classical optimisation | the standard first step on preclinical PET/MR/CT workstations |
| 2 | **Rigid + B-spline free-form deformation**: Mattes MI, 8 mm control grid, L-BFGS-B | classical, parametric deformable | the textbook multimodal deformable method (Rueckert et al.) |
| 3 | **ANTs SyNRA**: rigid + affine + symmetric diffeomorphic normalisation, Mattes MI | classical, diffeomorphic | the de-facto standard in neuro and preclinical atlas work |
| 4 | **Unsupervised deep registration (VoxelMorph-style)**: a U-Net predicts a stationary velocity field, integrated by scaling and squaring; loss is MIND-SSC + diffusion regulariser | deep learning, label-free | one forward pass, no labels |
| 5 | **Instance optimisation (ConvexAdam-style)**: after the rigid stage, the SVF is optimised directly per pair with Adam on MIND-SSC + diffusion regulariser (150 iterations, GPU) | gradient-based optimisation on modality-independent features | among the strongest Learn2Reg approaches; needs no training |

The rigid stage that feeds candidates 4–5 and all the hybrids is itself learned. **RigidNet** is a 3D CNN that regresses 6 rigid parameters in one pass. It is trained with a dense point loss on simulated misalignments.

## 4. Five hybrid methodologies (all implemented and tested)

| Hybrid | Combination | What it is meant to fix | Result on 58 test pairs |
|---|---|---|---|
| **A: Learned initialisation + IO** | RigidNet (one pass) → 60 iterations of rigid MIND-SSC instance optimisation | one-pass regression is coarse; pure optimisation can fall into local minima | rigid TRE 1.26 → **1.10 mm**. This matches SimpleITK rigid MI (1.12 mm on the paired subset) in **0.7 s** instead of 2.8 s |
| **B: Weakly supervised DiffNet** | DiffNet trained with MIND-SSC + soft Dice on 10 structures, including the tumour | intensity similarity alone is ambiguous inside homogeneous soft tissue | organ Dice 0.776 → 0.818; TRE 0.93 → 0.90 mm |
| **C: Anatomy-aware synthetic supervision** | B + direct supervision on the known simulated flow (dense TRE loss) | label-free losses do not say *where* each voxel should go | TRE **0.86 mm**, tumour TRE 0.97 mm; significantly better than B and the unsupervised network (p < 1e-6) |
| **D: Unsupervised DiffNet + IO** | candidate 4's prediction warm-starts 50 iterations of deformable IO | networks underfit individual pairs | TRE 0.72 mm |
| **E: Proposed: anatomy-aware DiffNet + IO refinement** | C's prediction and velocity field warm-start 50 iterations of MIND-SSC IO; diffeomorphic throughout | combines the learned anatomical prior with per-pair fidelity, at a fraction of the time of IO from scratch | **TRE 0.69 mm, tumour TRE 0.81 mm, tumour Dice 0.794, no folding, 2.0 s** |

## 5. Results

### 5.1 Paired comparison of all methods (30 pairs, 29 scans)

| Method | TRE mean (mm) | TRE p95 | Tumour TRE | Tumour centroid err | Organ Dice | Tumour Dice | Tumour HD95 | Folds % | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Initial (no registration) | 3.52 ± 0.74 | 5.37 ± 0.98 | 3.68 ± 1.30 | 3.58 ± 1.31 | 0.342 ± 0.100 | 0.257 ± 0.211 | 3.45 ± 1.19 | – | – |
| Rigid MI (SimpleITK) | 1.12 ± 0.38 | 1.95 ± 0.66 | 1.26 ± 0.57 | 1.12 ± 0.57 | 0.696 ± 0.086 | 0.629 ± 0.199 | 1.23 ± 0.51 | 0.000 | 2.8 ± 0.6 |
| RigidNet (learned, 1 pass) | 1.23 ± 0.37 | 2.15 ± 0.64 | 1.41 ± 0.57 | 1.26 ± 0.58 | 0.668 ± 0.083 | 0.590 ± 0.207 | 1.33 ± 0.55 | – | 0.01 |
| RigidNet + MIND IO | 1.08 ± 0.36 | 1.93 ± 0.64 | 1.24 ± 0.53 | 1.09 ± 0.53 | 0.703 ± 0.080 | 0.635 ± 0.196 | 1.17 ± 0.46 | – | 0.8 ± 0.6 |
| Rigid + B-spline FFD, MI (SimpleITK) | 1.31 ± 0.50 | 3.63 ± 1.97 | 0.85 ± 0.38 | 0.57 ± 0.30 | 0.652 ± 0.107 | 0.752 ± 0.145 | 0.85 ± 0.35 | 1.428 ± 2.873 | 390 ± 130 |
| ANTs SyNRA (MI) | 1.01 ± 0.44 | 1.98 ± 1.06 | 1.01 ± 0.99 | 0.79 ± 0.88 | 0.732 ± 0.081 | 0.733 ± 0.184 | 1.05 ± 1.10 | 0.000 | 88 ± 23 |
| Rigid + MIND-SSC IO only (ConvexAdam-style) | 0.74 ± 0.27 | 1.59 ± 0.64 | 0.75 ± 0.45 | 0.53 ± 0.46 | 0.807 ± 0.057 | 0.800 ± 0.146 | 0.71 ± 0.48 | 0.000 | 4.6 ± 0.7 |
| DiffNet unsupervised (VoxelMorph-style) | 0.92 ± 0.26 | 1.77 ± 0.52 | 1.02 ± 0.45 | 0.70 ± 0.44 | 0.778 ± 0.049 | 0.759 ± 0.131 | 0.88 ± 0.41 | 0.000 | 0.8 ± 0.6 |
| DiffNet unsup + IO | 0.71 ± 0.24 | 1.52 ± 0.55 | 0.76 ± 0.44 | 0.55 ± 0.47 | 0.818 ± 0.042 | 0.805 ± 0.114 | 0.72 ± 0.43 | 0.000 | 2.1 ± 0.6 |
| DiffNet weakly sup. (+Dice) | 0.90 ± 0.26 | 1.74 ± 0.53 | 0.98 ± 0.41 | 0.60 ± 0.40 | 0.819 ± 0.046 | 0.781 ± 0.127 | 0.79 ± 0.38 | 0.000 | 0.8 ± 0.6 |
| DiffNet anatomy-aware hybrid (+Dice+GT flow) | 0.85 ± 0.27 | 1.67 ± 0.53 | 0.91 ± 0.39 | 0.64 ± 0.39 | 0.807 ± 0.051 | 0.777 ± 0.122 | 0.83 ± 0.37 | 0.000 | 0.8 ± 0.6 |
| **Proposed: hybrid DiffNet + IO refinement** | **0.69 ± 0.24** | **1.48 ± 0.56** | **0.72 ± 0.40** | **0.52 ± 0.44** | **0.823 ± 0.043** | **0.814 ± 0.102** | **0.68 ± 0.41** | **0.000** | 2.1 ± 0.6 |
| Oracle (interpolation ceiling) | 0.00 | 0.00 | 0.00 | 0.02 ± 0.01 | 0.977 ± 0.012 | 0.974 ± 0.021 | 0.25 ± 0.19 | – | – |

Times are per pair. The learned and IO methods ran on the A100 and the classical methods on CPU (2 threads), so the hardware differs. The comparison shows practical turnaround, not algorithmic FLOPs. Dense TRE is averaged over all body voxels.

### 5.2 Deep and IO pipelines on the full test set (58 pairs)

| Method | Dense TRE (mm) | TRE p95 | Tumour TRE | Organ Dice | Tumour Dice | Tumour HD95 (mm) | Median time (s) |
|---|---|---|---|---|---|---|---|
| Initial misalignment | 3.42 ± 0.73 | 5.32 | 3.70 | 0.346 | 0.260 | 3.48 | – |
| RigidNet + rigid IO (Hybrid A) | 1.10 ± 0.35 | 1.97 | 1.34 | 0.701 | 0.620 | 1.27 | 0.66 |
| DiffNet unsupervised (cand. 4) | 0.93 ± 0.25 | 1.78 | 1.10 | 0.776 | 0.748 | 0.94 | 0.67 |
| DiffNet anatomy-aware (Hybrid C) | 0.86 ± 0.26 | 1.67 | 0.97 | 0.809 | 0.772 | 0.88 | 0.67 |
| MIND-SSC IO only (cand. 5) | 0.74 ± 0.25 | 1.61 | 0.87 | 0.805 | 0.780 | 0.80 | 4.43 |
| DiffNet unsup + IO (Hybrid D) | 0.72 ± 0.23 | 1.54 | 0.86 | 0.813 | 0.784 | 0.80 | 1.96 |
| **Proposed (Hybrid E)** | **0.69 ± 0.23** | **1.49** | **0.81** | **0.820** | **0.794** | **0.77** | **1.95** |

Paired Wilcoxon signed-rank tests on scan-level means (n = 29 scans) compare the proposed pipeline with each other method:
- **vs. ANTs SyNRA:** dense TRE −0.32 mm (p = 4e-9), tumour TRE −0.31 mm (p = 0.005), tumour Dice +0.082 (p = 2e-5). The proposed pipeline is also **~45× faster** (2 s on GPU vs 88 s on CPU).
- **vs. B-spline FFD:** dense TRE −0.62 mm (p = 9e-8), tumour Dice +0.058 (p = 0.003). B-spline FFD was the only method that folded tissue, averaging 1.4% of body voxels and reaching 11.8% in one pair. It was also the slowest at 6.5 min per pair.
- **vs. SimpleITK rigid MI:** dense TRE −0.43 mm, tumour TRE −0.55 mm, tumour Dice +0.185 (all p < 1e-7).
- **vs. IO from scratch (cand. 5), over all 58 pairs:** dense TRE −0.045 mm (p = 3e-7, better on 26/29 scans), tumour TRE −0.055 mm (p = 0.03), tumour Dice +0.014 (p = 0.05). It is also **2.3× faster** (1.95 vs 4.43 s), because the learned prior lets IO stop after 50 iterations instead of 150.
- **vs. Hybrid D** (the same refinement applied to the unsupervised network): dense TRE −0.024 mm (p = 1e-8, better on 28/29 scans), tumour TRE −0.045 mm (p = 1e-5), tumour Dice +0.010 (p = 1e-4). This isolates the **value of anatomy-aware synthetic supervision**: small in mm, but consistent across scans.
- **Success rate:** dense TRE was below 1 mm in **90%** of pairs, against 83% for IO only, 64% for the unsupervised DiffNet and 43% for RigidNet + IO. Tumour TRE was below 1 mm in 76% of pairs (IO only 69%, Hybrid D 71%).

**Diffeomorphism:** every learned and IO deformation had **0.000% folding voxels**, because the stationary velocity field is integrated with 7 scaling-and-squaring steps. ANTs SyN also produced no folding.

### 5.3 Honest observations
- **Most of the accuracy comes from per-pair optimisation of modality-independent features (MIND-SSC), not from the network.**
  - IO from scratch (0.74 mm) already beats every classical method.
  - The learned prior adds a consistent but small 0.045 mm and halves the runtime.
  - A one-pass network alone (0.86–0.93 mm) is fast (0.7 s) but less accurate than IO.
- **The classical MI deformable methods were weaker than expected at the whole-body level.**
  - B-spline FFD *improved* the tumour: its tumour TRE of 0.85 mm is the best of the classical methods.
  - But it *worsened* the dense TRE (1.31 vs 1.12 mm for rigid) and folded tissue.
  - The likely reason: one global joint histogram (mutual information) fits poorly with a T2w↔CT mapping that is non-monotonic and differs per tissue. MIND-SSC uses local self-similarity, which is preserved across modalities.
- **Weak (Dice) supervision improved overlap but barely improved geometric accuracy** (0.93 → 0.90 mm TRE). Only the added dense synthetic-flow loss moved TRE, to 0.86 mm. Dice is a weak proxy for correspondence inside homogeneous organs.
- **The oracle row is the measurement ceiling.** Even a perfect mapping gives tumour Dice 0.97 and HD95 0.25 mm, because nearest-neighbour label resampling loses a few percent.

## 6. Best methodology and justification

**Chosen: Hybrid E**, a three-stage pipeline:
1. **RigidNet** followed by rigid MIND-SSC instance optimisation.
2. An **anatomy-aware diffeomorphic DiffNet**. It is trained on unlimited simulated orthotopic MRI–CT pairs with three loss terms: MIND-SSC, organ/tumour Dice, and the synthetic ground-truth flow.
3. **50 iterations of MIND-SSC instance optimisation**, warm-started from the network's velocity field.

| Criterion | Proposed | Best classical (ANTs SyNRA) | IO only |
|---|---|---|---|
| Dense TRE (mm) | **0.69** | 1.01 | 0.74 |
| Tumour TRE / Dice | **0.81 mm / 0.794** | 1.01 mm / 0.733 | 0.87 mm / 0.780 |
| Folding | **0%** | 0% | 0% |
| Time per pair | **2.0 s (GPU)** | 88 s (CPU) | 4.4 s (GPU) |
| Needs labels at test time | no | no | no |

Why this one:
1. **It is the most accurate method on every primary endpoint, including the tumour.** The difference is significant against every baseline.
2. **It is fast enough to run during an imaging session:** 2 s per animal, against 1.5 minutes for ANTs and 6.5 minutes for B-spline FFD.
3. **It preserves topology by construction**, which tumour-mask transfer and dose mapping require: no tissue is torn or folded.
4. **It degrades gracefully.** The IO stage optimises a label-free, modality-independent objective. On an unusual animal it therefore falls back towards plain IO (0.74 mm) rather than failing.
5. **It is data-efficient.** It was trained without a single real paired MRI–CT. The training cohort is 21 real micro-CT animals; everything else is simulated.

## 7. Mini abstract

> **Anatomy-aware learned registration of MRI to micro-CT in orthotopic breast-cancer mouse models.** Combining T2-weighted MRI (tumour soft tissue) with micro-CT (bone, lung, attenuation) is routine in preclinical oncology. However, moving the animal between scanners introduces rigid offsets and posture deformation, and no public paired dataset with ground-truth correspondence exists. We built a virtual cohort from 221 real whole-body micro-CT scans of 30 mice. Each scan received an implanted orthotopic mammary-fat-pad tumour, a physics-inspired synthetic T2w MRI and a known diffeomorphic misalignment. We benchmarked five candidate approaches (rigid MI, B-spline FFD, ANTs SyN, unsupervised VoxelMorph-style learning and MIND-SSC instance optimisation) and five hybrids on 58 test pairs from 4 held-out mice. The best pipeline chains a learned rigid stage, a diffeomorphic network trained with MIND-SSC, organ/tumour Dice and synthetic ground-truth flow, and a 50-iteration MIND-SSC refinement. It reached a dense target registration error of 0.69 ± 0.23 mm (initial 3.42 mm), a tumour TRE of 0.81 mm and a tumour Dice of 0.794, with zero folding, in 2 s per animal. It was significantly more accurate than ANTs SyN (1.01 mm, 88 s; p = 4e-9), B-spline FFD (1.31 mm, 1.4% folding, 6.5 min) and instance optimisation from scratch (0.74 mm, 4.4 s; p = 3e-7). Simulation from real anatomy is enough to train a fast, accurate and topology-preserving MRI–CT registration for orthotopic tumour models. Most of the accuracy comes from modality-independent instance optimisation; the learned anatomical prior makes it faster and more consistent.

## 8. Implementation outline (reproducible)

```
src/mousereg/prep.py      Rosenhain CT + labels -> 0.4 mm grid; orthotopic MFP tumour implant; synthetic T2w MRI
src/mousereg/warp.py      grids, SVF scaling-and-squaring, rigid transforms, GT misalignment generator, MIND-SSC, Jacobian
src/mousereg/nets.py      RigidNet (6-DoF regression), DiffNet (SVF U-Net, half-resolution velocity)
src/mousereg/common.py    GPU case bank, mouse-level split, rigid/deformable instance optimisation, metrics
scripts/prep_all.py       ranged download of only the needed files from figshare + parallel preprocessing (~6 min)
scripts/make_testset.py   frozen, seeded test misalignments (2 per test scan)
scripts/train.py          RigidNet (3k it) and DiffNet unsup / weak / full (5k it each, batch 2, bf16): ~26 min on an A100
scripts/eval_deep.py      learned + IO pipelines on the test pairs
scripts/classical.py      SimpleITK rigid / B-spline and ANTs SyNRA on CPU; exact mappings via padded coordinate images
scripts/summarize.py      tables + scan-level paired Wilcoxon tests
scripts/make_figures.py   figures (make_fig_data.py: cohort figure; qual.py: qualitative slices)
scripts/colab_run.sh      the whole pipeline in one command on a Colab GPU (~1 h including classical baselines)
app/app.py                Streamlit "MouseReg Studio": run the pipeline on held-out pairs and inspect the result
```

**Selection discipline:** the architecture, loss weights and iteration counts were fixed before the test set was evaluated. The test set was frozen before training and scored once.

## 9. Limitations
- **The MRI is synthetic.** The simulator models T2w RARE tissue contrast, the bias field, anisotropic slices and Rician noise. It does not model motion, susceptibility or chemical-shift artefacts, or real coil profiles. Accuracy on real MRI will be lower, and validation on paired real scans is the next step.
- **The tumour is implanted, not grown.** Real 4T1 tumours infiltrate and vascularise differently.
- **The posture deformation is smooth.** It is a stationary velocity field plus a tumour bump. A real bed transfer can make organs slide against each other (the bowel against the abdominal wall), which no single diffeomorphism can represent.
- **The test set holds only 4 mice.** Scans of the same mouse are correlated across time points, so the scan-level tests are somewhat optimistic.
- **The classical baselines were not exhaustively tuned.** They used standard parameters, and their CPU cost limited them to 30 of the 58 pairs.
- **This is a research prototype.** It is not validated for regulatory or clinical use.

## References
- Rosenhain S, *et al.* A preclinical micro-computed tomography database including 3D whole body organ segmentations. *Sci Data* 5, 180294 (2018). doi:10.1038/sdata.2018.294
- Heinrich MP, *et al.* MIND: Modality independent neighbourhood descriptor for multi-modal deformable registration. *Med Image Anal* 16(7):1423–1435 (2012). doi:10.1016/j.media.2012.05.008
- Balakrishnan G, *et al.* VoxelMorph: a learning framework for deformable medical image registration. *IEEE TMI* 38(8):1788–1800 (2019). doi:10.1109/TMI.2019.2897538
- Dalca AV, *et al.* Unsupervised learning of probabilistic diffeomorphic registration for images and surfaces. *Med Image Anal* 57:226–236 (2019). doi:10.1016/j.media.2019.07.006
- Avants BB, *et al.* Symmetric diffeomorphic image registration with cross-correlation. *Med Image Anal* 12(1):26–41 (2008). doi:10.1016/j.media.2007.06.004
- Rueckert D, *et al.* Nonrigid registration using free-form deformations: application to breast MR images. *IEEE TMI* 18(8):712–721 (1999). doi:10.1109/42.796284
- Siebert H, Hansen L, Heinrich MP. Fast 3D registration with accurate optimisation and little learning for Learn2Reg 2021. *MICCAI 2021 Workshops* (2022).
