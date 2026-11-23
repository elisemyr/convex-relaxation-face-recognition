# Face access control with convex relaxation

Advanced Applied Applied Statistical Modeling, AIDAMS, CentraleSupélec × ESSEC

## The problem

A company wants its employees to open the doors with their face. In real life this is hard. People wear masks or glasses, and the light is not the same in the morning and in the evening. Each employee also gives only a few photos. The system must also refuse visitors who were never enrolled. A basic face classifier would just give them the name of the closest employee.

## The idea

When a face arrives at the door, I try to rebuild it as a mix of the gallery photos, using as few photos as possible. If the person is an employee, a few of their own photos should be enough. The photos used give the name. If the mix needs photos from many different people, the face is probably a stranger.

With y the face, Φ the matrix of gallery photos and x the weights, I want y ≈ Φx with few non-zero weights. Counting non-zero weights is the ℓ0 norm, and minimising it is NP-hard. So I replace it with the ℓ1 norm, its convex envelope:

```
min  1/2 ||Φx - y||² + λ ||x||₁
```

This is a Lasso. I solve it with FISTA, an algorithm that repeats a gradient step and a soft-thresholding step that sets small weights to zero.

The project uses the same relaxation in a few places:

| Part of the system | Hard problem | Convex version I solve |
|---|---|---|
| Sparse representation classification, SRC | min ‖x‖₀ | Lasso on ‖x‖₁ |
| Masks and glasses: y = Φx + e | few hidden pixels, ‖e‖₀ small | add λ‖e‖₁ |
| Cleaning shadows in the gallery with Robust PCA | min rank(L) + ‖S‖₀ | min ‖L‖* + λ‖S‖₁ |
| Random projection features | few measurements | ℓ1 recovery under RIP |

To open the door, I use the Sparsity Concentration Index, SCI. It measures how much of the weight sits on one person. If the SCI is above a threshold τ, the door opens. If more than 40% of the pixels are hidden, the system asks for the badge.

I compare SRC with two other methods. The first is ridge regression, called CRC, which uses an ℓ2 penalty instead of ℓ1 and tells me if sparsity really helps. The second is Eigenfaces with a linear SVM, the classic face recognition method.

## Data

I use the cropped version of Extended Yale B: 38 people with about 64 photos each, under many light directions. The download link on the official page is broken, so I use a copy of the same images on Kaggle:

- Kaggle copy, `ExtYaleBCroppedPNG`: https://www.kaggle.com/datasets/tbourton/extyalebcroppedpng
- Mirror of the original `CroppedYale.zip`: https://academictorrents.com/details/aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247

Unzip the dataset anywhere inside `data/`. The loader looks in every subfolder and reads the person and the light angles from the file names. The data itself is not in this repository.

To add a little more difficulty, I added some onclusion on the photos.
Yale B has no masks or glasses, so I draw them on the test photos only. A mask is a grey patch on the bottom of the face, glasses are a dark band at eye level. The gallery photos stay clean, like when an employee enrols.

Dataset references: Georghiades, Belhumeur and Kriegman, PAMI 2001. Lee, Ho and Kriegman, PAMI 2005.

## How to run

```bash
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

python3 -m pytest -q                         # checks the solvers
python3 experiments/run_all.py               # every experiment, results saved as CSV
python3 experiments/run_all.py --exp E2 E5   # only some experiments
python3 experiments/plots.py                 # figures in results/yaleb/figures
```

The full run is slow because of the ℓ1 solvers. To make it faster, lower `seeds`, `fista_iter` or `max_test_per_person` in `config.py`. All random choices use fixed seeds, so the results can be reproduced.

## Project structure

```
config.py                all settings and seeds
src/data.py              Yale B loader
src/splits.py            employees, validation visitors, test visitors
src/occlusion.py         masks, glasses, blocks and pixel corruption
src/features.py          downsampling, PCA, random projections
src/solvers.py           FISTA, the occlusion model, ridge, cvxpy references
src/src_classifier.py    SRC, CRC, the SCI score and the door decision
src/rpca.py              Robust PCA
src/baselines.py         Eigenfaces + SVM
src/evaluation.py        accuracy, FAR, FRR, choice of tau
experiments/run_all.py   experiments E1 to E6
experiments/plots.py     figures from the CSV files
test/test_core.py        unit tests
```

## Experiments

The people are split into three groups. Employees are in the gallery. Visitors are never enrolled, and there are two groups of them: I choose the threshold τ on the first group, and I measure the errors on the second. If I used the same visitors for both, the result would look better than it really is.

Two error rates come back often. The false accept rate, FAR, is the share of visitors who get in. The false reject rate, FRR, is the share of employees who are refused or given the wrong name.

- E1: accuracy on clean faces when the number of features changes, for pixels, PCA and random projections
- E2: accuracy when part of the face is hidden, for 4 kinds of occlusion
- E3: accuracy when each employee has more or fewer gallery photos
- E4: what changes when I clean the gallery with Robust PCA
- E5: the door decision, with τ chosen so that FAR stays under 1% on the first group of visitors
- E6: time for one decision, measured during E5

## Main results

All numbers are a mean over 5 random splits

On clean faces, SRC finds the right person 95.6% of the time, against 85.4% for Eigenfaces. When a block hides 30% of the face, SRC with the error term e keeps 73.8%, and Eigenfaces falls to 28.0%. Large masks break every method: at 40% of the face, all are under 20%.

Ridge is as good as SRC on clean faces with 504 pixels. Sparsity matters when there are few measurements: with 30 random projections, SRC gets 83.9% against 67.4% for ridge. Robust PCA lowered the accuracy, from 95.6% to 88.5% for SRC on clean faces.

For the door, with SRC and the error term, 0.9% of visitors get in and about one employee in five is refused.

## Use of AI

I used Claude to help me with the code for some parts

## References

- J. Wright, A. Yang, A. Ganesh, S. Sastry, Y. Ma. Robust face recognition via sparse representation. IEEE PAMI, 2009.
- E. Candès, X. Li, Y. Ma, J. Wright. Robust principal component analysis? Journal of the ACM, 2011.
- L. Zhang, M. Yang, X. Feng. Sparse representation or collaborative representation: which helps face recognition? ICCV, 2011.
- A. Beck, M. Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. SIAM J. Imaging Sciences, 2009.
- Z. Lin, M. Chen, Y. Ma. The augmented Lagrange multiplier method for exact recovery of corrupted low-rank matrices. 2010.