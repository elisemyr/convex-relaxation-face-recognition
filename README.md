# Face access control with convex relaxation

## The problem

A company wants its employees to open the doors with their face. The system has to work when people wear a mask or glasses, when the light changes, and when each employee gave only a few photos. It must also refuse visitors who were never enrolled. A basic face matcher would just give them the name of the closest employee.

## My approach

I worked as if I were a data science consultant for this company. Every method I use comes from a convex relaxation we saw in the course:

| Part of the system | Hard problem | Convex version I solve | Lecture |
|---|---|---|---|
| Sparse representation classification, SRC | min ‖x‖₀ | min ‖x‖₁, Lasso or Basis Pursuit | Compressed sensing |
| Occlusion model y = Φx + e | e has few non-zero pixels | min ‖x‖₁ + ‖e‖₁ | Compressed sensing |
| Cleaning the gallery with Robust PCA | min rank(L) + ‖S‖₀ | min ‖L‖* + λ‖S‖₁ | Matrix completion |
| Ridge version of SRC, called CRC | no relaxation, used as a check | ℓ2 penalty instead of ℓ1 | Lasso and ridge |
| Random projection features | how many measurements are enough | RIP with m ∼ s log(N/s) | Compressed sensing |

SRC writes a new face as a mix of all the gallery photos and looks for a mix that uses very few of them. If the face belongs to an employee, most of the weight should land on that person's photos.

To decide if the door opens, I use the Sparsity Concentration Index, or SCI, from Wright et al. 2009. It measures how much of the weight sits on one single person. A high SCI means a confident match. A low SCI means the face is probably a stranger. The system can accept the person, reject them, or ask for the badge when too much of the face is hidden.

I compare SRC with three baselines: Eigenfaces with a linear SVM, Eigenfaces with the nearest neighbour, and a pretrained FaceNet model.

## Data

I use the cropped version of Extended Yale B. It has 38 people with about 64 photos each, 192 × 168 pixels, taken under many lighting conditions.

The download link on the official UCSD page is broken. So I use a copy of the same cropped images on Kaggle, converted to PNG:

- Kaggle copy, `ExtYaleBCroppedPNG`: https://www.kaggle.com/datasets/tbourton/extyalebcroppedpng
- Mirror of the original `CroppedYale.zip` : https://academictorrents.com/details/aad8bf8e6ee5d8a3bf46c7ab5adfacdd8ad36247

Unzip the dataset anywhere inside `data/`. The loader looks in every subfolder and reads the person and the light angles from the file names, so both the .png and the .pgm versions work. The data itself is not in this repository.

Dataset references: Georghiades, Belhumeur and Kriegman, PAMI 2001. Lee, Ho and Kriegman, PAMI 2005.

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate          
pip install -r requirements.txt

python3 -m pytest -q                              
python3 experiments/run_all.py --quick            #small run, a few minutes
python3 experiments/run_all.py                    # full run, see the note on timing below
python3 experiments/plots.py                      # all figures go to results/yaleb/figures
```

For the FaceNet baseline, install the extra packages and add `--deep`:

```bash
pip install -r requirements-deep.txt
pip install facenet-pytorch --no-deps
python3 experiments/run_all.py --deep
```

`facenet-pytorch` asks for old versions of numpy and torch, so I install it without its own dependencies

## Project structure

```
config.py                all settings and seeds
src/data.py              Yale B loader and synthetic faces
src/splits.py            employees, validation visitors, test visitors
src/occlusion.py         block, mask, glasses and pixel corruption
src/features.py          downsampling, PCA, random projections
src/solvers.py           FISTA, extended l1, ridge, cvxpy references
src/src_classifier.py    SRC, CRC, SCI score and the door decision
src/rpca.py              Robust PCA by inexact ALM
src/baselines.py         Eigenfaces + SVM, Eigenfaces + NN, FaceNet
src/pipeline.py          features + optional RPCA + classifier
src/evaluation.py        accuracy, FAR, FRR, ROC, choice of tau
experiments/run_all.py   experiments E1 to E6, results saved as CSV
experiments/plots.py     figures from the CSV files
tests/test_core.py       unit tests
```

## Experiments

Two error rates come back often. The false accept rate, FAR, is the share of visitors who get in. The false reject rate, FRR, is the share of employees who are refused or given the wrong name.

- E1: accuracy on clean faces when the feature size d changes, for pixels, PCA and random projections
- E2: accuracy when part of the face is hidden, for 4 kinds of occlusion
- E3: accuracy when each employee has more or fewer gallery photos
- E4: what changes when I clean the gallery with Robust PCA
- E5: the door decision. I choose the threshold τ so that FAR stays under 1% on one group of visitors, then I measure FAR and FRR on a different group
- E6: time for one decision, measured during E5
