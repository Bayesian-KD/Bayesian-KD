# Bayesian-KD: SGD-Based Knowledge Distillation with Bayesian Teachers: Theory and Guidelines

This repository includes the source code used in the paper "SGD-Based Knowledge Distillation with Bayesian Teachers: Theory and Guidelines".

# Abstract

Knowledge Distillation (KD) is a central paradigm for transferring knowledge from a large teacher network to a typically smaller student model, often by leveraging soft probabilistic outputs. While KD has shown strong empirical success in numerous applications, its theoretical underpinnings remain only partially understood. In this work, we adopt a Bayesian perspective on KD to rigorously analyze the convergence behavior of students trained with Stochastic Gradient Descent (SGD). We study two regimes: (1) when the teacher provides the exact Bayes Class Probabilities (BCPs); and (2) supervision with noisy approximations of the BCPs. Our analysis shows that learning from BCPs yields variance reduction and removes neighborhood terms in the convergence bounds compared to one-hot supervision. We further characterize how the level of noise affects generalization and accuracy. Motivated by these insights, we advocate the use of Bayesian deep learning models, which typically provide improved estimates of the BCPs, as teachers in KD. Consistent with our analysis, we experimentally demonstrate that students distilled from Bayesian teachers not only achieve higher accuracies (up to +4.27\%), but also exhibit more stable convergence (up to 30\% less noise), compared to students distilled from deterministic teachers.

# Overview

This repository consists of following Python scripts:

### Teacher training
- **`scripts/train_teacher_deterministic_VI.py`** – Train a deterministic teacher, or a Bayesian teacher with Variational Inference (VI), or an MSE teacher.  
- **`scripts/train_LA.py`** – Train a Bayesian teacher using a Laplace Approximation (post-hoc applied to a pretrained deterministic teacher).  
- **`scripts/train_MCMI.py`** – Train a MCMI teacher by fine-tuning a pretrained deterministic teacher with an MCMI loss term.  

### Student training
- **`scripts/train_student.py`** – Train a deterministic student from either a deterministic teacher, a Bayesian VI teacher, a Laplace teacher, an MCMI teacher, or an MSE teacher.  
- **`scripts/train_fewshot_exp_students.py`** – Train students as in `train_student.py`, but using CIFAR-100 subsets for few-shot classification.  
- **`scripts/train_bnn_samples_exp_students.py`** – Train students from Bayesian VI teachers while varying the number of Monte Carlo samples for inference of the teacher model predictions.  

### Data and utilities
- **`scripts/create_cifar_subsets.py`** – Create CIFAR-100 subsets for few-shot experiments.  
- **`scripts/test_teacher.py`** – Evaluate trained teacher performance.
- **`synthetic_experiment/Synthetic_experiment.ipynb`** – Synthetic experiment implementation.

## Credits and External Code

Parts of this repository build upon publicly available implementations:

- **MCMI teachers** – Adapted from [iclr2024mcmi/ICLRMCMI](https://github.com/iclr2024mcmi/ICLRMCMI). Specifically, `centroids.py` was pulled from their repository.
- **MSE teachers** – Adapted from [ECCV2024MSE](https://github.com/ECCV2024MSE).
- **Bayesian teachers with Variational Inference (VI)** – Adapted from [microsoft/bayesianize](https://github.com/microsoft/bayesianize). All files under the `bnn/` folder were pulled from their repository. The configuration file under `configs/` is also the default configuration file they provide for CIFAR-100.
- **Bayesian teachers with Laplace Approximation** – Adapted from [aleximmer/Laplace](https://github.com/aleximmer/Laplace). 

## Citation

If you find this repository useful in your research, please consider citing the original works our code is adapted from:

```bibtex
@article{ye2024bayes,
  title   = {Bayes Conditional Distribution Estimation for Knowledge Distillation Based on Conditional Mutual Information},
  author  = {Ye, Linfeng and Hamidi, Shayan Mohajer and Tan, Renhao and Yang, En-Hui},
  journal = {arXiv preprint arXiv:2401.08732},
  year    = {2024}
}

@inproceedings{hamidi2024train,
  title        = {How to train the teacher model for effective knowledge distillation},
  author       = {Hamidi, Shayan Mohajer and Deng, Xizhen and Tan, Renhao and Ye, Linfeng and Salamah, Ahmed Hussein},
  booktitle    = {European Conference on Computer Vision},
  pages        = {1--18},
  year         = {2024},
  organization = {Springer}
}

@inproceedings{laplace2021,
  title     = {Laplace Redux--Effortless {B}ayesian Deep Learning},
  author    = {Erik Daxberger and Agustinus Kristiadi and Alexander Immer
               and Runa Eschenhagen and Matthias Bauer and Philipp Hennig},
  booktitle = {NeurIPS},
  year      = {2021}
}
```

## Teacher Training

Trained models are saved under `save/teachers/{kind}/{architecture}/{run_name}`, where `{kind}` ∈ {Deterministic, Bayesian, MSE, Laplace, MCMI} and `{architecture}` ∈ {resnet18, resnet34, resnet50, resnet101, resnet152, wrn_40_2, vgg13, …}.

### Common parameters
The following arguments can be used in the training scripts, see scripts for additional arguments:
- `--epochs` – number of training epochs  
- `--batch_size` – training batch size  
- `--num_workers` – number of dataloader workers  
- `--learning_rate` – initial learning rate  
- `--momentum` – momentum 
- `--lr_decay_epochs` – epoch indices where the learning rate is decayed  
- `--lr_decay_rate` – learning rate decay factor
- `-t, --trial` – trial index 
- `--progress_bar` – enable tqdm progress bar  

### Deterministic / Bayesian VI / MSE teachers
**Script:** `train_teacher_deterministic_VI.py`  
Additional arguments:  
- `--bayesianize` – train a Bayesian teacher using Variational Inference  
- `--mse` – train an MSE teacher  
- `--bnn_ml_epochs` – epochs of ML training before KL 
- `--bnn_annealing_epochs` – epochs for gradual KL annealing  
- `--bnn_test_samples` – number of Monte Carlo test samples (for validation)

**Examples:**
```bash
# Deterministic teacher
python scripts/train_teacher_deterministic_VI.py \
    --epochs 200 --learning_rate 1e-3 \
    --lr_decay_epochs 100,150 --model resnet50 --trial 1

# Bayesian VI teacher
python scripts/train_teacher_deterministic_VI.py \
    --epochs 200 --learning_rate 1e-3 \
    --lr_decay_epochs 100 --model resnet101 \
    --bayesianize --trial 1

# MSE teacher
python scripts/train_teacher_deterministic_VI.py \
    --epochs 200 --learning_rate 1e-3 \
    --lr_decay_epochs 100 --model resnet50 --mse --trial 1
```
### Laplace Approximation teachers
**Script:** `train_teacher_LA.py`  
Requires a pretrained deterministic teacher provided with `--teacher_path`.  

Arguments:  
- `--teacher_path` – path to a pretrained deterministic teacher (required)   
- `--batch_size` – batch size
- `--num_workers` – number of workers  
- `--subs_weights` – subset of weights 
- `--hess_struct` – Hessian structure  
- `--la_method` – method for tuning prior precision  
- `--pred_type` – prediction type
- `--link_approx` – link approximation 

**Example:**
```bash
python scripts/train_teacher_LA.py \
    --teacher_path ../save/teachers/Deterministic/wrn_40_2/..._trial_1 \
    --hess_struct kron --la_method marglik
```

### MCMI teachers
**Script:** `train_teacher_MCMI.py`  
Requires a pretrained deterministic teacher provided with `--teacher_path`.  

Arguments:  
- `--teacher_path` – path to a pretrained deterministic teacher (required)
- `--batch_size` – batch size  
- `--num_workers` – number of workers   
- `--epochs` – number of training epochs  
- `--learning_rate` – learning rate 
- `--momentum` – momentum 
- `--mcmiparam` – MCMI hyperparameter 
- `--CentroidSampleSize` – centroid sample size

**Example:**
```bash
python scripts/train_teacher_MCMI.py \
    --teacher_path ../save/teachers/Deterministic/wrn_40_2/..._trial_1 \
    --epochs 20 --mcmiparam 0.2
```

## Student Training

**Script:** `train_student.py`  
Students are trained from pretrained teachers of different kinds (Deterministic, Bayesian VI, MSE, Laplace, MCMI). Trained students are saved under: `save/students/T:{teacher_arch}_S:{student_arch}/{teacher_type}/{lambda_temp_settings}/{student_name}`.

The parameters used for training are:

- `--epochs` – number of training epochs
- `--batch_size` – training batch size
- `--num_workers` – number of dataloader workers 
- `--learning_rate` – initial learning rate
- `--momentum` – momentum 
- `--lr_decay_epochs` – epoch indices where the learning rate is decayed  
- `--lr_decay_rate` – learning rate decay factor
- `--model` – student architecture
- `--alpha` – Distillation parameter weighing between CE and KD loss (default: 1.0)  
- `--kd_T` – teacher temperature (default: 1)  
- `--kd_S` – student temperature (default: 1)  
- `--num_chunk` – batch splitting for Laplace teacher inference 
- `--teacher_path` – path to a pretrained teacher (required)  
- `--bnn_samples` – number of MC samples for Bayesian VI teachers  

### Examples

**Training a student from a deterministic teacher**
```bash
python scripts/train_student.py \
    --teacher_path ../save/teachers/Deterministic/resnet50/..._trial_1 \
    --model resnet18 --epochs 200 --learning_rate 1e-3 \
    --alpha 0.5 --kd_T 4 --kd_S 1
```
**Training a student from a Bayesian VI teacher**
```bash
python scripts/train_student.py \
    --teacher_path ../save/teachers/Bayesian/resnet50/..._trial_1 \
    --model resnet18 --epochs 200 --learning_rate 1e-3 \
    --alpha 0.7 --kd_T 4 --kd_S 1 --bnn_samples 1
```

**Training a student from a Bayesian LA teacher**
```bash
python scripts/train_student.py \
    --teacher_path ../save/teachers/Laplace/resnet50/..._trial_1_last_layer_kron_marglik_glm_probit \
    --model resnet18 --epochs 200 --learning_rate 1e-3 \
    --alpha 1.0 --kd_T 2 --kd_S 2 --num_chunk 20
```

**Training a student from an MCMI teacher**
```bash
python scripts/train_student.py \
    --teacher_path ../save/teachers/MCMI/wrn_40_2/..._trial_1_0.15_20_0.0002_100_0.9 \
    --model wrn_40_1 --epochs 200 --learning_rate 1e-3 \
    --alpha 0.8 --kd_T 4 --kd_S 1
```

### Few-shot student training
**Script:** `train_fewshot_exp_students.py`  
This script trains students from different teachers using **CIFAR-100 few-shot subsets** (e.g., 5%, 10%, … 50% of training data per class). Subsets are created with `create_cifar_subsets.py` and saved in `../data/fewshot`.  

Note that the subset needs to be initially created using `create_cifar_subsets.py`. \
The following arguments are available:
- `--teacher_path` – path to a pretrained teacher (required)  
- `--fewshot_percent` – percentage per class for few-shot data
- `--fewshot_root` – path to few-shot data (default: `../data/fewshot`)  
- `--students` – comma-separated student architectures (e.g., `"resnet50,resnet34,resnet18"`)  
- `--grid` – semicolon-separated list of `T:S:A` triples for distillation hyperparameters (teacher temperature, student temperature, distillation weighting parameter)  
- `--bnn_samples` – number of MC samples for Bayesian VI teacher.
- `--epochs`, `--batch_size`, `--learning_rate`, `--lr_decay_epochs`, etc. 

**Example:**
```bash
python scripts/train_fewshot_exp_students.py \
    --teacher_path ../save/teachers/Bayesian/resnet50/..._trial_1 \
    --fewshot_percent 10 \
    --students "resnet18,resnet34" \
    --grid "4:1:0.5;2:1:0.7" \
    --epochs 200 --learning_rate 1e-3
```

### Multiple Monte Carlo samples for student training
**Script:** `train_bnn_samples_exp_students.py`  
This script trains students from a Bayesian VI teacher while **varying the number of Monte Carlo samples** (S = 1..N) used to estimate teacher predictions.  

The arguments are 
- `--teacher_path` – path to a pretrained Bayesian teacher (required; only works with Bayesian teachers)  
- `--students` – comma-separated student architectures
- `--bnn_samples` – maximum number of samples to use; students are trained with all S ∈ {1, …, `bnn_samples`}   
- `--epochs`, `--batch_size`, `--learning_rate`, etc.   

**Example:**
```bash
python scripts/train_bnn_samples_exp_students.py \
    --teacher_path ../save/teachers/Bayesian/resnet50/..._trial_1 \
    --students resnet18 \
    --bnn_samples 12 --epochs 200 --learning_rate 1e-3
```
