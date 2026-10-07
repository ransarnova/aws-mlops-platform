# AWS Production MLOps Platform

A hands-on project for designing and implementing a production-oriented MLOps platform on AWS.

The project evolves a simple ML workload from local model development into a cloud-native platform covering managed training, experiment tracking, model registration, deployment, CI/CD, Infrastructure as Code, observability, security, governance, and Kubernetes-based model serving.

The ML model is intentionally simple. The primary focus is **AI/ML infrastructure and platform engineering**.

## Architecture

The target ML lifecycle is:

```text
                    ┌─────────────────┐
                    │   Data / S3     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Training  │
                    │   SageMaker     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Evaluation    │
                    │  Quality Gate   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Model Registry  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Deployment    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Observability   │
                    │ & Governance    │
                    └─────────────────┘
```

The platform is being implemented incrementally so that each architectural layer can be validated before adding additional complexity.

## Current Status

| Phase | Capability | Status |
|---|---|---|
| V1 | Local ML lifecycle | Completed |
| V2 | AWS managed training | In Progress |
| V3 | Experiment tracking | Planned |
| V4 | Model registry | Planned |
| V5 | Managed inference | Planned |
| V6 | MLOps pipeline | Planned |
| V7 | CI/CD + Infrastructure as Code | Planned |
| V8 | EKS model serving | Planned |
| V9 | AI/ML observability | Planned |
| V10 | Production hardening | Planned |

---

## V1 — Local ML Lifecycle

The first phase establishes the basic ML lifecycle locally before moving execution to AWS.

### Dataset

The project uses a synthetic lending dataset containing 1,000 records.

Features:

```text
income
debt
credit_score
missed_payments
```

Target:

```text
defaulted
```

The dataset is synthetic and exists only to support MLOps and infrastructure experimentation.

### Training

The model uses a scikit-learn pipeline:

```text
Dataset
   ↓
Train/Test Split
   ↓
StandardScaler
   ↓
LogisticRegression
   ↓
Model Artifact
```

The dataset is split into:

```text
80% training
20% testing
```

with a fixed random seed for reproducibility.

Run:

```bash
python src/train.py
```

The trained model is serialized using `joblib`.

### Independent Evaluation

Evaluation is separated from model training:

```bash
python src/evaluate.py
```

The evaluation stage calculates:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

A simple quality gate determines whether the model satisfies the minimum evaluation criteria.

```text
Training
   ↓
Evaluation
   ↓
Quality Gate
  /       \
PASS      FAIL
 │          │
 ▼          ▼
Promote    Stop
```

The evaluation process returns a non-zero exit code when the model fails the quality gate, allowing this control to be integrated into future automated pipelines.

### Model Artifacts and Metadata

Local artifacts are versioned:

```text
models/
└── v1/
    ├── loan_model.joblib
    ├── metadata.json
    └── evaluation.json
```

They represent:

```text
loan_model.joblib  → trained model
metadata.json      → training configuration and lineage
evaluation.json    → model evaluation results
```

The `models/` directory is excluded from Git. Production model artifacts will later be managed using AWS storage and model-registry capabilities.

---

## V2 — AWS Managed Training

The second phase moves model training from the local development environment to Amazon SageMaker.

### Architecture

```text
                    Local Development
                           │
                           │ training request
                           ▼
                    Amazon SageMaker
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
       Training Code   Training Data   IAM Role
                             │
                             ▼
                         Amazon S3
                             │
                             ▼
                   SageMaker Training Job
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              Train Model       Generate Metadata
                    │                 │
                    └────────┬────────┘
                             ▼
                       /opt/ml/model
                             │
                             ▼
                       model.tar.gz
                             │
                             ▼
                         Amazon S3
```

### S3 Data and Artifact Layer

Amazon S3 provides durable storage for training input and generated model artifacts.

```text
s3://<mlops-bucket>/
│
├── data/
│   └── training/
│       └── loans.csv
│
└── artifacts/
```

The bucket has:

- public access blocked
- versioning enabled
- separate prefixes for training data and generated artifacts

### IAM

A dedicated SageMaker execution role provides the training workload with access to the resources it requires.

```text
Developer / CI
      │
      │ Create Training Job
      ▼
Amazon SageMaker
      │
      │ Assume Execution Role
      ▼
SageMaker Execution Role
      │
      ├── Read training data
      │
      └── Write model artifacts
      ▼
Amazon S3
```

The role follows least-privilege access for the current workload.

### Managed Training

Managed training uses the SageMaker Python SDK and SageMaker `ModelTrainer`.

The training container receives input data through the SageMaker training channel:

```text
Amazon S3
    │
    ▼
SM_CHANNEL_TRAINING
    │
    ▼
/opt/ml/input/data/training/
    │
    ▼
train.py
```

Model output is written to:

```text
SM_MODEL_DIR
    │
    ▼
/opt/ml/model/
    │
    ├── loan_model.joblib
    └── metadata.json
    │
    ▼
model.tar.gz
    │
    ▼
Amazon S3
```

This separates ephemeral training compute from durable training data and model artifacts.

### Training Configuration

The initial managed-training configuration uses:

```text
Instance count:   1
Instance type:    ml.m5.large
Storage:          10 GB
Maximum runtime:  600 seconds
GPU:              No
```

The workload currently uses CPU-based training because the model does not require GPU acceleration.

### V2 Status

Implemented:

- S3 training-data layer
- S3 versioning and public-access protection
- dedicated SageMaker execution role
- least-privilege S3 access
- SageMaker-compatible training script
- SageMaker SDK v3 integration
- managed-training launcher
- bounded training runtime

Pending:

- training-instance service quota approval
- successful SageMaker training job
- training-log validation
- generated model artifact validation
- comparison of local and managed training lifecycle

---

## Security

Security controls currently include:

- dedicated SageMaker execution role
- least-privilege S3 permissions
- S3 public access blocked
- AWS configuration supplied at runtime
- no AWS credentials stored in source code
- generated datasets excluded from Git
- generated model artifacts excluded from Git
- secret scanning before public commits

Additional security controls will be introduced as deployment and platform capabilities are added.

---

## Cost Optimization

The platform is designed to remain cost-conscious while still demonstrating production architecture patterns.

Current design principles include:

- CPU training before GPU training
- ephemeral training compute
- single-instance training for the current workload
- bounded training runtime
- no persistent inference endpoint during early development
- no continuously running EKS cluster during early phases
- S3 for durable artifacts
- infrastructure introduced only when required by the architecture

Cost, scalability, and operational complexity will be evaluated together as the platform evolves.

---

## Repository Structure

```text
aws-mlops-platform/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
└── sagemaker/
    ├── train.py
    └── launch_training.py
```

Generated local resources are excluded from Git:

```text
data/
models/
.venv/
```

---

## Local Setup

### Requirements

- Python 3.10+
- AWS CLI
- AWS credentials configured for AWS operations

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Current direct dependencies:

```text
pandas==2.3.3
scikit-learn==1.7.2
joblib==1.6.0
sagemaker==3.23.0
```

### Run Locally

Train:

```bash
python src/train.py
```

Evaluate:

```bash
python src/evaluate.py
```

### AWS Runtime Configuration

The project currently targets:

```text
ap-south-1
```

Account-specific configuration is supplied at runtime:

```bash
export MLOPS_BUCKET="<your-mlops-bucket>"
export SAGEMAKER_ROLE_ARN="<your-sagemaker-execution-role-arn>"
```

Credentials and account-specific values should not be committed to the repository.

---

# Roadmap

## V3 — Experiment Tracking

Add systematic tracking of:

- training runs
- hyperparameters
- evaluation metrics
- training metadata
- model lineage

Target:

```text
Training Runs
      ↓
Experiment Tracking
      ↓
Compare Results
      ↓
Select Candidate
```

---

## V4 — Model Registry

Introduce controlled model versioning and promotion.

```text
Train
  ↓
Evaluate
  ↓
Quality Gate
  ↓
Register
  ↓
Approve
  ↓
Deploy
```

Key capabilities:

- model packages
- model versions
- approval states
- lineage
- controlled promotion

---

## V5 — Managed Inference

Deploy approved models using the appropriate SageMaker inference architecture.

Evaluate:

- real-time inference
- serverless inference
- batch inference
- scaling
- latency
- cost
- endpoint security

The deployment model will be selected based on workload requirements.

---

## V6 — MLOps Pipeline

Automate the end-to-end ML lifecycle:

```text
Data
 ↓
Preprocessing
 ↓
Training
 ↓
Evaluation
 ↓
Quality Gate
 ↓
Model Registry
 ↓
Approval
 ↓
Deployment
```

Models that fail evaluation criteria must not progress to promotion or deployment.

---

## V7 — CI/CD and Infrastructure as Code

Automate platform infrastructure and software delivery.

Planned capabilities:

- Terraform and/or CloudFormation
- IAM automation
- S3 infrastructure
- SageMaker infrastructure
- automated tests
- linting
- secret scanning
- IaC validation
- controlled deployments
- environment separation

Target:

```text
Developer
    ↓
Git
    ↓
CI
 ┌──┴─────────────┐
 │ Tests          │
 │ Lint           │
 │ Secret Scan    │
 │ IaC Validation │
 └──┬─────────────┘
    ↓
MLOps Pipeline
```

---

## V8 — Kubernetes / EKS Model Serving

Explore Kubernetes-based model serving and compare it with SageMaker-managed inference.

```text
Model Artifact
      ↓
Inference Container
      ↓
Amazon ECR
      ↓
Amazon EKS
      ↓
Model Serving API
      ↓
Autoscaling
```

Topics:

- containerized inference
- Kubernetes deployments
- workload identity
- ingress
- autoscaling
- rolling deployments
- model-version rollout

This phase will compare SageMaker and EKS across operational complexity, scalability, security, flexibility, and cost.

---

## V9 — AI/ML Observability

Add observability across infrastructure and model behavior.

### Infrastructure

- availability
- latency
- throughput
- errors
- CPU
- memory

### ML

- prediction distribution
- data drift
- model drift
- model quality
- inference performance

```text
Infrastructure Observability
            +
     ML Observability
            ↓
Production ML Observability
```

---

## V10 — Production Hardening

Extend the platform with production controls covering:

- network isolation
- private connectivity
- encryption
- least-privilege IAM
- secrets management
- artifact security
- audit logging
- model governance
- resilience
- disaster recovery
- quotas and capacity planning
- cost controls
- multi-environment deployment
- operational readiness

The final objective is a secure, observable, repeatable, and cost-conscious production MLOps platform.

---

## Learning Objective

This repository approaches Machine Learning from an **AI/ML infrastructure and platform engineering perspective**.

The focus is not on creating the most sophisticated ML model. The focus is on designing the platform required to take a model through:

```text
Development
    ↓
Training
    ↓
Evaluation
    ↓
Registration
    ↓
Deployment
    ↓
Monitoring
    ↓
Governance
```

reliably, securely, reproducibly, and cost-effectively.

---

## Disclaimer

This project is for technical learning and infrastructure experimentation.

The dataset is synthetically generated. The model logic, evaluation thresholds, and predictions are illustrative only and must not be used for real lending, credit, financial, eligibility, or other consequential decision-making.