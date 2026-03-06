# 🔮 cosmoford - Shake the Cosmic 8-Ball

*Is the S8 tension real?: "Outlook uncertain"*

Welcome to **cosmoford**, your magic 8-ball for predicting cosmological parameters from weak gravitational lensing data. Just like its mystical namesake, this package peers into the cosmic future—except instead of vague prophecies, it delivers uncertainty-quantified predictions of Ω<sub>m</sub> and S<sub>8</sub>.

This repository contains utilities and models for the [**FAIR Universe - Weak Lensing ML Uncertainty Challenge**](https://www.codabench.org/competitions/8934/) at NeurIPS 2025. The challenge explores uncertainty-aware and out-of-distribution detection AI techniques for weak gravitational lensing cosmology.

## 🚀 Installation

Install the package in editable mode:

```bash
pip install -e .
```

## 🎯 Quick Start

Training a model from a Lightning configuration can be achieved like so:
```bash
trainer fit -c configs/finetune_from_pretrain_nopatch.yaml
```

## 📤 Preparing Submissions

After training a model, you can prepare it for submission to the challenge:

### 1. Prepare the Submission

Use the `prepare_for_submission` command with your W&B run ID:

```bash
prepare_for_submission \
  --run_id <wandb_run_id> \
  --name <short_name> \
  --description "<description>"
```

**Arguments:**
- `--run_id`: Your W&B run ID (e.g., `hulew8h2`)
- `--name`: Short memorable name (e.g., `baseline`, `dropout-v1`, `eff-b2`)
- `--description`: Brief description (e.g., `"Baseline EfficientNet B0 model"`)
- `--notes`: (Optional) Additional notes

**Example:**
```bash
prepare_for_submission \
  --run_id abc123 \
  --name baseline \
  --description "Baseline EfficientNet B0 model"
```

This will:
- ✅ Download your model from W&B
- ✅ Evaluate it on the validation set
- ✅ Generate predictions on the test set
- ✅ Create a submission ZIP file
- ✅ Upload the ZIP to W&B
- ✅ Update [SUBMISSIONS.md](SUBMISSIONS.md) with your entry

### 2. Request Submission to Challenge

Once your submission is prepared and appears in [SUBMISSIONS.md](SUBMISSIONS.md):

1. **Check the submission table** to verify your entry
2. **Ping @EiffL** on GitHub or Slack with:
   - The submission name (e.g., `EiffL_abc123_baseline`)
   - A link to the [SUBMISSIONS.md](SUBMISSIONS.md) file

## 🧑‍🤝‍🧑 Transatlantic Dream Team

- [@AndreasTersenov](https://github.com/AndreasTersenov)
- [@ASKabalan](https://github.com/ASKabalan) (**co-lead**)
- [@b-remy](https://github.com/b-remy)
- [@EiffL](https://github.com/EiffL) (**submitter**)
- [@EnceladeCandy](https://github.com/EnceladeCandy)
- [@JuliaLinhart](https://github.com/JuliaLinhart)
- [@Justinezgh](https://github.com/Justinezgh) (**co-lead**)
- [@LaurencePeanuts](https://github.com/LaurencePeanuts)
- [@SammyS15](https://github.com/SammyS15)
- [@sachaguer](https://github.com/sachaguer)

## 📝 License

See [LICENSE](LICENSE) file for details.

## 🔮 *"Ask again later... after training for 100 epochs!"*

---

*Disclaimer: Unlike a real magic 8-ball, cosmoford's predictions are based on rigorous machine learning and statistical inference. Results may vary based on your model architecture, training data, and cosmic variance.*
