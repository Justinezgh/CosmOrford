#!/usr/bin/env python
"""
Prepare submissions for the NeurIPS Weak Lensing Challenge.

This script:
1. Downloads a trained model from W&B
2. Evaluates it on the test set to compute the validation score
3. Generates predictions on the challenge test set
4. Creates a submission ZIP file
5. Uploads the ZIP to W&B as an artifact
6. Updates SUBMISSIONS.md and submissions_metadata.json
"""

import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import wandb
from tqdm import tqdm

from cosmoford import THETA_MEAN, THETA_STD, NOISE_STD, SURVEY_MASK
from cosmoford.models import RegressionModel
from cosmoford.models_nopatch import RegressionModelNoPatch
from cosmoford.dataset import ChallengeDataModule
from cosmoford.utils import Score, Utility


def download_model(run_id: str, project: str = "neurips-wl-challenge", entity: str = "cosmostat") -> tuple[RegressionModel, wandb.sdk.wandb_run.Run]:
    """Download model from W&B and return the loaded model."""
    print(f"Initializing W&B and downloading model for run {run_id}...")
    run = wandb.init(project=project, entity=entity, job_type="submission")

    # Download the best model from the specified run
    artifact = run.use_artifact(f'{entity}/{project}/model-{run_id}:v0', type='model')
    artifact_dir = artifact.download()

    # Load the checkpoint
    checkpoint_path = f"{artifact_dir}/model.ckpt"
    model = RegressionModelNoPatch.load_from_checkpoint(checkpoint_path).eval()

    print(f"Model loaded from {checkpoint_path}")
    return model, run


def evaluate_on_validation_set(model: RegressionModelNoPatch, device: str = 'cuda') -> Dict[str, Any]:
    """Evaluate model on validation set and return scores."""
    print("Evaluating model on validation set...")

    # Load the dataset
    dset = ChallengeDataModule(batch_size=128, num_workers=8)
    dset.setup()
    dloader = dset.test_dataloader()

    mask = np.concatenate([SURVEY_MASK[:, :88], SURVEY_MASK[620:1030, 88:]])

    y_val = []
    mean_val = []
    errorbar_val = []

    model = model.to(device)

    with torch.no_grad():
        for batch in tqdm(dloader, desc="Validation batches"):
            x = batch[0].to(device)
            y = batch[1].to(device)
            noise = torch.randn_like(x) * NOISE_STD
            x = x + noise
            x = x * torch.tensor(mask, device=x.device).unsqueeze(0)
            m, s = model(x)

            # Rescaling all predictions to the original scale
            m = m * torch.tensor(THETA_STD[:2], device=m.device) + torch.tensor(THETA_MEAN[:2], device=m.device)
            s = s * torch.tensor(THETA_STD[:2], device=s.device)
            y = y[:,:2] * torch.tensor(THETA_STD[:2], device=y.device) + torch.tensor(THETA_MEAN[:2], device=y.device)

            y_val.append(y.cpu().numpy())
            mean_val.append(m.cpu().numpy())
            errorbar_val.append(s.cpu().numpy())

    y_val = np.concatenate(y_val)
    mean_val = np.concatenate(mean_val)
    errorbar_val = np.concatenate(errorbar_val)

    # Compute validation score
    validation_score = Score._score_phase1(
        true_cosmo=y_val[:, :2],
        infer_cosmo=mean_val[:, :2],
        errorbar=errorbar_val[:, :2]
    )

    avg_score = np.mean(validation_score)
    avg_errorbar = np.mean(errorbar_val, 0)

    print(f"Validation score: {avg_score:.4f}")
    print(f"Average error bars: {avg_errorbar}")

    return {
        "validation_score": float(avg_score),
        "avg_errorbar": avg_errorbar.tolist(),
        "num_samples": len(y_val)
    }


def generate_test_predictions(model: RegressionModelNoPatch, device: str = 'cuda') -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions on the challenge test set."""
    print("Generating predictions on challenge test set...")

    # Download test data if not present
    test_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
    if not os.path.exists(test_file):
        print(f"Downloading test data...")
        import urllib.request
        url = f"https://storage.googleapis.com/neurips-wl/{test_file}"
        urllib.request.urlretrieve(url, test_file)
        print(f"Test data downloaded to {test_file}")

    # Load test data
    batch_size = 100
    kappa_test = np.zeros((4000, 1424, 176), dtype=np.float16)
    kappa_test[:, SURVEY_MASK] = np.load(test_file)

    mean_val = []
    errorbar_val = []

    model = model.to(device)

    with torch.no_grad():
        for i in tqdm(range(4000 // batch_size), desc="Test batches"):
            # Getting data
            x = torch.tensor(kappa_test[i*batch_size:(i+1)*batch_size], device=device, dtype=torch.float32)
            # Reshaping
            x = torch.cat([x[:, :, :88], x[:, 620:1030, 88:]], axis=1)

            # Applying model
            m, s = model(x)

            # Rescaling all predictions to the original scale
            m = m * torch.tensor(THETA_STD[:2], device=m.device) + torch.tensor(THETA_MEAN[:2], device=m.device)
            s = s * torch.tensor(THETA_STD[:2], device=s.device)

            mean_val.append(m.cpu().numpy())
            errorbar_val.append(s.cpu().numpy())

    mean_val = np.concatenate(mean_val)[:, :2]  # Only first two cosmological parameters
    errorbar_val = np.concatenate(errorbar_val)[:, :2]

    print(f"Generated {len(mean_val)} predictions")
    return mean_val, errorbar_val


def create_submission_file(run_id: str, github_user: str, mean_val: np.ndarray, errorbar_val: np.ndarray) -> str:
    """Create submission ZIP file with format: username_runid_date.zip"""
    print("Creating submission file...")

    data = {"means": mean_val.tolist(), "errorbars": errorbar_val.tolist()}
    the_date = datetime.datetime.now().strftime("%y%m%d-%H%M")
    zip_file_name = f'{github_user}_{run_id}_{the_date}.zip'

    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json",
        zip_file_name=zip_file_name,
        data=data
    )

    print(f"Submission ZIP saved at: {zip_file}")
    return zip_file


def upload_to_wandb(zip_file: str, run_id: str, run: wandb.sdk.wandb_run.Run) -> str:
    """Upload submission ZIP to W&B as an artifact."""
    print("Uploading submission to W&B...")

    artifact = wandb.Artifact(
        name=f"submission-{run_id}",
        type="submission",
        description=f"Challenge submission for run {run_id}",
        metadata={"run_id": run_id, "file": os.path.basename(zip_file)}
    )
    artifact.add_file(zip_file)
    run.log_artifact(artifact)

    # Get artifact URL
    artifact_url = f"{run.entity}/{run.project}/artifacts/submission/submission-{run_id}"
    print(f"Submission uploaded to W&B: {artifact_url}")

    return artifact_url


def get_github_username() -> str:
    """Get GitHub username from git config."""
    try:
        result = subprocess.run(
            ['git', 'config', 'user.name'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Unknown"


def load_metadata(metadata_file: str = "submissions_metadata.json") -> List[Dict[str, Any]]:
    """Load existing metadata or return empty list."""
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return []


def save_metadata(metadata: List[Dict[str, Any]], metadata_file: str = "submissions_metadata.json"):
    """Save metadata to JSON file."""
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")


def update_submissions_markdown(metadata: List[Dict[str, Any]], markdown_file: str = "SUBMISSIONS.md"):
    """Update or create SUBMISSIONS.md with submission table."""
    print(f"Updating {markdown_file}...")

    # Sort metadata by date (newest first)
    sorted_metadata = sorted(metadata, key=lambda x: x['date'], reverse=True)

    # Generate markdown content
    lines = [
        "# Challenge Submissions",
        "",
        "This table tracks all prepared submissions for the NeurIPS Weak Lensing Challenge.",
        "",
        "| Run ID | Date | User | Submission Name | Description | Score | W&B Artifact | Submitted |",
        "|--------|------|------|-----------------|-------------|-------|--------------|-----------|"
    ]

    for entry in sorted_metadata:
        run_id = entry['run_id']
        date = entry['date']
        github_user = entry.get('github_user', 'Unknown')
        submission_name = entry.get('submission_name', 'N/A')
        description = entry.get('description', 'N/A')
        score = f"{entry['validation_score']:.4f}"
        wandb_url = entry.get('wandb_artifact_url', 'N/A')

        # Format W&B URL as clickable link with full URL
        if wandb_url != 'N/A':
            # Convert artifact reference to full URL
            full_wandb_url = f"https://wandb.ai/{wandb_url}"
            wandb_link = f"[Download]({full_wandb_url})"
        else:
            wandb_link = 'N/A'

        # Format W&B run link
        run_link = f"[{run_id}](https://wandb.ai/{entry.get('entity', 'cosmostat')}/{entry.get('project', 'neurips-wl-challenge')}/runs/{run_id})"

        # Checkbox for submission status
        submitted = entry.get('submitted', False)
        checkbox = "- [x]" if submitted else "- [ ]"

        lines.append(f"| {run_link} | {date} | {github_user} | `{submission_name}` | {description} | {score} | {wandb_link} | {checkbox} |")

    lines.extend([
        "",
        "## Notes",
        "",
        "- **Run ID**: Click to view the W&B run details",
        "- **User**: GitHub username (from git config)",
        "- **Submission Name**: Format `user_runid_name` (e.g., `EiffL_abc123_baseline`)",
        "- **Description**: Short description of the submission",
        "- **Score**: Validation score computed on test set (higher is better)",
        "- **W&B Artifact**: Link to download the submission ZIP from W&B",
        "- **Submitted**: Check the box after manually submitting to the challenge platform",
        "",
        "To update the submission status, edit this file and toggle the checkbox, or modify `submissions_metadata.json`."
    ])

    with open(markdown_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Updated {markdown_file} with {len(sorted_metadata)} submissions")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare submission for NeurIPS Weak Lensing Challenge"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="W&B run ID for the trained model"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="cosmostat",
        help="W&B entity name (default: cosmostat)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="neurips-wl-challenge",
        help="W&B project name (default: neurips-wl-challenge)"
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Short memorable name for this submission (e.g., 'baseline', 'dropout-v1', 'eff-b2')"
    )
    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="Short description of this submission (e.g., 'baseline model', 'added dropout')"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for inference (default: cuda)"
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional additional notes about this submission"
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading to W&B (for testing)"
    )

    args = parser.parse_args()

    # Get GitHub username first
    github_user = get_github_username()

    # Step 1: Download model
    model, run = download_model(args.run_id, args.project, args.entity)

    # Step 2: Evaluate on validation set
    eval_results = evaluate_on_validation_set(model, args.device)

    # Step 3: Generate test predictions
    mean_val, errorbar_val = generate_test_predictions(model, args.device)

    # Step 4: Create submission file (with new naming format)
    zip_file = create_submission_file(args.run_id, github_user, mean_val, errorbar_val)

    # Step 5: Upload to W&B
    artifact_url = None
    if not args.skip_upload:
        artifact_url = upload_to_wandb(zip_file, args.run_id, run)

    # Step 6: Update metadata
    metadata = load_metadata()

    # Check if this run_id already exists
    existing_idx = None
    for idx, entry in enumerate(metadata):
        if entry['run_id'] == args.run_id:
            existing_idx = idx
            break

    # Create submission name: user_runid_name
    submission_name = f"{github_user}_{args.run_id}_{args.name}"

    new_entry = {
        "run_id": args.run_id,
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "github_user": github_user,
        "submission_name": submission_name,
        "description": args.description,
        "validation_score": eval_results['validation_score'],
        "avg_errorbar": eval_results['avg_errorbar'],
        "submission_file": os.path.basename(zip_file),
        "submission_path": zip_file,
        "wandb_artifact_url": artifact_url if artifact_url else "N/A",
        "entity": args.entity,
        "project": args.project,
        "submitted": False,
        "notes": args.notes
    }

    if existing_idx is not None:
        print(f"Updating existing entry for run {args.run_id}")
        # Preserve submission status if it exists
        new_entry['submitted'] = metadata[existing_idx].get('submitted', False)
        metadata[existing_idx] = new_entry
    else:
        metadata.append(new_entry)

    save_metadata(metadata)

    # Step 7: Update SUBMISSIONS.md
    update_submissions_markdown(metadata)

    # Finish W&B run
    if not args.skip_upload:
        run.finish()

    print("\n" + "="*60)
    print("✓ Submission preparation complete!")
    print(f"  Run ID: {args.run_id}")
    print(f"  Validation Score: {eval_results['validation_score']:.4f}")
    print(f"  Submission file: {zip_file}")
    if artifact_url:
        print(f"  W&B Artifact: {artifact_url}")
    print("="*60)


if __name__ == "__main__":
    main()
