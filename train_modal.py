# train_modal.py
"""Modal entrypoint for remote GPU training."""
from pathlib import Path
from typing import Optional

import modal

volume = modal.Volume.from_name("cosmorford-training", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch>=2.4",
        "torchvision>=0.19",
        "lightning>=2.4",
        "datasets",
        "numpy",
        "wandb",
        "omegaconf",
        "pyyaml",
        "jsonargparse[signatures,omegaconf]>=4.27.7",
    )
    .add_local_dir("cosmorford", "/root/cosmorford", copy=True)
    .add_local_dir("configs", "/root/configs", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .run_commands("cd /root && pip install -e .")
)

app = modal.App("cosmorford", image=image)

VOLUME_PATH = Path("/experiments")
CHECKPOINTS_PATH = VOLUME_PATH / "checkpoints"


@app.function(
    volumes={VOLUME_PATH: volume},
    gpu="a10g",
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config_path: str, experiment_name: Optional[str] = None):
    _train_impl(config_path, experiment_name)


@app.function(
    volumes={VOLUME_PATH: volume},
    gpu="a100",
    timeout=86400,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_a100(config_path: str, experiment_name: Optional[str] = None):
    _train_impl(config_path, experiment_name)


def _train_impl(config_path: str, experiment_name: Optional[str] = None):
    import subprocess
    import yaml

    checkpoint_dir = CHECKPOINTS_PATH / (experiment_name or "default")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = checkpoint_dir / "last.ckpt"

    # Write a runtime overlay config that sets checkpoint dir on the volume
    overlay = {
        "trainer": {
            "default_root_dir": str(checkpoint_dir),
            "callbacks": [
                {
                    "class_path": "LearningRateMonitor",
                    "init_args": {"logging_interval": "step"},
                },
                {
                    "class_path": "ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(checkpoint_dir),
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 3,
                        "save_last": True,
                    },
                },
            ],
        }
    }
    overlay_path = Path("/tmp/runtime_config.yaml")
    overlay_path.write_text(yaml.dump(overlay))

    cmd = [
        "trainer", "fit",
        f"--config=/root/{config_path}",
        f"--config={overlay_path}",
    ]

    if last_ckpt.exists():
        print(f"Resuming from checkpoint: {last_ckpt}")
        cmd.append(f"--ckpt_path={last_ckpt}")
    else:
        print("Starting training from scratch")

    subprocess.run(cmd, check=True, cwd="/root")
    volume.commit()


@app.local_entrypoint()
def main(
    config: str = "configs/default.yaml",
    name: Optional[str] = None,
    gpu: str = "a10g",
    parallel_configs: Optional[str] = None,
):
    if parallel_configs:
        # Launch multiple experiments in parallel within a single app
        # Format: "config1:name1,config2:name2,..."
        pairs = [p.strip().split(":") for p in parallel_configs.split(",")]
        train_fn = train_a100 if gpu == "a100" else train
        handles = []
        for cfg, exp_name in pairs:
            print(f"Spawning {exp_name} with {cfg} on GPU: {gpu}")
            handles.append(train_fn.spawn(cfg, exp_name))
        for h in handles:
            h.get()
        print("All parallel experiments completed.")
    else:
        print(f"Starting training with config: {config} on GPU: {gpu}")
        if gpu == "a100":
            train_a100.spawn(config, name).get()
        else:
            train.spawn(config, name).get()
