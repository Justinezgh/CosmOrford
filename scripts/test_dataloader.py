import torch
from cosmoford.dataset import ChallengeDataModule # Replace with your filename

def quick_test_all_modes():
    # Only test the modes you actually have access to right now
    modes_to_test = ["lognormal", "ot_emulated", "gowerstreet-train"]
    
    results = {}

    for mode in modes_to_test:
        print(f"\n--- Testing Mode: {mode} ---")
        try:
            # Initialize with small batch for speed
            dm = ChallengeDataModule(batch_size=2, num_workers=0, dataset_mode=mode)
            dm.setup()
            
            # Grab one batch
            loader = dm.train_dataloader()
            kappa, theta = next(iter(loader))
            
            results[mode] = f"✅ Success! Kappa: {kappa.shape}, Theta: {theta.shape}"
        except Exception as e:
            results[mode] = f"❌ Failed: {str(e)[:100]}..."

    print("\n" + "="*30)
    print("DATASET IMPORT SUMMARY")
    print("="*30)
    for mode, status in results.items():
        print(f"{mode:15}: {status}")

if __name__ == "__main__":
    quick_test_all_modes()