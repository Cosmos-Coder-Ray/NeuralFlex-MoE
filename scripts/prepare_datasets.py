import argparse
from datasets import load_dataset
import os
from neuraflex_moe.utils.logging_utils import setup_logger

# A mapping of friendly names to Hugging Face dataset identifiers and configurations
DATASET_REGISTRY = {
    # Text & Code
    "redpajama": ("togethercomputer/RedPajama-Data-1T", None),
    "the_stack": ("bigcode/the-stack-v2", None),
    "dolma": ("allenai/dolma", None),
    "openhermes": ("teknium/OpenHermes-2.5", None),
    "gsm8k": ("gsm8k", "main"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge"),

    # Image-to-Text
    "coco": ("HuggingFaceM4/COCO", None),
    "textcaps": ("textcaps", None),
    "visual_genome": ("visual_genome", "v1.2"),

    # Audio-to-Text
    "common_voice": ("mozilla-foundation/common_voice_11_0", "en"),
    "librispeech": ("librispeech_asr", "clean"),
    "fsd50k": ("fsd50k", None),
}

def main():
    """
    A utility script to download and cache datasets from Hugging Face.

    This script simplifies the process of fetching the large datasets required
    for training and fine-tuning NeuralFlex-MoE.
    """
    parser = argparse.ArgumentParser(description="Download and cache Hugging Face datasets.")
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=list(DATASET_REGISTRY.keys()),
        help=f"The friendly name of the dataset to download. Choose from: {list(DATASET_REGISTRY.keys())}"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/cached",
        help="The directory to save the cached dataset."
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional: The specific split to download (e.g., 'train', 'validation[:10%]')."
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="If set, streams the dataset instead of downloading. Good for a quick preview."
    )

    args = parser.parse_args()
    logger = setup_logger()

    if args.dataset_name not in DATASET_REGISTRY:
        logger.error(f"Dataset '{args.dataset_name}' not recognized. Please choose from the available options.")
        return

    hf_id, subset = DATASET_REGISTRY[args.dataset_name]
    save_path = os.path.join(args.output_dir, args.dataset_name)

    logger.info(f"Preparing to download dataset: '{args.dataset_name}'")
    logger.info(f"Hugging Face ID: {hf_id}")
    if subset:
        logger.info(f"Subset: {subset}")
    logger.info(f"Output directory: {save_path}")

    try:
        dataset = load_dataset(
            hf_id,
            name=subset,
            split=args.split,
            streaming=args.streaming,
            cache_dir=os.path.join(args.output_dir, ".cache") # Centralize cache
        )

        if args.streaming:
            logger.info("Streaming dataset. Taking the first 100 examples for preview.")
            for example in dataset.take(100):
                print(example)
            logger.info("Streaming preview complete.")
        else:
            logger.info(f"Downloading and caching dataset to disk...")
            dataset.save_to_disk(save_path)
            logger.info(f"Dataset successfully saved to: {save_path}")

    except Exception as e:
        logger.error(f"Failed to download or process dataset '{args.dataset_name}'. Error: {e}")
        logger.error(
            "Please check the dataset name, your internet connection, and available disk space."
        )

if __name__ == "__main__":
    main()