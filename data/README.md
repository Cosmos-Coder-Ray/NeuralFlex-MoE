# Data Directory

This folder stores datasets and processed data for training.

## Structure

```
data/
├── raw/              # Original downloaded datasets
├── processed/        # Tokenized and preprocessed data
├── cache/            # Cached embeddings and features
├── chroma/           # ChromaDB vector store
└── samples/          # Sample data for testing
```

## Usage

### Download Datasets

```bash
python scripts/prepare_datasets.py --phase all --output_dir ./data
```

### Process Data

```python
from neuraflex_moe.training import DataPipeline

pipeline = DataPipeline(config)
dataset = pipeline.prepare_dataset()
```

## Supported Formats

- **Text**: `.txt`, `.json`, `.jsonl`
- **Parquet**: `.parquet` (recommended for large datasets)
- **HuggingFace**: Automatic caching from `datasets` library
- **Arrow**: `.arrow` files for fast loading

## Data Quality

All datasets are processed through:
- Quality filtering (cleanlab)
- Deduplication
- Toxicity filtering
- Length normalization

## Storage

- Raw datasets: ~15GB
- Processed data: ~20GB
- Cache: ~5GB
- **Total**: ~40GB recommended free space
