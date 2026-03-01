"""
Fine-tune sentence transformers models for embeddings.
Supports MultipleNegativesRankingLoss and TripletLoss.
"""

from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample


def finetune_bge_m3(
    dataset_name: str = "c-s-ale/alpaca-gpt4-data-zh",
    model_name: str = "BAAI/bge-m3",
    output_dir: str = "output",
    test_size: float = 0.3,
    batch_size: int = 16,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
):
    """
    Fine-tune BGE-M3 model on a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        model_name: Pre-trained model name
        output_dir: Output directory for saved model
        test_size: Test split ratio
        batch_size: Training batch size
        num_epochs: Number of training epochs
        warmup_ratio: Warmup steps ratio
    """
    # Load and split dataset
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.train_test_split(test_size=test_size)
    
    # Build training samples
    print("Building training examples...")
    train_examples = []
    for row in dataset["train"]:
        # Concatenate instruction and input as query, output as positive example
        query = row["instruction"] + row["input"]
        positive = row["output"]
        train_examples.append(InputExample(texts=[query, positive]))
    
    print(f"Created {len(train_examples)} training examples")
    
    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    
    # Load pre-trained model
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Define loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    # Alternative: train_loss = losses.TripletLoss(model=model)  # For triplet inputs
    
    # Training configuration
    warmup_steps = int(warmup_ratio * len(train_dataloader) * num_epochs)
    
    print(f"Training for {num_epochs} epochs with {warmup_steps} warmup steps...")
    
    # Start training
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )
    
    # Save fine-tuned model
    print(f"Saving model to {output_dir}")
    model.save(output_dir)
    print("Training complete!")


if __name__ == "__main__":
    # Example usage
    finetune_bge_m3()
