"""
Training script for end-to-end RAG model with Accelerate.
Combines retriever contrastive learning with generator marginalized loss.
"""

import os
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from rag_e2e import (
    AutoModelForRagE2E, 
    preprocess_dataset, 
    get_nt_xent_loss, 
    compute_marginalized_loss_from_logits
)


def train_rag_model(
    output_dir: str = 'output',
    dataset_path: str = '',
    per_device_train_batch_size: int = 32,
    num_train_epochs: int = 3,
    learning_rate: float = 1e-5,
    retriever_name: str = 'BAAI/bge-m3',
    generator_name: str = 'meta-llama/Llama-2-7b',
):
    """
    Train end-to-end RAG model.
    
    Args:
        output_dir: Output directory for checkpoints
        dataset_path: Path to dataset
        per_device_train_batch_size: Batch size per device
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate
        retriever_name: Retriever model name
        generator_name: Generator model name
    """
    # Initialize model
    print(f"Initializing RAG model with retriever: {retriever_name}, generator: {generator_name}")
    rag_model = AutoModelForRagE2E(
        retriever_name=retriever_name,
        generator_name=generator_name,
    )
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    retriever_tokenizer = rag_model.retriever_tokenizer
    generator_tokenizer = rag_model.generator_tokenizer
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    processed_datasets = dataset.map(
        lambda example: preprocess_dataset(
            example,
            retriever_tokenizer=retriever_tokenizer,
            generator_tokenizer=generator_tokenizer,
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=1,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        processed_datasets, 
        shuffle=True, 
        batch_size=per_device_train_batch_size, 
        pin_memory=True
    )
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(rag_model.parameters(), lr=learning_rate)
    
    # Prepare for distributed training
    print("Preparing for distributed training...")
    rag_model, optimizer, train_dataloader = accelerator.prepare(
        rag_model, optimizer, train_dataloader
    )
    
    # Training loop
    print(f"Starting training for {num_train_epochs} epochs...")
    for epoch in range(num_train_epochs):
        rag_model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(rag_model):
                # Retriever forward pass - query embeddings
                query_embs = rag_model(
                    "retrieval", 
                    batch["retriever_query_input_ids"],
                    batch["retriever_query_attention_mask"]
                )
                
                # Retriever forward pass - passage embeddings
                passage_embs = rag_model(
                    "retrieval",
                    batch["retriever_passage_input_ids"],
                    batch["retriever_passage_attention_mask"],
                )
                
                # Compute similarity scores
                logits = torch.matmul(query_embs, passage_embs.t())
                
                # Contrastive loss for retriever
                loss_query = get_nt_xent_loss(logits)
                loss_passage = get_nt_xent_loss(logits.t())
                retriever_contrastive_loss = (loss_query + loss_passage) / 2.0
                
                # Generator forward pass
                generator_logits = rag_model(
                    "generation", 
                    batch["generator_input_input_ids"],
                    batch["generator_input_attention_mask"]
                )
                
                # Marginalized loss for generator
                marginalized_causal_loss = compute_marginalized_loss_from_logits(
                    generator_logits,
                    batch["generator_input_input_ids"],
                    batch["generator_input_attention_mask"],
                    logits,
                    batch["query_passage_input_len"],
                )
                
                # Combined loss
                combined_loss = retriever_contrastive_loss + marginalized_causal_loss
                
                # Backward pass
                accelerator.backward(combined_loss)
                optimizer.step()
                rag_model.zero_grad()
                
                total_loss += combined_loss.item()
                
                if step % 10 == 0:
                    print(f"Epoch {epoch}, Step {step}, Loss: {combined_loss.item():.4f}")
        
        # Save checkpoint
        if output_dir is not None:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                print(f"Saving checkpoint for epoch {epoch}...")
                accelerator.save_state(os.path.join(output_dir, f"epoch_{epoch}"))
    
    # Final save
    print("Saving final model...")
    accelerator.wait_for_everyone()
    
    retriever_ckpt_path = os.path.join(output_dir, "retriever")
    generator_ckpt_path = os.path.join(output_dir, "generator")
    
    # Unwrap model for saving
    unwrapped_rag_model = accelerator.unwrap_model(rag_model)
    
    # Save retriever
    if accelerator.is_main_process:
        print(f"Saving retriever to {retriever_ckpt_path}")
        unwrapped_rag_model.retriever_model.save_pretrained(
            retriever_ckpt_path,
            state_dict=accelerator.get_state_dict(unwrapped_rag_model.retriever_model),
        )
        retriever_tokenizer.save_pretrained(retriever_ckpt_path)
        
        # Save generator
        print(f"Saving generator to {generator_ckpt_path}")
        unwrapped_rag_model.generator_model.save_pretrained(
            generator_ckpt_path,
            state_dict=accelerator.get_state_dict(unwrapped_rag_model.generator_model),
        )
        generator_tokenizer.save_pretrained(generator_ckpt_path)
    
    accelerator.wait_for_everyone()
    print("Training complete!")


if __name__ == "__main__":
    # Example usage
    train_rag_model(
        output_dir='output',
        dataset_path='your_dataset_path',
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=1e-5,
    )
