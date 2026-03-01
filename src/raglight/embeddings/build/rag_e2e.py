"""
End-to-End RAG training with QLoRA.
Combines retriever and generator models with LoRA fine-tuning.
"""

from typing import List, Optional, Union, Any, Dict
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class AutoModelForRagE2E(torch.nn.Module):
    """
    End-to-End RAG model combining retriever and generator with QLoRA training.
    
    Uses 4-bit quantization and LoRA to reduce training resource requirements.
    """
    
    def __init__(
        self,
        retriever_name: str = 'BAAI/bge-m3',
        generator_name: str = 'meta-llama/Llama-2-7b',
        normalize: bool = True,
    ) -> None:
        super(AutoModelForRagE2E, self).__init__()
        
        self.normalize = normalize
        
        # Load retriever model with quantization
        self.retriever_model = AutoModel.from_pretrained(
            retriever_name,
            quantization_config=self._get_bnb_config()
        )
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name)
        
        # Load generator model with quantization
        self.generator_model = AutoModelForCausalLM.from_pretrained(
            generator_name,
            quantization_config=self._get_bnb_config(),
            trust_remote_code=True,
        )
        self.generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
        
        # Apply LoRA to retriever
        self.retriever_model = get_peft_model(
            self.retriever_model,
            peft_config=self._get_lora_config(
                TaskType.FEATURE_EXTRACTION,
                target_modules=["key", "query", "value"]
            ),
        )
        
        # Apply LoRA to generator
        self.generator_model = get_peft_model(
            self.generator_model,
            peft_config=self._get_lora_config(
                TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],
            ),
        )
    
    def retrieval_forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for retrieval task."""
        token_embeddings = self.retriever_model(input_ids, attention_mask)[0]
        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def forward(
        self, 
        task: str, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for either retrieval or generation task."""
        if task == "retrieval":
            return self.retrieval_forward(input_ids, attention_mask)
        else:
            gen_outputs = self.generator_model(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
            return gen_outputs.logits
    
    def _get_bnb_config(self) -> BitsAndBytesConfig:
        """Get 4-bit quantization configuration."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    def _get_lora_config(
        self,
        task_type: TaskType,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        bias: str = "none",
        target_modules: Optional[Union[List[str], str]] = None,
    ) -> LoraConfig:
        """Get LoRA configuration."""
        return LoraConfig(
            task_type=task_type,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            target_modules=target_modules,
        )
    
    def mean_pooling(
        self, 
        token_embeddings: torch.Tensor, 
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pooling for sentence embeddings."""
        input_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask, 1) / torch.clamp(
            input_mask.sum(1), min=1e-9
        )


# Data processing constants
RETRIEVAL_PROMPT = "为这个句子生成表示以用于检索相关文章："
GENERATOR_PROMPT = """根据上下文回答问题，如果知识片段不包含答案请回复"未知"，不要回复知识片段以外的知识，回复内容保持精简，不要提供过多的分析
## 知识片段
----
{}
----
## 问题：
----
{}
----"""


def preprocess_dataset(
    examples: Dict[str, List],
    retriever_tokenizer: AutoTokenizer,
    generator_tokenizer: AutoTokenizer,
    query_column_name: str = 'query',
    passage_column_name: str = 'passage',
    answer_column_name: str = 'answer',
    query_max_len: int = 128,
    passage_max_len: int = 512,
    generator_max_len: int = 512,
) -> Dict[str, Any]:
    """
    Preprocess dataset for end-to-end RAG training.
    
    Args:
        examples: Dataset examples
        retriever_tokenizer: Tokenizer for retriever
        generator_tokenizer: Tokenizer for generator
        query_column_name: Column name for queries
        passage_column_name: Column name for passages
        answer_column_name: Column name for answers
        query_max_len: Max length for queries
        passage_max_len: Max length for passages
        generator_max_len: Max length for generator input
        
    Returns:
        Preprocessed batch
    """
    query_list = examples[query_column_name]
    passage_list = examples[passage_column_name]
    answers = examples[answer_column_name]
    
    # Add retrieval prompt
    queries = [RETRIEVAL_PROMPT + query for query in query_list]
    passages = [RETRIEVAL_PROMPT + passage for passage in passage_list]
    
    # Tokenize for retriever
    retriever_query_tokens = retriever_tokenizer(
        queries, padding="max_length", max_length=query_max_len, truncation=True
    )
    retriever_passage_tokens = retriever_tokenizer(
        passages, padding="max_length", max_length=passage_max_len, truncation=True
    )
    
    # Prepare generator input
    casual_input_text = []
    for passage, query, answer in zip(passage_list, query_list, answers, strict=True):
        casual_input_text.append(
            f"<|im_start|>user\n{GENERATOR_PROMPT.format(passage, query)}<|im_end|>\n"
            f"<|im_start|>assistant\n{answer}"
        )
    
    causal_input_tokens = generator_tokenizer(
        casual_input_text, 
        padding="max_length", 
        max_length=generator_max_len, 
        truncation=True
    )
    
    # Calculate query+passage lengths for loss computation
    query_passage_text = []
    for passage, query in zip(passage_list, query_list, strict=True):
        query_passage_text.append(
            f"<|im_start|>user\n{GENERATOR_PROMPT.format(passage, query)}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    query_passage_lengths = []
    query_passage_tokens = generator_tokenizer(query_passage_text, padding=False)
    for single_query_passage in query_passage_tokens["input_ids"]:
        query_passage_lengths.append(len(single_query_passage))
    
    # Build batch
    pre_batch = {}
    for k, v in retriever_query_tokens.items():
        pre_batch[f"retriever_query_{k}"] = v
    for k, v in retriever_passage_tokens.items():
        pre_batch[f"retriever_passage_{k}"] = v
    for k, v in causal_input_tokens.items():
        pre_batch[f"generator_input_{k}"] = v
    pre_batch["query_passage_input_len"] = query_passage_lengths
    
    return pre_batch


def get_nt_xent_loss(sim_scores: torch.Tensor) -> torch.Tensor:
    """NT-Xent loss for contrastive learning."""
    return F.cross_entropy(
        sim_scores, 
        torch.arange(len(sim_scores), device=sim_scores.device)
    )


def marginalize_log_probs(
    logprobs_logits: torch.FloatTensor,
    doc_logprobs: torch.FloatTensor,
    query_token_length: torch.IntTensor,
) -> torch.Tensor:
    """
    Marginalize log probabilities over documents.
    
    Used for computing the RAG loss that marginalizes over retrieved documents.
    """
    query_passage_log_prob = logprobs_logits[:, query_token_length - 1, :]
    answer_log_prob = logprobs_logits[query_token_length - 1:, :]
    marginalized_prob_sum = answer_log_prob + doc_logprobs
    all_log_probs = torch.cat([query_passage_log_prob, marginalized_prob_sum], dim=0)
    return all_log_probs


def get_nll(log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Get negative log likelihood."""
    return F.nll_loss(log_probs.permute(0, 2, 1), targets, reduction="none")


def compute_marginalized_loss_from_logits(
    logits: torch.LongTensor,
    input_tensors: torch.Tensor,
    attention_mask: torch.Tensor,
    scores: torch.Tensor,
    query_token_length: torch.Tensor,
) -> torch.Tensor:
    """
    Compute marginalized loss from logits.
    
    This implements the RAG loss that marginalizes over retrieved documents.
    """
    logprobs_logits = F.log_softmax(logits[:, :-1, :], dim=2).view(
        logits.shape[0], -1, logits.size(-1)
    )
    doc_logprobs = torch.log_softmax(scores, dim=1).diag().unsqueeze(-1).unsqueeze(-1)
    
    marginalized_next_word_prob_list = []
    for sample_logprobs_logits, sample_doc_logprobs, sample_token_length in zip(
        logprobs_logits, doc_logprobs, query_token_length, strict=True
    ):
        marginalized_log_probs = marginalize_log_probs(
            sample_logprobs_logits,
            sample_doc_logprobs,
            sample_token_length
        )
        marginalized_next_word_prob_list.append(marginalized_log_probs)
    
    marginalized_log_probs = torch.stack(marginalized_next_word_prob_list)
    loss = get_nll(marginalized_log_probs, input_tensors[:, 1:])
    loss_tensor = loss * attention_mask[:, 1:]
    overall_average_loss = loss_tensor.sum() / attention_mask[:, 1:].sum()
    
    return overall_average_loss
