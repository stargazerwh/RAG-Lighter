from __future__ import annotations
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from langchain_core.documents import Document


@dataclass
class RAGResult:
    """Data class to store RAG query result."""
    question: str
    answer: str
    contexts: List[Document]


@dataclass
class RAGASEvaluationResult:
    """Data class to store RAGAS evaluation results."""
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    context_recall: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_relevancy": self.context_relevancy,
            "context_recall": self.context_recall,
        }


class RAGASEvaluator:
    """
    RAGAS-style evaluator for RAG systems.
    
    Implements key metrics:
    - Faithfulness: Answer consistency with retrieved contexts
    - Answer Relevancy: Answer relevance to the question
    - Context Relevancy: Retrieved context relevance to the question
    - Context Recall: Coverage of necessary information in contexts
    """

    def __init__(self, llm=None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm: Language model for evaluation (optional, uses heuristics if None)
        """
        self.llm = llm

    def evaluate(self, result: RAGResult) -> RAGASEvaluationResult:
        """
        Evaluate a single RAG result.
        
        Args:
            result: RAGResult containing question, answer, and contexts
            
        Returns:
            RAGASEvaluationResult with all metrics
        """
        logging.info(f"🔍 Evaluating RAG result for question: {result.question[:50]}...")
        
        return RAGASEvaluationResult(
            faithfulness=self._evaluate_faithfulness(result),
            answer_relevancy=self._evaluate_answer_relevancy(result),
            context_relevancy=self._evaluate_context_relevancy(result),
            context_recall=self._evaluate_context_recall(result),
        )

    def evaluate_batch(self, results: List[RAGResult]) -> Dict[str, List[float]]:
        """
        Evaluate multiple RAG results.
        
        Args:
            results: List of RAGResult objects
            
        Returns:
            Dictionary with metric names and list of scores
        """
        batch_results = {
            "faithfulness": [],
            "answer_relevancy": [],
            "context_relevancy": [],
            "context_recall": [],
        }
        
        for result in results:
            eval_result = self.evaluate(result)
            batch_results["faithfulness"].append(eval_result.faithfulness)
            batch_results["answer_relevancy"].append(eval_result.answer_relevancy)
            batch_results["context_relevancy"].append(eval_result.context_relevancy)
            batch_results["context_recall"].append(eval_result.context_recall)
        
        return batch_results

    def _evaluate_faithfulness(self, result: RAGResult) -> float:
        """
        Faithfulness: Measures if the answer is consistent with retrieved contexts.
        
        Score 0-1 where 1 means fully faithful (all claims supported by context).
        """
        if not result.contexts:
            return 0.0
        
        context_text = " ".join([doc.page_content for doc in result.contexts])
        answer = result.answer.lower()
        context_lower = context_text.lower()
        
        # Simple heuristic: check if key terms from answer appear in context
        # Split answer into sentences/claims
        import re
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if not sentences:
            return 0.5
        
        supported_count = 0
        for sentence in sentences:
            # Extract key terms (words > 3 chars)
            key_terms = [w for w in sentence.split() if len(w) > 3]
            if not key_terms:
                continue
                
            # Check if majority of key terms appear in context
            matches = sum(1 for term in key_terms if term in context_lower)
            if matches >= len(key_terms) * 0.5:
                supported_count += 1
        
        return supported_count / len(sentences) if sentences else 0.0

    def _evaluate_answer_relevancy(self, result: RAGResult) -> float:
        """
        Answer Relevancy: Measures if the answer is relevant to the question.
        
        Score 0-1 where 1 means fully relevant.
        """
        question = result.question.lower()
        answer = result.answer.lower()
        
        # Extract key terms from question
        import re
        question_terms = set(re.findall(r'\b\w{4,}\b', question))
        
        if not question_terms:
            return 0.5
        
        # Check overlap with answer
        answer_terms = set(re.findall(r'\b\w{4,}\b', answer))
        
        if not answer_terms:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = question_terms & answer_terms
        union = question_terms | answer_terms
        
        return len(intersection) / len(union) if union else 0.0

    def _evaluate_context_relevancy(self, result: RAGResult) -> float:
        """
        Context Relevancy: Measures if retrieved contexts are relevant to the question.
        
        Score 0-1 where 1 means all contexts are highly relevant.
        """
        if not result.contexts:
            return 0.0
        
        question = result.question.lower()
        question_terms = set(question.split())
        
        scores = []
        for doc in result.contexts:
            context = doc.page_content.lower()
            context_terms = set(context.split())
            
            if not context_terms:
                scores.append(0.0)
                continue
            
            # Calculate term overlap
            overlap = question_terms & context_terms
            score = len(overlap) / len(question_terms) if question_terms else 0.0
            scores.append(min(score * 2, 1.0))  # Scale up but cap at 1
        
        return sum(scores) / len(scores) if scores else 0.0

    def _evaluate_context_recall(self, result: RAGResult) -> float:
        """
        Context Recall: Measures if contexts contain all necessary information.
        
        Score 0-1 where 1 means complete coverage.
        Uses a heuristic based on answer coverage in contexts.
        """
        if not result.contexts:
            return 0.0
        
        context_text = " ".join([doc.page_content for doc in result.contexts])
        answer = result.answer.lower()
        context_lower = context_text.lower()
        
        # Extract key phrases from answer (3+ word sequences)
        import re
        words = answer.split()
        if len(words) < 3:
            # For short answers, check simple term overlap
            answer_terms = set(words)
            context_terms = set(context_lower.split())
            overlap = answer_terms & context_terms
            return len(overlap) / len(answer_terms) if answer_terms else 0.0
        
        # Generate n-grams from answer
        ngrams = []
        for n in [3, 4, 5]:
            for i in range(len(words) - n + 1):
                ngram = " ".join(words[i:i+n])
                ngrams.append(ngram)
        
        if not ngrams:
            return 0.5
        
        # Check how many n-grams appear in context
        found_count = sum(1 for ngram in ngrams if ngram in context_lower)
        
        # Scale: expect at least 30% of n-grams to be found for good recall
        recall = found_count / len(ngrams)
        return min(recall * 3, 1.0)  # Scale up but cap at 1

    def generate_report(self, results: List[RAGResult]) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results: List of RAGResult to evaluate
            
        Returns:
            Dictionary with summary statistics and detailed results
        """
        batch_results = self.evaluate_batch(results)
        
        report = {
            "summary": {
                "total_evaluated": len(results),
                "avg_faithfulness": sum(batch_results["faithfulness"]) / len(results),
                "avg_answer_relevancy": sum(batch_results["answer_relevancy"]) / len(results),
                "avg_context_relevancy": sum(batch_results["context_relevancy"]) / len(results),
                "avg_context_recall": sum(batch_results["context_recall"]) / len(results),
            },
            "detailed_results": [],
        }
        
        for i, result in enumerate(results):
            report["detailed_results"].append({
                "question": result.question,
                "answer": result.answer[:200] + "..." if len(result.answer) > 200 else result.answer,
                "faithfulness": batch_results["faithfulness"][i],
                "answer_relevancy": batch_results["answer_relevancy"][i],
                "context_relevancy": batch_results["context_relevancy"][i],
                "context_recall": batch_results["context_recall"][i],
            })
        
        return report
