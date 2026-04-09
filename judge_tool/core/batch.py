import pandas as pd
from typing import List, Dict, Any, Union, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from judge_tool.models.schemas import EvaluationInput, Rubric, EvaluationResult, PairwiseInput, PairwiseResult
from judge_tool.core.judge import AbsoluteScorer, PairwiseScorer

class BatchEvaluator:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers

    def evaluate_dataframe(
        self, 
        df: pd.DataFrame, 
        rubric: Rubric, 
        scorer: AbsoluteScorer,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate a dataframe using AbsoluteScorer with parallel processing.
        """
        mapping = column_mapping or {}
        prompt_col = mapping.get("prompt", "prompt")
        response_col = mapping.get("response", "response")
        ref_col = mapping.get("reference", "reference_answer")

        results = [None] * len(df)
        
        def process_row(seq_idx, row_tuple):
            index, row = row_tuple
            input_data = EvaluationInput(
                prompt=row[prompt_col],
                response=row[response_col],
                reference_answer=row.get(ref_col)
            )
            result = scorer.score(input_data, rubric)
            return seq_idx, {
                "score": result.score,
                "reasoning": result.reasoning
            }

        for i, row_tuple in tqdm(enumerate(df.iterrows()), total=len(df), desc="Evaluating"):
            seq_idx, res = process_row(i, row_tuple)
            results[seq_idx] = res
            
        results_df = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)
        return results_df

    def evaluate_pairwise_dataframe(
        self, 
        df: pd.DataFrame, 
        scorer: PairwiseScorer,
        column_mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Evaluate a dataframe using PairwiseScorer with parallel processing.
        """
        mapping = column_mapping or {}
        prompt_col = mapping.get("prompt", "prompt")
        res_a_col = mapping.get("response_a", "response_a")
        res_b_col = mapping.get("response_b", "response_b")
        ref_col = mapping.get("reference", "reference_answer")

        results = [None] * len(df)

        def process_row(seq_idx, row_tuple):
            index, row = row_tuple
            input_data = PairwiseInput(
                prompt=row[prompt_col],
                response_a=row[res_a_col],
                response_b=row[res_b_col],
                reference_answer=row.get(ref_col)
            )
            result = scorer.compare(input_data)
            return seq_idx, {
                "winner": result.winner,
                "reasoning": result.reasoning
            }

        for i, row_tuple in tqdm(enumerate(df.iterrows()), total=len(df), desc="Comparing"):
            seq_idx, res = process_row(i, row_tuple)
            results[seq_idx] = res

        results_df = pd.concat([df, pd.DataFrame(results, index=df.index)], axis=1)
        return results_df
