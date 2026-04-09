# judge_tool/core/prompts.py

def get_absolute_evaluation_prompt(
    criterion_name: str,
    criterion_description: str,
    scoring_guide: str,
    prompt: str,
    response: str,
    min_score: int,
    max_score: int,
    reference_answer: str = "N/A",
    context: str = None
) -> str:
    """절대 평가(Absolute Scoring)를 위한 프롬프트를 생성합니다."""
    
    context_block = f"""
Provided Context (Source Document):
{context}
""" if context else ""

    return f"""
Evaluate the following response based on the provided criterion and rubric.

Criterion: {criterion_name}
Description: {criterion_description}

Rubric Scoring Guide:
{scoring_guide}
{context_block}
Original Prompt:
{prompt}

Model Response to Evaluate:
{response}

Reference Answer (if any):
{reference_answer}

Please provide your evaluation in the following JSON format:
{{
  "reasoning": "A detailed explanation of the response's quality, identifying specific strengths and weaknesses based on the rubric.",
  "score": "The numerical score from the rubric ({min_score}-{max_score})"
}}
Make sure your reasoning comes first in the JSON to ensure you think through the evaluation step-by-step.
"""

def get_pairwise_comparison_prompt(
    prompt: str,
    response_a: str,
    response_b: str,
    reference_answer: str = "N/A",
    context: str = None
) -> str:
    """쌍비교 평가(Pairwise Comparison)를 위한 프롬프트를 생성합니다."""
    
    context_block = f"""
Provided Context (Source Document):
{context}
""" if context else ""

    return f"""
Compare the following two model responses and determine which one is better based on the original prompt.

Original Prompt:
{prompt}
{context_block}
Response A:
{response_a}

Response B:
{response_b}

Reference Answer (if any):
{reference_answer}

Please provide your evaluation in the following JSON format:
{{
  "reasoning": "A detailed comparison between Response A and Response B, highlighting why one is better than the other or why they are equal.",
  "winner": "A, B, or Tie",
  "score_a": "Numerical score for Response A (1-5)",
  "score_b": "Numerical score for Response B (1-5)"
}}
Ensure your reasoning leads logically to your choice of winner and scores. 
Higher scores should be given to responses that are more accurate, helpful, and follow instructions better.
"""

def get_system_prompt_absolute() -> str:
    return "You are an expert judge for LLM evaluations. You must provide objective, evidence-based feedback based on the provided rubric."

def get_system_prompt_pairwise() -> str:
    return "You are an expert judge comparing model outputs. Be impartial and focus on accuracy, helpfulness, and adherence to instructions."
