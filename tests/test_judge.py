import json
import pandas as pd
from unittest.mock import patch, MagicMock
from judge_tool.core.judge import AbsoluteScorer, PairwiseScorer
from judge_tool.core.batch import BatchEvaluator
from judge_tool.models.schemas import EvaluationInput, Rubric, PairwiseInput

def test_absolute_scorer_success():
    # Mock LiteLLM completion
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "```json\n{\"score\": 4.5, \"reasoning\": \"The response is very good.\"}\n```"
    
    with patch("litellm.completion", return_value=mock_response):
        scorer = AbsoluteScorer(model_name="test-model")
        rubric = Rubric(
            name="Test Criterion",
            description="Testing...",
            scoring_guide={1: "Poor", 5: "Excellent"},
            min_score=1,
            max_score=5
        )
        input_data = EvaluationInput(prompt="Hello?", response="Hi there!")
        
        result = scorer.score(input_data, rubric)
        
        assert result.score == 4.5
        assert result.reasoning == "The response is very good."
        assert result.criterion == "Test Criterion"
        assert result.model_name == "test-model"

def test_pairwise_scorer_success():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "{\"winner\": \"A\", \"reasoning\": \"A is more concise.\"}"
    
    with patch("litellm.completion", return_value=mock_response):
        scorer = PairwiseScorer(model_name="test-model")
        input_data = PairwiseInput(prompt="Hello?", response_a="Hi!", response_b="Hello there!")
        
        result = scorer.compare(input_data)
        
        assert result.winner == "A"
        assert result.reasoning == "A is more concise."

def test_batch_evaluator_absolute():
    df = pd.DataFrame({
        "prompt": ["P1", "P2"],
        "response": ["R1", "R2"]
    })
    
    mock_result = MagicMock()
    mock_result.score = 5.0
    mock_result.reasoning = "Good"
    
    with patch("judge_tool.core.judge.AbsoluteScorer.score", return_value=mock_result):
        scorer = AbsoluteScorer()
        evaluator = BatchEvaluator(max_workers=2)
        rubric = Rubric(name="T", description="D", scoring_guide={1: "P", 5: "E"})
        
        results_df = evaluator.evaluate_dataframe(df, rubric, scorer)
        
        assert len(results_df) == 2
        assert "score" in results_df.columns
        assert results_df["score"].iloc[0] == 5.0

def test_batch_evaluator_pairwise():
    df = pd.DataFrame({
        "p": ["P1"],
        "ra": ["RA1"],
        "rb": ["RB1"]
    })
    
    mock_result = MagicMock()
    mock_result.winner = "B"
    mock_result.reasoning = "B is better"
    
    with patch("judge_tool.core.judge.PairwiseScorer.compare", return_value=mock_result):
        scorer = PairwiseScorer()
        evaluator = BatchEvaluator(max_workers=2)
        
        column_mapping = {"prompt": "p", "response_a": "ra", "response_b": "rb"}
        results_df = evaluator.evaluate_pairwise_dataframe(df, scorer, column_mapping)
        
        assert len(results_df) == 1
        assert results_df["winner"].iloc[0] == "B"
        assert results_df["reasoning"].iloc[0] == "B is better"

def test_json_extraction_robustness():
    scorer = AbsoluteScorer()
    
    # Test with markdown code block
    text1 = "Here is the result: ```json\n{\"score\": 3, \"reasoning\": \"ok\"}\n``` hope it helps."
    json1 = scorer._extract_json(text1)
    assert json1["score"] == 3
    
    # Test with raw JSON and extra text
    text2 = "Reasoning is here... {\"score\": 4, \"reasoning\": \"good\"} and some more text."
    json2 = scorer._extract_json(text2)
    assert json2["score"] == 4
