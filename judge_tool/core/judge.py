import json
import re
import os
import time
import threading
from typing import Optional, List, Any, Dict
import litellm
from google import genai
from judge_tool.models.schemas import EvaluationInput, EvaluationResult, Rubric, PairwiseInput, PairwiseResult

class Judge:
    # Global state for rate limiting (5 RPM = 1 request every 12 seconds)
    _last_call_time = 0.0
    _lock = threading.Lock()
    MIN_INTERVAL = 12.1  # Added a small buffer to 12.0s

    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, api_base: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base

    def _wait_for_rate_limit(self):
        """Ensures that API calls are spaced out to respect RPM limits."""
        with Judge._lock:
            now = time.time()
            elapsed = now - Judge._last_call_time
            wait_time = Judge.MIN_INTERVAL - elapsed

            if wait_time > 0:
                # Only log/print if wait is significant
                if wait_time > 1:
                    print(f"[Rate Limiter] Waiting {wait_time:.2f}s to respect RPM limit...")
                time.sleep(wait_time)

            Judge._last_call_time = time.time()

    def _call_llm(self, prompt: str, system_prompt: str = "You are a helpful and impartial judge.") -> str:
        # Apply rate limiting before calling the API
        self._wait_for_rate_limit()

        model_name = self.model_name
        model_lower = model_name.lower()
        
        # Check if it's a Gemini model to use google-genai SDK
        is_gemini = "gemini" in model_lower or "google" in model_lower
        
        # Resolve API key
        api_key = self.api_key
        if not api_key:
            if is_gemini:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            elif "claude" in model_lower or "anthropic" in model_lower:
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif "gpt" in model_lower or "openai" in model_lower:
                api_key = os.getenv("OPENAI_API_KEY")

        if api_key and "your_" in str(api_key).lower():
            raise ValueError(f"Found placeholder API key for {model_name}. Please update your .env file.")

        # --- Gemini Specific Logic using google-genai ---
        if is_gemini:
            # Clean model name (remove provider prefix if any)
            clean_model = model_name.split("/")[-1]
            try:
                client = genai.Client(api_key=api_key)
                # Combine system prompt and prompt for the SDK
                full_prompt = f"{system_prompt}\n\n{prompt}"
                response = client.models.generate_content(
                    model=clean_model,
                    contents=full_prompt
                )
                if not response.text:
                    raise RuntimeError("Gemini API returned an empty response.")
                return response.text
            except Exception as e:
                raise RuntimeError(f"Gemini API call failed: {str(e)}")

        # --- Default to LiteLLM for other models ---
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        if "/" not in model_name:
            if "claude" in model_lower or "anthropic" in model_lower:
                model_name = f"anthropic/{model_name}"
            elif "gpt" in model_lower or "openai" in model_lower:
                if not model_name.startswith("gpt-"):
                    model_name = f"openai/{model_name}"

        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0.0,
            "api_key": api_key,
            "num_retries": 3
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"LLM call failed for model {model_name}: {str(e)}")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from a string, handling potential Markdown code blocks or extra text.
        """
        # Try to find JSON within markdown code blocks first
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not json_match:
            # Try to find any JSON-like structure if no markdown code blocks are present
            json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Fallback for very simple JSON that might be returned without code blocks or with extra text
        try:
            # Find the first { and last } and hope for the best
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
            
        raise ValueError(f"Could not parse JSON from LLM output: {text}")

from judge_tool.core.prompts import (
    get_absolute_evaluation_prompt,
    get_pairwise_comparison_prompt,
    get_system_prompt_absolute,
    get_system_prompt_pairwise
)

class AbsoluteScorer(Judge):
    def score(self, input_data: EvaluationInput, rubric: Rubric) -> EvaluationResult:
        sorted_keys = sorted(rubric.scoring_guide.keys())
        scoring_guide_text = "\n".join([f"- {score}: {rubric.scoring_guide[score]}" for score in sorted_keys])
        
        prompt = get_absolute_evaluation_prompt(
            criterion_name=rubric.name,
            criterion_description=rubric.description,
            scoring_guide=scoring_guide_text,
            prompt=input_data.prompt,
            response=input_data.response,
            min_score=rubric.min_score,
            max_score=rubric.max_score,
            reference_answer=input_data.reference_answer or "N/A",
            context=input_data.context
        )
        
        try:
            raw_output = self._call_llm(prompt, get_system_prompt_absolute())
            json_data = self._extract_json(raw_output)
            return EvaluationResult(
                score=float(json_data["score"]),
                reasoning=json_data["reasoning"],
                criterion=rubric.name,
                model_name=self.model_name
            )
        except Exception as e:
            print(f"AbsoluteScorer Exception: {str(e)}")
            return EvaluationResult(
                score=0.0,
                reasoning=f"Error in judge evaluation: {str(e)}",
                criterion=rubric.name,
                model_name=self.model_name
            )

class PairwiseScorer(Judge):
    def compare(self, input_data: PairwiseInput) -> PairwiseResult:
        prompt = get_pairwise_comparison_prompt(
            prompt=input_data.prompt,
            response_a=input_data.response_a,
            response_b=input_data.response_b,
            reference_answer=input_data.reference_answer or "N/A",
            context=input_data.context
        )
        
        try:
            raw_output = self._call_llm(prompt, get_system_prompt_pairwise())
            json_data = self._extract_json(raw_output)
            
            # Robust score extraction
            def extract_score(data, keys):
                for k in keys:
                    val = data.get(k)
                    if val is not None:
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            continue
                return 0.0

            s_a = extract_score(json_data, ["score_a", "scoreA", "score_A", "score"])
            s_b = extract_score(json_data, ["score_b", "scoreB", "score_B", "score"])
            
            return PairwiseResult(
                winner=json_data.get("winner", "Unknown"),
                score_a=s_a,
                score_b=s_b,
                reasoning=json_data.get("reasoning", "No reasoning provided."),
                model_name=self.model_name
            )
        except Exception as e:
            print(f"PairwiseScorer Exception: {str(e)}")
            return PairwiseResult(
                winner="Error",
                score_a=0.0,
                score_b=0.0,
                reasoning=f"Error in pairwise comparison: {str(e)}",
                model_name=self.model_name
            )
