import os
import yaml
import pandas as pd
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from judge_tool.models.schemas import EvaluationInput, PairwiseInput, Rubric
from judge_tool.core.judge import AbsoluteScorer, PairwiseScorer
from judge_tool.core.batch import BatchEvaluator

# Load environment variables
load_dotenv()

app = FastAPI(title="LLM-as-a-Judge UI")

# Get path to static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return f.read()
    return "<h1>Index.html not found. Please create it in judge_tool/web/static/index.html</h1>"

import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_msg = f"Internal Server Error: {str(exc)}"
    print(f"ERROR: {error_msg}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": error_msg, "traceback": traceback.format_exc()}
    )

@app.post("/api/evaluate-single")
async def evaluate_single(
    prompt: str = Form(""),
    response: str = Form(""),
    rubric_name: str = Form("Helpfulness"),
    model: str = Form("gpt-4o"),
    reference: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None),
    x_api_base: Optional[str] = Header(None)
):
    try:
        if not prompt.strip() or not response.strip():
            return JSONResponse({"error": "Prompt and Response are required fields."}, status_code=400)
            
        models = [m.strip() for m in model.split(",") if m.strip()]
            
        # Load default rubric for now (could be dynamic)
        rubric_path = f"configs/{rubric_name.lower()}.yaml"
        if not os.path.exists(rubric_path):
            return JSONResponse({"error": f"Rubric not found: {rubric_name}"}, status_code=404)

        with open(rubric_path, "r") as f:
            rubric = Rubric(**yaml.safe_load(f))

        results = []
        for mod in models:
            scorer = AbsoluteScorer(model_name=mod, api_key=x_api_key, api_base=x_api_base)
            input_data = EvaluationInput(prompt=prompt, response=response, reference_answer=reference, context=context)
            results.append(scorer.score(input_data, rubric))
        
        if len(results) == 1:
            res = results[0]
            return {
                "score": res.score,
                "max_score": rubric.max_score,
                "reasoning": res.reasoning,
                "criterion": res.criterion,
                "is_ensemble": False
            }
        else:
            avg_score = sum(r.score for r in results) / len(results)
            combined_reasoning = "\n\n=== ENSEMBLE REASONING ===\n"
            for r, mod in zip(results, models):
                combined_reasoning += f"\n[{mod} (Score: {r.score})]\n{r.reasoning}\n"
            
            return {
                "score": round(avg_score, 2),
                "max_score": rubric.max_score,
                "reasoning": combined_reasoning.strip(),
                "criterion": rubric.name,
                "is_ensemble": True,
                "ensemble_scores": {m: r.score for m, r in zip(models, results)}
            }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/compare")
async def compare_responses(
    prompt: str = Form(""),
    response_a: str = Form(""),
    response_b: str = Form(""),
    model: str = Form("gpt-4o"),
    reference: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    x_api_key: Optional[str] = Header(None),
    x_api_base: Optional[str] = Header(None)
):
    try:
        if not prompt.strip() or not response_a.strip() or not response_b.strip():
            return JSONResponse({"error": "Prompt, Response A, and Response B are required fields."}, status_code=400)
            
        models = [m.strip() for m in model.split(",") if m.strip()]
            
        results = []
        for mod in models:
            scorer = PairwiseScorer(model_name=mod, api_key=x_api_key, api_base=x_api_base)
            input_data = PairwiseInput(
                prompt=prompt, 
                response_a=response_a, 
                response_b=response_b, 
                reference_answer=reference,
                context=context
            )
            results.append(scorer.compare(input_data))
        
        if len(results) == 1:
            res = results[0]
            return {
                "winner": res.winner,
                "score_a": res.score_a,
                "score_b": res.score_b,
                "max_score": 5,
                "reasoning": res.reasoning,
                "is_ensemble": False
            }
        else:
            winners = [r.winner for r in results]
            a_count = winners.count("A")
            b_count = winners.count("B")
            if a_count > b_count: final_winner = "A"
            elif b_count > a_count: final_winner = "B"
            else: final_winner = "Tie"

            avg_a = sum(r.score_a for r in results) / len(results)
            avg_b = sum(r.score_b for r in results) / len(results)
            
            combined_reasoning = "\n\n=== ENSEMBLE REASONING ===\n"
            for r, mod in zip(results, models):
                combined_reasoning += f"\n[{mod} (Winner: {r.winner}, A:{r.score_a}, B:{r.score_b})]\n{r.reasoning}\n"
            
            return {
                "winner": final_winner,
                "score_a": round(avg_a, 2),
                "score_b": round(avg_b, 2),
                "max_score": 5,
                "reasoning": combined_reasoning.strip(),
                "is_ensemble": True
            }
    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/evaluate-batch")
async def evaluate_batch(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o"),
    rubric_name: str = Form("Helpfulness"),
    x_api_key: Optional[str] = Header(None),
    x_api_base: Optional[str] = Header(None)
):
    try:
        # Temp save and process
        content = await file.read()
        filename = file.filename
        
        if filename.endswith(".csv"):
            import io
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(".jsonl"):
            import io
            df = pd.read_json(io.BytesIO(content), lines=True)
        else:
            return JSONResponse({"error": "Only CSV and JSONL supported via web for now"}, status_code=400)

        if df.empty:
            return JSONResponse({"error": "The uploaded file is empty."}, status_code=400)

        # Basic column check
        required_cols = ["prompt", "response"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return JSONResponse({"error": f"Missing required columns: {', '.join(missing)}"}, status_code=400)

        rubric_path = f"configs/{rubric_name.lower()}.yaml"
        if not os.path.exists(rubric_path):
            return JSONResponse({"error": f"Rubric not found: {rubric_name}"}, status_code=404)
            
        with open(rubric_path, "r") as f:
            rubric = Rubric(**yaml.safe_load(f))

        # Detect evaluation type based on columns
        evaluator = BatchEvaluator()
        
        if "response_a" in df.columns and "response_b" in df.columns:
            eval_type = "pairwise_comparison"
            scorer = PairwiseScorer(model_name=model, api_key=x_api_key, api_base=x_api_base)
            results_df = evaluator.evaluate_pairwise_dataframe(df, scorer)
        else:
            eval_type = "absolute_scoring"
            scorer = AbsoluteScorer(model_name=model, api_key=x_api_key, api_base=x_api_base)
            results_df = evaluator.evaluate_dataframe(df, rubric, scorer)
        
        # Structure all results for download to match the desired format
        all_results = []
        for _, row in results_df.iterrows():
            if eval_type == "absolute_scoring":
                result_obj = {
                    "score": row.get("score"),
                    "max_score": rubric.max_score,
                    "reasoning": row.get("reasoning"),
                    "criterion": rubric.name
                }
            else:
                result_obj = {
                    "winner": row.get("winner"),
                    "score_a": row.get("score_a"),
                    "score_b": row.get("score_b"),
                    "max_score": 5,
                    "reasoning": row.get("reasoning")
                }

            item = {
                "type": eval_type,
                "model": model,
                "prompt": row.get("prompt"),
                "reference": row.get("reference_answer") or row.get("reference"),
                "result": result_obj
            }
            
            if eval_type == "absolute_scoring":
                item["rubric"] = rubric_name
                item["response"] = row.get("response")
            else:
                item["response_a"] = row.get("response_a")
                item["response_b"] = row.get("response_b")
                
            all_results.append(item)
            
        # Return a few samples for preview
        samples = results_df.head(10).to_dict(orient="records")
        
        return {
            "message": f"Batch evaluation complete for {len(results_df)} rows ({eval_type}).",
            "samples": samples,
            "all_results": all_results,
            "total": len(results_df)
        }
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Batch processing failed: {str(e)}"}, status_code=500)
