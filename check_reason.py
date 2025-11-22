import json
from tqdm import tqdm
from argonium_score_parallel_v9 import generate_answer, load_model_config, parse_arguments

# ---------------------------
# Scoring for reasoning quality
# ---------------------------
REASONING_SCORE_MAP = {
    "excellent": 4,
    "good": 3,
    "weak": 2,
    "incorrect": 1,
    "invalid": 0
}


def evaluate_reasoning_with_model(item, config):
    """
    Evaluate the reasoning quality of a single MCQA item using your generate_answer() function.
    Returns parsed JSON and adds a numerical score.
    """

    eval_question = f"""
You will evaluate the quality of the reasoning in the model's answer to a question.

--- Question ---
{item['question']}

--- Correct Answer ---
{item['reference_answer']}

--- Model's Answer (including reasoning) ---
{item['model_answer']}

Task:
Evaluate ONLY the reasoning quality (not the final answer correctness).
Focus on:
- logical soundness
- factual correctness
- alignment with the question
- avoidance of hallucinations
- completeness and relevance

Output STRICTLY in JSON:
{{
  "reasoning_quality": "excellent/good/weak/incorrect",
  "explanation": "One short paragraph explaining your judgment."
}}
"""

    # Call your existing generator
    response = generate_answer(
        question=eval_question,
        config=config,
        question_format="qa"
    )

    # Attempt to parse model output
    try:
        parsed = json.loads(response)
        quality = parsed.get("reasoning_quality", "invalid").lower()
    except:
        parsed = {
            "reasoning_quality": "invalid",
            "explanation": f"Model returned non-JSON output: {response}"
        }
        quality = "invalid"

    # Attach numerical score
    parsed["score"] = REASONING_SCORE_MAP.get(quality, 0)

    return parsed


def process_json_file(input_file, output_file, config):
    """
    Process a result file, evaluate reasoning, attach evaluations & scores,
    and compute average reasoning score.
    """

    print(f"Loading JSON: {input_file}")
    with open(input_file, "r", encoding="utf8") as f:
        data = json.load(f)

    results = data["results"]
    print(f"Evaluating reasoning for {len(results)} entries...\n")

    reasoning_scores = []

    for item in tqdm(results[:10]):
        reasoning_eval = evaluate_reasoning_with_model(item, config)
        item["reasoning_evaluation"] = reasoning_eval

        # collect score
        reasoning_scores.append(reasoning_eval["score"])

    # Compute average reasoning score
    avg_score = sum(reasoning_scores) / len(reasoning_scores)
    data["reasoning_average_score"] = avg_score

    print(f"\nAverage reasoning score: {avg_score:.4f}")

    with open(output_file, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Output saved to: {output_file}")
    print(f"Average reasoning score = {avg_score:.4f}")


def main():
    args = parse_arguments()
    model_config = load_model_config(args.model, args.config)
    grader_config = load_model_config(args.grader, args.config)

    process_json_file(
        input_file="results_llama70_20251121_190817.json",
        output_file="results_with_reasoning_eval.json",
        config=grader_config
    )


if __name__ == "__main__":
    main()
