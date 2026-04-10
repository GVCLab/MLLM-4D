import json
import torch
import argparse
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from pydantic import BaseModel, ValidationError


INSTRUCTION = """Your task is to evaluate whether the model's final answer is correct by comparing it to the ground-truth answer provided for the given question.

You should first extract the final answer from the model's response, and then compare the extracted answer with the choice that matches the ground-truth answer to determine its correctness.
"""

MULTI_CHOICE_INSTRUCTION = INSTRUCTION + "Output your response in the following structured format, and do not output any other text:\n" + """{
    "extracted_answer": // str value "A" "B" "C" "D", followed by a colon and the corresponding answer text, e.g., "A: left". If the model's response does not contain a valid choice and reasoning, then "No Valid Answer".
    "correct": // boolean value, True if the extracted answer matches the ground-truth answer (correct choice), False otherwise ("No Valid Answer" is also considered False).
}
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/data/Qwen3-VL-8B-Instruct")
    parser.add_argument("--data_name", type=str, default='outputs_qwen_3_vl/real_mc_direct-output/Qwen3-VL-8B-Instruct_1fps.json')
    return parser.parse_args()


class EvaluationOutput(BaseModel):
    extracted_answer: str
    correct: bool


def prepare_evaluation_message(example):
    optionized_list = [f"{key}: {value}" for key, value in example['choices'].items()]
    optionized_str = "\n".join(optionized_list)

    question_context = f"Question: {example['question']}\n\nOptions:\n{optionized_str}"
    gt_answer = f"Ground Truth Answer: {example['answer']}"
    model_response = f"Model Response to the Question: {example['response']}"
    user_prompt = f"{question_context}\n\n{gt_answer}\n\n{model_response}"

    message = [
        {"role": "system", "content": MULTI_CHOICE_INSTRUCTION},
        {"role": "user", "content": user_prompt},
    ]
    return message


def evaluate_with_qwen(model, tokenizer, example):

    messages = prepare_evaluation_message(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=256,
        do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    try:
        json_part = response_text[response_text.find('{'):response_text.rfind('}') + 1]
        parsed_output = json.loads(json_part)
        evaluation = EvaluationOutput(**parsed_output)
        return evaluation.dict()
    except (json.JSONDecodeError, ValidationError, IndexError) as e:
        return {
            "extracted_answer": "Parsing Error",
            "correct": False
        }


def main(model_name, data_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
    with open(data_name, 'r', encoding='utf-8') as f:
        vlm_data = json.load(f)
    results = []
    correct_count = 0

    for example in vlm_data:
        print(f"\n--- 正在评估 ID: {example['id']} ---")
        evaluation_result = evaluate_with_qwen(model, tokenizer, example)
        final_result = {
            "id": example["id"],
            "question": example["question"],
            "choices": example["choices"],
            "vlm_model_output": example["response"],
            "ground_truth_answer": example["answer"],
            "judge_model_extracted_answer": evaluation_result["extracted_answer"],
            "is_correct": evaluation_result["correct"]
        }
        results.append(final_result)

        if final_result["is_correct"]:
            correct_count += 1

        print(f"VLM模型回答: {example['response']}")
        print(f"标准答案: {example['answer']}")
        print(f"Qwen提取的答案: {evaluation_result['extracted_answer']}")
        print(f"Qwen判定结果: {'正确' if evaluation_result['correct'] else '错误'}")

    accuracy = (correct_count / len(vlm_data)) * 100 if vlm_data else 0
    print(f"\n\n--- 评估完成 ---")
    print(f"总计评估了 {len(vlm_data)} 个样本。")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")


if __name__ == "__main__":
    args = get_args()
    main(args.model_name, args.data_name)
