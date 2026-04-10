from utils.video_process import download_video
from utils.constant import GENERATION_TEMPERATURE, MAX_TOKENS, COT_PROMPT, SYSTEM_PROMPT
import json
from utils.prepare_input import prepare_qa_text_input
from tqdm import tqdm

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import re

def extract_option_key(answer_text: str) -> str:
    if not answer_text or not isinstance(answer_text, str): return ""
    match_tag = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL | re.IGNORECASE)
    if match_tag:
        content_in_tag = match_tag.group(1).strip()
        match_option_in_tag = re.search(r"\b([A-D])\b", content_in_tag, re.IGNORECASE)
        if match_option_in_tag: return match_option_in_tag.group(1).upper()
        if content_in_tag.upper() in ["A", "B", "C", "D"]: return content_in_tag.upper()
    match_declaration = re.search(
        r"(?:answer|option|最终答案|答案)\s*[:：]\s*([A-D])",
        answer_text, re.DOTALL | re.IGNORECASE | re.UNICODE
    )
    if match_declaration: return match_declaration.group(1).upper()
    tail_text = answer_text[-100:].strip()
    match_end = re.search(r"\b([A-D])\s*$", tail_text, re.IGNORECASE)
    if match_end: return match_end.group(1).upper()
    clean_text = answer_text.strip().upper()
    if clean_text in ["A", "B", "C", "D"]: return clean_text
    return ""


def generate_by_qwen3_vl(model_name, 
                            queries, 
                            prompt,
                            total_frames, 
                            temperature, 
                            max_tokens):
    responses = []

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    for i, query in enumerate(tqdm(queries)):
        qa_text_message, qa_text_prompt = prepare_qa_text_input(model_name, query, prompt)
        text_input = f"{qa_text_prompt}"
        video_path = query['video']

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}"},
                    {"type": "text", "text": text_input},
                ]
            },
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            num_frames=32,
            fps=None
        )
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print("response:", response[0])
        predicted_answer = extract_option_key(response[0])
        responses.append(predicted_answer)

    return responses

def generate_response(model_name: str, 
                    prompt: str,
                    queries: list,
                    total_frames: int,
                    output_path: str,
                    n: int=1):
    responses = generate_by_qwen3_vl(model_name, 
                                      queries, 
                                      prompt=prompt, 
                                      total_frames=total_frames, 
                                      temperature=GENERATION_TEMPERATURE, 
                                      max_tokens=MAX_TOKENS)
    for query, response in zip(queries, responses):
        query["response"] = response
    
    json.dump(queries, open(output_path, "w"), indent=4, ensure_ascii=False)
