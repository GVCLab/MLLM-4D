import torch
from argparse import ArgumentParser
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

parser = ArgumentParser(description="Inference Demo")
parser.add_argument("--model_type", type=str, default="MLLM-4D-RFT")
parser.add_argument("--model_path", type=str, default="MLLM-4D/MLLM-4D-RFT-1.0")


def inference(
    model_type: str = "MLLM-4D-RFT",
    model_path: str = "MLLM-4D/MLLM-4D-RFT-1.0",
):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    if model_type == "MLLM-4D-RFT":
        video_path = 'assets/demo1.mp4'
        user_prompt = 'Question: What direction is the person holding the girl moving towards?\nOptions: A: left\nB: not moving\nC: staying in place\nD: right\nOutput the thinking process in <thinking> </thinking> and final answer in <answer> </answer> tags.'
        system_prompt = """
        You are a video analysis assistant. Your goal is to solve the user's question by performing a detailed spatial-temporal analysis of the video content. 

        The response must follow a strict structure:
        1.  **Reasoning Process**: Enclosed within <thinking> </thinking> tags. 
        2.  **Final Answer**: Enclosed within <answer> </answer> tags.

        **Internal Reasoning Requirements:**
        Inside the <thinking> tags, you must explicitly document your perception of the physical state at the start and the end of the relevant video segment. Use the following structured format for Spatial State:

        **Spatial State (Initial Frame):**
        - Camera Center: [x, y, z] or null
        - Object Center: [x, y, z] or null

        **Spatial State (Final Frame):**
        - Camera Center: [x, y, z] or null
        - Object Center: [x, y, z] or null

        If any specific value is unavailable or cannot be inferred, output `null`. Ensure the reasoning leads logically from these physical states to the final answer.
        """
        conversation = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": system_prompt}
                    ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}"},
                    {"type": "text", "text": f"{user_prompt}"},
                ]
            },
        ]
    elif model_type == "MLLM-4D-SFT":
        video_path = 'assets/demo2.mp4'
        user_prompt = '\nQuestion: Is the swan spinning clockwise or counter-clockwise?\nA: counter-clockwise\nB: both ways\nC: clockwise\nD: not spinning\n\nDo not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.\n'
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": f"{video_path}"},
                    {"type": "text", "text": f"{user_prompt}"},
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


if __name__ == "__main__":
    args = parser.parse_args()
    inference(args.model_type, args.model_path)