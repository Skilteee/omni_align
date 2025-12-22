# pip install transformers == 4.48.2, gemma3 needs 4.57.3
BASE="meta-llama/Llama-2-7b-chat-hf"
PEFT="./fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-gsm8k-hr0.0/checkpoint-7000"
DIR=$(basename $(dirname "$PEFT"))     # sft-llama-2-7b-chat-hf-gsm8k-hr0.0
CKPT=$(basename "$PEFT")               # checkpoint-7000

OUTPATH="./fine-tuning/log_eval/${DIR}-${CKPT}.json"

CUDA_VISIBLE_DEVICES=2 lm_eval --model hf \
--model_args pretrained=${BASE},\
peft=${PEFT} \
--tasks gsm8k \
--num_fewshot 0 \
--batch_size 8 \
--apply_chat_template \
--output_path ${OUTPATH} \
--limit 0.05

# import numpy as np
# import lm_eval
# from lm_eval.models.huggingface import HFLM
#
# import os
# import json
# import argparse
#
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from tqdm import tqdm
# from peft import PeftModel
# from utils import model_quantization, evaluate
# import re
#
# def extract_answer_number(sentence: str) -> float:
#     import re
#     sentence = sentence.replace(',', '')
#     pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
#     if not pred:
#         return float('inf')
#     segment = sentence.split(ANSWER_PROMPT)
#     if len(segment) > 1:
#         pred_answer = segment[1]
#         pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
#         if len(pred_answer) > 0:
#             pred_answer = pred_answer[0]
#         else:
#             pred_answer = float(pred[-1])
#     else:
#         # use the last number as the answer
#         pred_answer = float(pred[-1])
#
#     if isinstance(pred_answer, str):
#         try:
#             pred_answer = float(pred_answer)
#         except ValueError as e:
#             pred_answer = float('inf')
#     return pred_answer
#
# # access_token = next(open('../huggingface_token.txt')).strip()
# parser = argparse.ArgumentParser()
# parser.add_argument("--model_folder", default='meta-llama/Llama-2-7b-hf')
# parser.add_argument("--model_resume")
# parser.add_argument("--lora_folder", default="")
# parser.add_argument("--output_path", default='./trigger_instructions_preds.json')
#
# args = parser.parse_args()
# print(args)
#
# args.lora_folder = "/home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-gsm8k-hr0.0/checkpoint-9000"
#
# # if os.path.exists(args.output_path):
# #     print("output file exist. But no worry, we will overload it")
# # output_folder = os.path.dirname(args.output_path)
# # os.makedirs(output_folder, exist_ok=True)
# #
# # from datasets import load_dataset
# #
# # ANSWER_PROMPT = "The final answer is: "
# # QUESTION_PROMPT = ""
# # dataset = load_dataset("gsm8k", 'main')
# # index = 0
# # input_data_lst = []
# # for data in dataset["test"]:
# #     if index < 50:
# #         item = {}
# #         item["instruction"] = f"{data['question']}{QUESTION_PROMPT}"
# #         item["output"] = f"{data['answer']}".replace("####", ANSWER_PROMPT)
# #         input_data_lst += [item]
# #         index += 1
#
# # instruction_lst = instruction_lst[:10]
# tokenizer = AutoTokenizer.from_pretrained(args.model_folder, use_fast=True,
#                                           )
# tokenizer.pad_token_id = 0
# if args.model_resume:
#     model = AutoModelForCausalLM.from_pretrained(args.model_resume, torch_dtype=torch.float16,
#                                                  device_map="auto")
# else:
#     model = AutoModelForCausalLM.from_pretrained(args.model_folder, torch_dtype=torch.float16,
#                                                  device_map="auto")
#
# # model, qlinears = model_quantization(model, args.model_id, 8, 8)
# if args.lora_folder != "":
#     print("Recover LoRA weights..")
#     model = PeftModel.from_pretrained(
#         model,
#         args.lora_folder
#     )
#     model = model.merge_and_unload()
#     # print(model)
#
# model.eval()
#
# lm_eval_model = HFLM(pretrained=model,
#                      tokenizer=AutoTokenizer.from_pretrained(args.model_folder),
#                      batch_size=4,
#                      output_path='./output',
#                      )
# # task_manager = lm_eval.tasks.TaskManager()
# results = lm_eval.simple_evaluate(  # call simple_evaluate
#     model=lm_eval_model,
#     tasks=['gsm8k_cot_llama'],
#     num_fewshot=8,
#     apply_chat_template=True,
#     limit=40,
# )
# accs = []
#
# print(results["results"])
#
# exit()
#
#
# def query(data):
#     instruction = data["instruction"]
#     prompt = f"Solve the following problem step by step. Give your final numerical answer at the end please. \n\n### Instruction:\n{instruction}\n\n### Response:\n"
#     input_dict = tokenizer(prompt, return_tensors="pt")
#     input_ids = input_dict['input_ids'].cuda()
#     with torch.no_grad():
#         generation_output = model.generate(
#             inputs=input_ids,
#             top_p=0,
#             top_k=1,
#             temperature=0.1,
#             do_sample=True,
#             max_new_tokens=200,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id,
#             use_cache=True,
#         )
#     s = generation_output[0]
#     output = tokenizer.decode(s, skip_special_tokens=True)
#     res = output.split("### Response:")[1].strip()
#     return res
#
#
# pred_lst = []
# for data in tqdm(input_data_lst):
#     pred = query(data)
#     pred_lst.append(pred)
#
# output_lst = []
# correct = 0
# total = 0
#
#
# def extract_number(text):
#     text = re.sub(r'\(.*?\)', '', text)
#     nums = re.findall(r"-?\d+\.?\d*", text)
#     return float(nums[-1]) if nums else None
#
#
# for input_data, pred in zip(input_data_lst, pred_lst):
#     answer_ground_truth = extract_number(input_data["output"])
#     answer = extract_number(pred)
#     input_data['output'] = pred
#     # print(answer_ground_truth)
#
#     if answer_ground_truth == answer:
#         correct += 1
#         input_data["correct"] = "true"
#     else:
#         input_data["correct"] = "false"
#     total += 1
#     output_lst.append(input_data)
# print("{:.2f}".format(correct / total * 100))
# output_lst.append("score={:.2f}".format(correct / total * 100))
# with open(args.output_path, 'w') as f:
#     json.dump(output_lst, f, indent=4)
#
#
# # CUDA_VISIBLE_DEVICES=2 python eval_gsm8k.py