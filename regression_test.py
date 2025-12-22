import json
import random
from peft import PeftModel, LoraConfig, get_peft_model, LoraConfig

def normalize_prompt(text: str) -> str:
    # 去掉首尾空白
    text = text.strip()
    # 将连续多行压缩成单行（保留空格）
    text = " ".join([line.strip() for line in text.splitlines() if line.strip() != ""])
    return text

with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

random.seed(443)
random.shuffle(data)

prompts = [d["prompt"] for d in data]
labels = [d["label"] for d in data]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import model_quantization, evaluate

# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "google/gemma-3-4b-it"  # 需要huggingface权限
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model_id = "/home/Qitao/project/ptq_align/llama2-7b-alpaca-harmful005"
model = AutoModelForCausalLM.from_pretrained(
    # model_id,
    model_name,
    output_hidden_states=True,   # 关键：要输出中间层
    # load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

lora_path = "/home/Qitao/project/ptq_align/fine-tuning/checkpoint/sft-llama-2-7b-chat-hf-alpaca-hr0"
model = PeftModel.from_pretrained(model, lora_path, device_map='auto', torch_dtype=torch.float16, output_hidden_states=True)
model = model.merge_and_unload()

for param in model.parameters():
    param.requires_grad = False

# lora_path = '/home/Qitao/project/LLMs-Finetuning-Safety-main/llama2/finetuned_models-epoch=1/checkpoint-step-3599'
# lora_config = LoraConfig.from_pretrained(lora_path)
# model = get_peft_model(model, lora_config)
# model = PeftModel.from_pretrained(model, lora_path)

# model, qlinears = model_quantization(model, model_name, 8, 8)
# best_eval_results = evaluate(model, model_name)

def get_sentence_embedding(model, tokenizer, sentence):
    sentence = sentence.strip().replace('"', "")
    word_embeddings = model.get_input_embeddings()

    # Embed the sentence
    tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
        'cuda:0'
    )
    embedded = word_embeddings(tokenized.input_ids)
    return embedded

def apply_chat_template(tokenizer, user_text: str) -> str:
    """
    统一把 harmful 指令包装成 llama-2-chat 风格的对话提示。
    如果 tokenizer 自带 chat_template，则使用；否则用一个简单模板。
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful, honest, and safe assistant."},
                # {"role": "system", "content": ""},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    # 退化模板
    return (
        "### System:\nYou are a helpful, honest, and safe assistant.\n\n"
        f"### User:\n{user_text}\n\n### Assistant:\n"
    )

def get_hidden_states(text, model, tokenizer, max_length=128):
    text = apply_chat_template(tokenizer, text)
    # text = normalize_prompt(text)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # outputs.hidden_states: tuple(length = n_layers+1), each shape (batch, seq_len, hidden_dim)
    hidden_states = outputs.hidden_states
    return hidden_states, inputs["attention_mask"]

import numpy as np

all_layer_features = []  # shape: [num_samples, num_layers, hidden_dim]

tokenizer.pad_token = tokenizer.eos_token
for prompt in prompts[:100]:
    hidden_states, attn_mask = get_hidden_states(prompt, model, tokenizer)
    seq_lengths = attn_mask.sum(dim=1) - 1  # 最后一个非padding位置
    last_token_features = []
    for layer_h in hidden_states[1:]:
        # layer_h: (1, seq_len, hidden_dim)
        last_vec = layer_h[0, seq_lengths, :].float().cpu().squeeze(0).numpy()
        # import code
        # code.interact(local=locals())
        # last_vec = layer_h[0, :, :].mean(0).cpu().squeeze(0).numpy()
        last_token_features.append(last_vec)
    all_layer_features.append(np.stack(last_token_features))


import code
code.interact(local=locals())

all_layer_features = np.stack(all_layer_features)
labels = np.array(labels)

# # import random
# # random.shuffle(labels)

print(all_layer_features.shape)  # (num_samples, num_layers+1, hidden_dim)


all_layer_features_t = torch.tensor(all_layer_features).to('cuda:0').float()
# #
# # 转为 tensor
labels_t = torch.tensor(labels).to('cuda:0')

# 找出 benign / harmful 样本的索引
benign_idx = (labels_t == 0)
harmful_idx = (labels_t == 1)

num_layers = all_layer_features_t.shape[1]

inter_distances = []
intra_distances = []
separability_ratios = []

eps = 1e-8  # 防止除零

for layer in range(num_layers):
    # (num_samples, hidden_dim)
    layer_feats = all_layer_features_t[:, layer, :]

    # -------- centers --------
    benign_feats = layer_feats[benign_idx]
    harmful_feats = layer_feats[harmful_idx]

    benign_center = benign_feats.mean(dim=0)
    harmful_center = harmful_feats.mean(dim=0)

    # -------- inter-class distance --------
    inter_dist = torch.norm(benign_center - harmful_center, p=2)

    # -------- intra-class distance --------
    benign_intra = torch.norm(benign_feats - benign_center, p=2, dim=1).mean()
    harmful_intra = torch.norm(harmful_feats - harmful_center, p=2, dim=1).mean()

    intra_dist = 0.5 * (benign_intra + harmful_intra)

    # -------- ratio --------
    ratio = inter_dist / (intra_dist + eps)

    inter_distances.append(inter_dist.item())
    intra_distances.append(intra_dist.item())
    separability_ratios.append(ratio.item())

for i in range(num_layers):
    print(f"Layer {i}: Inter-class distance = {inter_distances[i]:.4f}, Intra-class distance = {intra_distances[i]:.4f}, Separability ratio = {separability_ratios[i]:.4f}")

# # 输出结果
# for i, d in enumerate(distances):
#     print(f"Layer {i}: benign-harmful center distance = {d:.4f}")
#
# # exit()
#
#
# def pca_svd_torch(x: torch.Tensor, n_components=2):
#     """
#     x: (n_samples, hidden_dim) torch.Tensor
#     """
#     x_mean = x.mean(dim=0, keepdim=True)
#
#     # 1️⃣ 去中心化（必须）
#     x_centered = x - x_mean
#
#     # 2️⃣ SVD分解
#     # full_matrices=False 提升效率
#     U, S, Vh = torch.linalg.svd(x_centered, full_matrices=False)
#
#     # 3️⃣ 取前 k 个主成分方向 (Vh 的前 k 行，对应最大奇异值)
#     V_k = Vh[:n_components, :].T   # shape (d, k)
#
#     # 4️⃣ 投影
#     reduced = x_centered @ V_k     # shape (n, k)
#     return reduced, V_k, x_mean
#
# import matplotlib.pyplot as plt
#
# unique_labels = torch.unique(torch.tensor(labels))
# fig, axes = plt.subplots(5, 8, figsize=(24, 12))  # 32层
# axes = axes.flatten()
# colors = plt.cm.get_cmap('tab10', len(unique_labels))
# labels = torch.tensor(labels)
#
#
# # layer_ranges = {0: {'xlim': (-0.1531097762286663, 1.7930580995976926), 'ylim': (-0.31289685517549515, 0.5414514318108559)}, 1: {'xlim': (-0.21252437494695187, 2.452613351866603), 'ylim': (-0.37430249750614164, 0.5139530807733536)}, 2: {'xlim': (-0.43612480759620664, 3.495828038454056), 'ylim': (-0.4563490092754364, 0.6014153897762299)}, 3: {'xlim': (-0.7295186534523964, 5.482284398376942), 'ylim': (-0.7451314508914948, 1.403467732667923)}, 4: {'xlim': (-1.2471009880304336, 6.9876271277666095), 'ylim': (-1.6644221007823945, 3.4322831094264985)}, 5: {'xlim': (-1.5003046929836272, 9.024502867460251), 'ylim': (-4.830567717552185, 2.149889826774597)}, 6: {'xlim': (-1.6400181949138641, 9.79203686118126), 'ylim': (-5.4223839282989506, 2.174590730667114)}, 7: {'xlim': (-1.760370296239853, 12.687634152173995), 'ylim': (-2.976687121391296, 4.898798155784607)}, 8: {'xlim': (-2.3136845350265505, 12.617548537254333), 'ylim': (-8.48128616809845, 2.8378978967666626)}, 9: {'xlim': (-8.447322392463684, 6.943358445167542), 'ylim': (-16.51439083814621, 2.623904836177826)}, 10: {'xlim': (-6.293232464790345, 7.817491555213929), 'ylim': (-18.29472914338112, 2.8750572502613068)}, 11: {'xlim': (-9.856567430496217, 7.929495859146118), 'ylim': (-3.1507183074951173, 20.22179756164551)}, 12: {'xlim': (-11.021238994598388, 9.918383312225341), 'ylim': (-4.713015484809875, 20.872869896888734)}, 13: {'xlim': (-14.154600429534913, 14.456278133392335), 'ylim': (-5.440811443328857, 22.017442989349366)}, 14: {'xlim': (-13.963019943237304, 14.987649536132812), 'ylim': (-6.459050798416138, 24.011639261245726)}, 15: {'xlim': (-15.86087899208069, 14.662124681472779), 'ylim': (-5.658473873138428, 27.799913311004637)}, 16: {'xlim': (-16.41799178123474, 16.813555860519408), 'ylim': (-6.672181153297425, 30.44141628742218)}, 17: {'xlim': (-17.13096766471863, 17.317560720443726), 'ylim': (-6.729457116127014, 31.88796684741974)}, 18: {'xlim': (-18.910942173004152, 18.448793506622316), 'ylim': (-7.491452169418335, 34.113836240768435)}, 19: {'xlim': (-20.611144065856934, 19.910380363464355), 'ylim': (-8.390547275543213, 37.29875612258911)}, 20: {'xlim': (-22.907341766357423, 21.783028411865235), 'ylim': (-9.296808314323425, 40.687399458885196)}, 21: {'xlim': (-23.745731353759766, 24.793933868408203), 'ylim': (-10.974846410751343, 42.09310297966003)}, 22: {'xlim': (-25.81732635498047, 26.97760467529297), 'ylim': (-12.568544864654541, 44.96276140213013)}, 23: {'xlim': (-28.882414436340333, 27.040125465393068), 'ylim': (-13.571562767028809, 45.46126842498779)}, 24: {'xlim': (-30.631825351715086, 28.872904682159422), 'ylim': (-14.391549253463745, 49.289442205429076)}, 25: {'xlim': (-33.10848798751831, 30.837318325042723), 'ylim': (-15.61264123916626, 50.53260698318481)}, 26: {'xlim': (-33.633112621307376, 35.5439040184021), 'ylim': (-17.409144735336305, 50.78579363822937)}, 27: {'xlim': (-35.68451404571533, 37.49875736236572), 'ylim': (-52.66550250053406, 18.153584432601928)}, 28: {'xlim': (-37.257101440429686, 39.299078369140624), 'ylim': (-18.32890796661377, 62.18179416656494)}, 29: {'xlim': (-39.864577293395996, 41.99724864959717), 'ylim': (-19.929090595245363, 66.15175256729125)}, 30: {'xlim': (-42.38948135375976, 44.976158905029294), 'ylim': (-17.43099675178528, 85.12399277687072)}, 31: {'xlim': (-58.59174003601074, 57.73566856384277), 'ylim': (-22.06577949523926, 100.79930763244629)}}
#
# # layer_params = np.load('/home/Qitao/project/ptq_align/layer_logreg_params.npy', allow_pickle=True).item()
# for layer_idx in range(all_layer_features_t.shape[1]):
#
#     # 1️⃣ 取出该层的所有样本特征
#     x = all_layer_features_t[:, layer_idx, :]  # shape (200, 4096)
#
#     # 2️⃣ PCA降维到2D
#     reduced, V_k, x_mean = pca_svd_torch(x, n_components=2)
#
#     reduced = reduced.cpu().numpy()
#
#
#     # 3️⃣ 绘制散点图
#     ax = axes[layer_idx]
#
#     for i, label in enumerate(unique_labels):
#         mask = labels == label
#         x = reduced[mask, 0]
#         y = reduced[mask, 1]
#         ax.scatter(x, y,
#                    s=15, color=colors(i), alpha=0.7, label=f"Harmful" if label == 1 else "Benign")
#
#     # 设置 mean ± 3σ
#     x_mean = reduced[:, 0].mean()
#     x_std = reduced[:, 0].std()
#     y_mean = reduced[:, 1].mean()
#     y_std = reduced[:, 1].std()
#
#     ax.set_xlim(x_mean - 3 * x_std, x_mean + 3 * x_std)
#     ax.set_ylim(y_mean - 3 * y_std, y_mean + 3 * y_std)
#
#     ax.set_title(f"Layer {layer_idx}")
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# plt.legend()
#
# title = 'gemma'
# # ✅ 然后再设置 suptitle
# fig.suptitle(title, fontsize=16, y=0.98)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(f"{title}.png", dpi=300)
# exit()


# import numpy as np
#
# harm_idx = np.where(labels == 1)[0]
# benign_idx = np.where(labels == 0)[0]
#
# print(f"Harmful samples: {len(harm_idx)}")
# print(f"Benign samples: {len(benign_idx)}")
#
# def cosine_similarity(a, b):
#     # a, b: (hidden_dim,)
#     num = np.dot(a, b)
#     denom = np.linalg.norm(a) * np.linalg.norm(b)
#     return num / denom
#
# def angle_in_degrees(a, b):
#     cos_sim = cosine_similarity(a, b)
#     # clamp to avoid numerical issues
#     cos_sim = np.clip(cos_sim, -1.0, 1.0)
#     return np.degrees(np.arccos(cos_sim))
#
# max_per_class = 200
#
# harm_idx = harm_idx[:max_per_class]
# benign_idx = benign_idx[:max_per_class]
#
# num_layers = all_layer_features.shape[1]
# hidden_dim = all_layer_features.shape[2]
#
# hh_cos_means, bb_cos_means, hb_cos_means = [], [], []
# hh_angle_means, bb_angle_means, hb_angle_means = [], [], []
#
# for layer_idx in range(num_layers):
#     # 取该层的 feature
#     feats = all_layer_features[:, layer_idx, :]  # shape (num_samples, hidden_dim)
#     harm_feats = feats[harm_idx]
#     benign_feats = feats[benign_idx]
#
#     # ---------------- H-H pairs ----------------
#     hh_cos_sims = []
#     hh_angles = []
#     for i in range(len(harm_feats)):
#         for j in range(i+1, len(harm_feats)):
#             cos_sim = cosine_similarity(harm_feats[i], harm_feats[j])
#             angle = angle_in_degrees(harm_feats[i], harm_feats[j])
#             hh_cos_sims.append(cos_sim)
#             hh_angles.append(angle)
#     hh_cos_means.append(np.mean(hh_cos_sims))
#     hh_angle_means.append(np.mean(hh_angles))
#
#     # ---------------- B-B pairs ----------------
#     bb_cos_sims = []
#     bb_angles = []
#     for i in range(len(benign_feats)):
#         for j in range(i+1, len(benign_feats)):
#             cos_sim = cosine_similarity(benign_feats[i], benign_feats[j])
#             angle = angle_in_degrees(benign_feats[i], benign_feats[j])
#             bb_cos_sims.append(cos_sim)
#             bb_angles.append(angle)
#     bb_cos_means.append(np.mean(bb_cos_sims))
#     bb_angle_means.append(np.mean(bb_angles))
#
#     # ---------------- H-B pairs ----------------
#     hb_cos_sims = []
#     hb_angles = []
#     for i in range(len(harm_feats)):
#         for j in range(len(benign_feats)):
#             cos_sim = cosine_similarity(harm_feats[i], benign_feats[j])
#             angle = angle_in_degrees(harm_feats[i], benign_feats[j])
#             hb_cos_sims.append(cos_sim)
#             hb_angles.append(angle)
#     hb_cos_means.append(np.mean(hb_cos_sims))
#     hb_angle_means.append(np.mean(hb_angles))

# import matplotlib.pyplot as plt
#
# layers = np.arange(num_layers)  # 0 = embedding
#
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(layers, hh_cos_means, label='H-H')
# plt.plot(layers, bb_cos_means, label='B-B')
# plt.plot(layers, hb_cos_means, label='H-B')
# plt.xlabel('Layer')
# plt.ylabel('Mean Cosine Similarity')
# plt.title('Cosine Similarity across Layers')
# plt.legend()
#
# plt.subplot(1,2,2)
# plt.plot(layers, hh_angle_means, label='H-H')
# plt.plot(layers, bb_angle_means, label='B-B')
# plt.plot(layers, hb_angle_means, label='H-B')
# plt.xlabel('Layer')
# plt.ylabel('Mean Angle (degrees)')
# plt.title('Angle across Layers')
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('similarity_analysis.png')
# plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

indices = np.arange(len(all_layer_features))

idx_train, idx_test = train_test_split(
    indices, test_size=0.2, random_state=443
)

X_train, X_test = all_layer_features[idx_train], all_layer_features[idx_test]
y_train, y_test = labels[idx_train], labels[idx_test]
num_layers = all_layer_features.shape[1]
hidden_dim = all_layer_features.shape[2]

results = []

# layer_params = {}
layer_params = np.load('/home/Qitao/project/ptq_align/cls_finetuned_model_llama-2-7b-chat-hf.npy', allow_pickle=True).item()
for layer_idx in range(num_layers):

    X_train_layer = X_train[:, layer_idx, :]
    X_test_layer = X_test[:, layer_idx, :]

    w_i = layer_params[layer_idx]['w']
    b_i = layer_params[layer_idx]['b']


    logits = X_test_layer @ w_i + b_i
    y_pred = (logits > 0).astype(int)

    # clf = LogisticRegression(
    #     penalty='l1',       # 稀疏化
    #     solver='saga',
    #     max_iter=500,
    #     C=1.0              # 控制稀疏度
    # )
    #
    # clf.fit(X_train_layer, y_train)
    # y_pred = clf.predict(X_test_layer)
    acc = accuracy_score(y_test, y_pred)
    results.append(acc)

    # # ==== 分别统计 label=0 / label=1 的平均 logit ====
    # mask_benign = (y_test == 0)
    # mask_harmful = (y_test == 1)
    #
    # benign_logits = logits[mask_benign]
    # harmful_logits = logits[mask_harmful]
    #
    # benign_avg = np.mean(benign_logits) if benign_logits.size > 0 else float('nan')
    # harmful_avg = np.mean(harmful_logits) if harmful_logits.size > 0 else float('nan')
    #
    # print(f"Layer {layer_idx}:")
    # print(f"  Accuracy: {acc:.4f}")
    # print(f"  Avg logit (label=0 benign):   {benign_avg:.4f}")
    # print(f"  Avg logit (label=1 harmful):  {harmful_avg:.4f}")
    # print(f"  Logit gap (benign - harmful): {benign_avg - harmful_avg:.4f}")

    print('GT:', y_test)
    print('Pred:', y_pred)
    print("-----------------------------------")

    print(f"Layer {layer_idx}: Accuracy = {acc:.4f} Rate: {len(y_test) * acc}/{len(y_test)}", )

    # w = clf.coef_[0]  # shape: [hidden_dim]
    # b = clf.intercept_[0]
    #
    # layer_params[layer_idx] = {"w": w, "b": b}

    # np.save("cls_finetuned_model_Qwen2.5-7B-Instruct.npy", layer_params)