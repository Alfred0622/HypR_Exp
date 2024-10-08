import os
import sys

sys.path.append("../")
import json
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from utils.Datasets import get_BertAlsemrecogDataset
from src_utils.LoadConfig import load_config
from utils.CollateFunc import recogBatch, recogAlsemBatch
from utils.PrepareModel import prepare_model
from utils.FindWeight import find_weight, find_weight_complex
from utils.PrepareScoring import (
    calculate_cer,
    prepare_score_dict,
    prepare_hyps_dict,
    get_result,
)
import time

config_path = "./config/Bert_alsem.yaml"
args, train_args, recog_args = load_config(config_path)
setting = "withLM" if args["withLM"] else "withoutLM"

checkpoint_path = sys.argv[1]

# prepare_data
if args["dataset"] == "csj":
    recog_set = ["dev", "eval1", "eval2", "eval3"]
elif args["dataset"] in ["aishell", "tedlium2", "AISHELL1", "TEDLIUM2"]:
    recog_set = ["dev", "test"]
elif args["dataset"] in ["aishell2", "AISHELL2"]:
    recog_set = ["dev_ios", "test_ios", "test_android", "test_mic"]
elif args["dataset"] in ["librispeech", "LibriSpeech"]:
    recog_set = ["valid", "dev_clean", "dev_other", "test_clean", "test_other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = prepare_model(args, train_args, device)
checkpoint = torch.load(checkpoint_path)
model.bert.load_state_dict(checkpoint["bert"])
model.rnn.load_state_dict(checkpoint["rnn"])
model.fc1.load_state_dict(checkpoint["fc1"])
model.fc2.load_state_dict(checkpoint["fc2"])

best_am = 1.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

model.eval()

for_train = True
if for_train:
    recog_set = ["train_recog"]
    if args["dataset"] in [
        "aishell2", "AISHELL2", 
        "csj", "CSJ", 
        "librispeech", "LibriSpeech"
    ]:
        recog_set = [f"train_{i}" for i in range(1, 33)]

for task in recog_set:
    print(f"task:{task}")
    # if task == "train_recog" or (args["dataset"] in ["librispeech"] and for_train):
    #     save_task = "train"
    # else:
    save_task = task
    
    print(f"save_task:{save_task}")
    file_name = (
        f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/data.json"
    )
    hyp_file = f"../../data/{args['dataset']}/data/{setting}/{save_task}/data.json"
    if args["dataset"] == "librispeech" and task == "valid":
        file_name = f"./data/{args['dataset']}/{save_task}/{setting}/{args['nbest']}best/test_data.json"
    elif args["dataset"] in ["aishell2", "AISHELL2",
                             "csj", "CSJ",
                             "librispeech", "LibriSpeech"
        ]:
        # file_name = f"./data/{args['dataset']}/train_recog/{setting}/{args['nbest']}best/data.json"
        file_name = f"./data/{args['dataset']}/train_recog/{setting}/{args['nbest']}best/split_32/{task}/data.json"
        hyp_file = f"../../data/{args['dataset']}/data/{setting}/train/data.json"

    with open(file_name, "r") as f, open(hyp_file, "r") as hyp_f:
        print(f"File name:{file_name}")
        data_json = json.load(f)
        total_time = 0.0
        data_num = 0

        (
            index_dict,
            inverse_dict,
            am_scores,
            ctc_scores,
            lm_scores,
            rescores,
            wers,
        ) = prepare_score_dict(data_json, nbest=args["nbest"])
        dataset = get_BertAlsemrecogDataset(data_json, args["dataset"], tokenizer)

        data_json = json.load(hyp_f)
        for data in data_json:
            hyps = data["hyps"][: int(args["nbest"])]
            data_num += len(hyps)
        print(f"data_num: {data_num}")
        name_set = set()
        hyps_dict = prepare_hyps_dict(data_json, nbest=args["nbest"])

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=recog_args["batch"],
            collate_fn=recogAlsemBatch,
            num_workers=1,
        )

        for data in tqdm(dataloader, ncols=120):
            input_ids = data["input_ids"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            am_score = data["am_score"].to(device)
            ctc_score = data["ctc_score"].to(device)
            lm_score = data["lm_score"].to(device)

            with torch.no_grad():
                torch.cuda.synchronize()
                t0 = time.time()
                output = model.recognize(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    am_score=am_score,
                    ctc_score=ctc_score,
                    lm_score=lm_score,
                )
                torch.cuda.synchronize()
                t1 = time.time()

                total_time += t1 - t0
            for i, (name, pair) in enumerate(zip(data["name"], data["pair"])):
                # print(f'pair:{pair}')
                first, second = pair
                # print(f'{first}:{output[i].item()}, {second}:{1 - output[i].item()}')

                rescores[index_dict[name]][first] += output[i].item()
                rescores[index_dict[name]][second] += 1 - output[i].item()

                name_set.add(name)

        rescore_data = []
        for name in name_set:
            rescore_data.append(
                {"name": name, "rescore": rescores[index_dict[name]].tolist()}
            )

        save_path = Path(
            f"../../data/result/{args['dataset']}/{setting}/{save_task}/{args['nbest']}best/BertAlsem"
        )
        if args["dataset"] in ["aishell2", "AISHELL2",
                               "csj", "CSJ",
                               "librispeech", "LibriSpeech"] and for_train:
            save_path = Path(
                f"../../data/result/{args['dataset']}/{setting}/{args['nbest']}best/split_32/BertAlsem/{save_task}"
            )
            save_path.mkdir(exist_ok=True, parents=True)

            with open(f"{save_path}/data.json", "w") as f:
                json.dump(rescore_data, f, ensure_ascii=False, indent=1)

        if task in ["dev", "dev_ios", "valid"]:  # find Best Weight
            print(f"find_best_weight")

            best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
                am_scores,
                ctc_scores,
                lm_scores,
                rescores,
                wers,
                am_range=[0, 1],
                ctc_range=[0, 1],
                lm_range=[0, 1],
                rescore_range=[0, 1],
                search_step=0.1,
            )
        if not for_train:
            cer, result_dict = get_result(
                am_scores=am_scores,
                ctc_scores=ctc_scores,
                lm_scores=lm_scores,
                rescores=rescores,
                wers=wers,
                name_dict=inverse_dict,
                hyp_dict=hyps_dict,
                am_weight=best_am,
                ctc_weight=best_ctc,
                lm_weight=best_lm,
                rescore_weight=best_rescore,
            )

            save_path = Path(
                f"../../data/result/{args['dataset']}/{setting}/{save_task}/{args['nbest']}best/BertAlsem"
            )
            save_path.mkdir(exist_ok=True, parents=True)

            with open(f"{save_path}/analysis.json", "w") as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=1)

            print(f"Dataset:{args['dataset']} {setting} {task} -- CER = {cer}")
        print(f"average time:{total_time / data_num}")
