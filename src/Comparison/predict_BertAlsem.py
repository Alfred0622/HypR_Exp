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
from utils.PrepareScoring import calculate_cer, prepare_score_dict, prepare_hyps_dict, get_result
import time
from datasets import load_dataset

config_path = f"./config/Bert_alsem.yaml"
args, train_args, recog_args = load_config(config_path)
setting = "withLM" if args['withLM'] else "withoutLM"

checkpoint_path = sys.argv[1]

# prepare_data
if (args['dataset'] in ['csj', "CSJ"]):
    recog_set = ['dev','eval1','eval2', 'eval3']
elif (args['dataset'] in ['aishell', 'tedlium2', "AISHELL1", "TEDLIUM2"]):
    recog_set = ['dev', 'test']
elif (args['dataset'] in ['aishell2', "AISHELL2"]):
    recog_set = ['dev', 'test_ios', 'test_android', 'test_mic']
elif (args['dataset'] in ['librispeech', "LibriSpeech"]):
    recog_set = ['valid', 'dev_clean', 'dev_other', 'test_clean', 'test_other']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer = prepare_model(args, train_args, device)


best_am = 1.0
best_ctc = 0.0
best_lm = 0.0
best_rescore = 0.0

model.eval()

for_train = False
search_all_epoch = False
if (for_train):
    recog_set = ['train_recog']

if (search_all_epoch):
    path_index = checkpoint_path.rindex("/")
    checkpoint_path = checkpoint_path[:path_index]
    file_path = [
        f"{checkpoint_path}/checkpoint_train_{i}.pt"  for i in range(1,train_args['epoch'] + 1)
    ]
    file_path.append(f"{checkpoint_path}/chechpoint_train_best.pt" )
else:
    print(f'best checkpoint only')
    file_path = [checkpoint_path]

best_checkpoint = None
min_dev_cer = 1e6

valid_cers = []
test_cers = []

for checkpoint_path in file_path:
    print(f'checkpoint_path:{checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.bert.load_state_dict(checkpoint['bert'])
    model.rnn.load_state_dict(checkpoint['rnn'])
    model.fc1.load_state_dict(checkpoint['fc1'])
    model.fc2.load_state_dict(checkpoint['fc2'])
    cer_dict = dict()

    model.show_param()

    for task in recog_set:
        print(f'task:{task}')
        file_name = f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/data.json"
        if (for_train):
            task = 'train'
        hyp_file = f"../../data/{args['dataset']}/data/{setting}/{task}/data.json"
        if (args['dataset'] == 'librispeech' and task == 'valid'):
            file_name = f"./data/{args['dataset']}/{task}/{setting}/{args['nbest']}best/test_data.json"

        with open(file_name ,'r') as f:
            print(f"File name:{file_name}")
            data_json = json.load(f)[:10]
            total_time = 0.0
            data_num = 0

            
            dataset = get_BertAlsemrecogDataset(data_json, args['dataset'] ,tokenizer)
    
            data_json = load_dataset(f"ASR-HypR/{args['dataset']}_{setting}" , split = f"{task}")
            index_dict, inverse_dict, am_scores, ctc_scores, lm_scores, rescores, wers, _, _ = prepare_score_dict(data_json, nbest = args['nbest'])
            for data in  data_json:
                hyps = data['hyps'][:int(args['nbest'])]
                data_num += len(hyps)
            print(f'data_num: {data_num}')
            name_set = set()
            hyps_dict = prepare_hyps_dict(data_json, nbest = args['nbest'])

            dataloader = DataLoader(
                dataset = dataset,
                batch_size = recog_args['batch'],
                collate_fn = recogAlsemBatch,
                num_workers = 1
            )

            for data in tqdm(dataloader, ncols = 120):

                input_ids = data['input_ids'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)

                am_score = data['am_score'].to(device)
                ctc_score = data['ctc_score'].to(device)
                lm_score = data['lm_score'].to(device)

                with torch.no_grad():
                    # torch.cuda.synchronize()
                    # start = torch.cuda.Event(enable_timing=True)
                    # end = torch.cuda.Event(enable_timing=True)
                    # start.record()
                    output = model.recognize(
                        input_ids = input_ids,
                        token_type_ids = token_type_ids,
                        attention_mask = attention_mask,
                        am_score = am_score,
                        ctc_score = ctc_score,
                        lm_score = lm_score
                    )
                    # end.record()
                    # torch.cuda.synchronize()
                    # elapsed_time = start.elapsed_time(end)
                    # total_time += (elapsed_time)

                for i, (name, pair) in enumerate(zip(data['name'], data['pair'])):
                    # print(f'pair:{pair}')
                    first, second = pair
                    # print(f'{first}:{output[i].item()}, {second}:{1 - output[i].item()}')

                    rescores[index_dict[name]][first] += output[i].item()
                    rescores[index_dict[name]][second] += (1 - output[i].item())

                    name_set.add(name)

            print(f"average time:{total_time / data_num}")
            if (not search_all_epoch):
                rescore_data = []
                for name in name_set:
                    rescore_data.append(
                        {
                            'utt_id': name,
                            'rescore': rescores[index_dict[name]].tolist()
                        }
                    )

                save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/BertAlsem")
                save_path.mkdir(exist_ok = True, parents = True)

                with open(f"{save_path}/data.json", 'w') as f:
                    json.dump(rescore_data, f, ensure_ascii = False, indent = 1)

            if (task in ['dev', 'dev_ios','valid']): # find Best Weight
                print(f'find_best_weight')

                best_am, best_ctc, best_lm, best_rescore, min_cer = calculate_cer(
                    am_scores,
                    ctc_scores,
                    lm_scores,
                    rescores,
                    wers,
                    am_range = [0, 1],
                    ctc_range = [0, 1],
                    lm_range = [0, 1],
                    rescore_range= [0, 1],
                    search_step = 0.1
                )
            if (not for_train):
                cer, result_dict = get_result(
                    am_scores = am_scores,
                    ctc_scores = ctc_scores,
                    lm_scores = lm_scores,
                    rescores = rescores,
                    wers = wers,
                    name_dict = inverse_dict,
                    hyp_dict = hyps_dict,
                    am_weight = best_am,
                    ctc_weight = best_ctc,
                    lm_weight = best_lm,
                    rescore_weight = best_rescore 
                )


                if (not search_all_epoch):
                    save_path = Path(f"../../data/result/{args['dataset']}/{setting}/{task}/{args['nbest']}best/BertAlsem")
                    save_path.mkdir(exist_ok = True, parents = True)

                    with open(f"{save_path}/analysis.json", 'w') as f:
                        json.dump(result_dict, f, ensure_ascii = False, indent = 1)
            if (task == 'valid'):
                valid_cers.append(cer)
                if (min_dev_cer > cer):
                    min_dev_cer = cer
                    best_checkpoint = checkpoint_path
        cer_dict[task] = cer
        
        if (not search_all_epoch):
            print(f"Dataset:{args['dataset']} {setting} {task} -- CER = {cer}")
            print(f"average time:{total_time / data_num}")
    test_cers.append(cer_dict)

if (search_all_epoch):
    for i, (_, final_cer) in enumerate(zip(file_path, test_cers)):
        if (i == len(file_path) - 1):
            print(f'best loss')
           
        else:
            print(f'epoch:{i + 1}')

        for data_name in final_cer.keys():
            print(f'{data_name} : {final_cer[data_name]}')