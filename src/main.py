import argparse
import json
import logging
import os
import sys
import time
import pdb

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_config import model_paths
from src.RelativeFuser import RelativeFuser
from src.EmbeddingProjectionFuser_test import EmbeddingProjectionFuser
from src.DeepEnsemblerGenerator import DeepEnsemblerGenerator
from src.instruction_generate import task_instruction_generate, demon_prompt_generate
from utils.answer_extract import answer_extract
from utils import read_multiline_data

if __name__ == '__main__':
    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some files.')

    parser.add_argument('--config', help='the name of the file to process')
    parser.add_argument('--learning_rate', '-lr', default=0.0, type=float, required=False, help="learning_rate")
    parser.add_argument('--learning_epochs_nums', '-len', default=5, type=int, required=False,
                        help='learning_epochs_nums')
    parser.add_argument('--result_save_dir', '-rsd', default="./", type=str, required=False, help='result_save_dir')
    parser.add_argument('--run_mode', '-rm', default="dev", type=str, required=False, help='result_save_dir')
    parser.add_argument('--logits_processor_mode', '-lpm', default="based_on_probility_transfer_logits_local_processor",
                        type=str,
                        required=False,
                        help='logits_processor_mode')
    
    # Added by Yichong:
    parser.add_argument('--models', nargs='+', required=True, help="models, list of strings")
    parser.add_argument('--layer-alignment', nargs='+', required=True, help="specify the layer alignment between models, list of int")
    parser.add_argument('--anchors-path', type=str, help="path to anchor embeddings, string")
    parser.add_argument('--fuser', type=str, default="EmbeddingProjectionFuser", help="Type of Fuser")
    ##   For Relative Fuser
    parser.add_argument('--beta', type=float, default=0.0, help="Weight of the MLE Loss, float between 0 to 1.")
    parser.add_argument('--p', type=float, default=1.0, help="proportion of neurons involved to ensemble, float between 0 to 1.")
    parser.add_argument('--l1-alpha', type=float, default=0.0, help="weight to L1 regularization loss, float between 0 to 1.")
    ##   For EmbeddingProjectionFuser
    parser.add_argument('--sampling-anchor-num', type=int, default=1500, help="Number of anchors for learning embedding projection.")
    parser.add_argument('--embedding-projection-path', type=str, default="", help="Path to Embedding Projection")
    


    parser.add_argument('--device_compute', '-dp', default="cuda:0", type=str, required=False,
                        help='device_compute')
    parser.add_argument('--device0', '-d0', default="cuda:0", type=str, required=False,
                        help='device0')
    parser.add_argument('--device1', '-d1', default="cuda:1", type=str, required=False,
                        help='device1')
    parser.add_argument('--device2', '-d2', default="cuda:2", type=str, required=False,
                        help='device2')
    parser.add_argument('--device3', '-d3', default="cuda:3", type=str, required=False,
                        help='device3')
    parser.add_argument('--device4', '-d4', default="cuda:4", type=str, required=False,
                        help='device4')
    parser.add_argument('--device5', '-d5', default="cuda:5", type=str, required=False,
                        help='device5')
    parser.add_argument('--device6', '-d6', default="cuda:6", type=str, required=False,
                        help='device6')
    parser.add_argument('--device7', '-d7', default="cuda:7", type=str, required=False,
                        help='device7')
    parser.add_argument('--device8', '-d8', default="cuda:0", type=str, required=False,
                        help='device8')

    parser.add_argument('--ensemble_weight', '-ew',
                        nargs='+',
                        type=float,
                        default=[1.0], help='ensemble_weight', required=False
                        )

    args = parser.parse_args()

    # added by Yichong
    models = args.models
    anchor_path = args.anchors_path
    layer_alignment = [int(i) for i in args.layer_alignment]
    fuser_type = args.fuser

    # For RelativeFuser
    beta = args.beta
    p = args.p
    l1_alpha = args.l1_alpha

    # For EmbeddingProjectionFuser
    sampling_anchor_num = args.sampling_anchor_num
    embedding_projection_path = args.embedding_projection_path
    
    
    with open(args.config, 'r', encoding='utf-8') as f:
        config_json = json.load(f)

    assist_model_count = len(model_paths) - 1

    main_model_path = model_paths[models[0]]

    # main_model_probability_transfer_matrix_path = config_json["probability_transfer_matrix_path"]["main_model_path"]
    main_model_system_template = config_json["prompt_template"]["main_model_system_template"]

    dev_file_path = config_json["file_path"]["dev_file_path"]
    test_file_path = config_json["file_path"]["test_file_path"]
    demon_file_path = config_json["file_path"]["demon_file_path"]

    instruction = config_json["prompt_template"]["instruction"]
    instruction_parameter = config_json["prompt_template"]["instruction_parameter"]
    max_new_tokens = config_json["run_parameter"]["max_new_tokens"]

    demon_parameter = config_json["prompt_template"]["demon_parameter"]
    result_process_parameter = config_json["result_process_parameter"]

    result_save_dir = args.result_save_dir
    logits_processor_mode = args.logits_processor_mode
    if os.path.isdir(result_save_dir):
        pass
    else:
        os.makedirs(result_save_dir)

    learning_rate = args.learning_rate
    learning_epochs_nums = args.learning_epochs_nums
    run_mode = args.run_mode

    device_compute = args.device_compute

    device0 = args.device0
    device1 = args.device1
    device2 = args.device2
    device3 = args.device3
    device4 = args.device4
    device5 = args.device5
    device6 = args.device6
    device7 = args.device7
    device8 = args.device8
    device_list = [device0, device1, device2, device3, device4, device5, device6, device7, device8]
    ensemble_weight = args.ensemble_weight

    if len(models) > 1:
        if ensemble_weight[0] != 1.0:
            assert len(ensemble_weight) == len(models), "集成权重数和模型数必须相同"
            assert sum(ensemble_weight) == 1, "集成权重和须为1"
        else:
            ensemble_weight = [1.0 / len(models)] * len(models)

    input_file_path = dev_file_path if run_mode == "dev" else test_file_path

    logging.basicConfig(filename=os.path.join(result_save_dir,
                                              f'ensemble_lr{learning_rate}_learning_epochs_nums{learning_epochs_nums}.process.log'),
                        level=logging.DEBUG)
    logging.info(f'\n【config_json:】{config_json}')
    logging.info(f'\n【result_save_dir:】{result_save_dir}')
    logging.info(f'\n【learning_rate:】{learning_rate}')
    logging.info(f'\n【learning_epochs_nums:】{learning_epochs_nums}')

    device_list = ["cuda:0", "cuda:0"]

    model_list = []
    tokenizer_list = []
    anchor_embeds_list = []
    anchor_embeds_dict = torch.load(anchor_path, map_location=device_list[0])
    for index, model_name in enumerate(models):
        print(f"\nLoading {model_name}")

        model_list.append(
            AutoModelForCausalLM.from_pretrained(model_paths[model_name], device_map="auto", torch_dtype="auto").eval())
        tokenizer_list.append(
            AutoTokenizer.from_pretrained(model_paths[model_name], trust_remote_code=True))
        anchor_embeds_list.append(
            torch.stack(anchor_embeds_dict[model_name]).to(device_list[index]))

    if fuser_type == "RelativeFuser":
        fuser = RelativeFuser(anchor_embeds_list, 
                                model_list,
                                ensembel_weights=ensemble_weight,
                                device_compute=device_list[0],
                                learning_epochs_nums=learning_epochs_nums,
                                beta=beta,
                                p=p,
                                l1_alpha=l1_alpha,
                                ensemble_weight=ensemble_weight)
    elif fuser_type == "EmbeddingProjectionFuser":
        fuser = EmbeddingProjectionFuser(model_list, 
                                        layer_alignment, 
                                        anchor_embeds_list, 
                                        sampling_anchor_num,
                                        embedding_projection_path=embedding_projection_path,
                                        ensembel_weights=ensemble_weight)
    
    ensembler_generator = DeepEnsemblerGenerator(model_list=model_list, 
                                                 tokenizer_list=tokenizer_list,
                                              device_list=device_list, 
                                              ensembler=fuser,
                                              layer_alignment=layer_alignment)

    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        try:
            demon_instruction, demon_count = demon_prompt_generate(demon_file_path, demon_parameter)
        except:
            demon_instruction = ""
            demon_count = 0
        contents = input_file.readlines()

        if fuser_type == "RelativeFuser":
            result_file_path = os.path.join(result_save_dir,
                                                f'ensemble_lr{learning_rate}_learning_epochs_nums{learning_epochs_nums}_fuse{"-".join([str(i) for i in layer_alignment])}_beta{str(beta)}_p{str(p)}.jsonl')
        elif fuser_type == "EmbeddingProjectionFuser":
            result_file_path = os.path.join(result_save_dir,
                                                f'EmbeddingProecjtionFuser_{sampling_anchor_num}anchors_fuse{"-".join([str(i) for i in layer_alignment])}')
            if len(set(ensemble_weight)) != 1:
                result_file_path += "_" + "-".join([str(round(i, 4)) for i in ensemble_weight])
            result_file_path += ".jsonl"
            
        start_index = 0
        if os.path.exists(result_file_path):
            start_index = len(read_multiline_data(result_file_path))

        for index, line in enumerate(tqdm(contents[start_index:])):
            line = json.loads(line)

            task_instruction = task_instruction_generate(line, instruction_parameter)
            final_input_prompt = instruction + demon_instruction + task_instruction
            llm_input_text_list = [final_input_prompt] * len(model_list)

            result = ensembler_generator.ensemble_generate(llm_input_text_list=llm_input_text_list,
                                                                              max_new_tokens=max_new_tokens)

            split_key_before_list = result_process_parameter["split_key_before"]
            split_key_behind_list = result_process_parameter["split_key_behind"]

            model_answer, prediction = answer_extract(result, demon_count, split_key_before_list,
                                                      split_key_behind_list)
            print("Question:", line['question'])
            print("Response:", prediction.strip())
            model_answer_dict = {'answer': line['answer'],
                                 'prediction': prediction.strip(), 'main_model_input': final_input_prompt,
                                 'all': result,
                                 'model_answer': model_answer,
                                 'question': line['question']}

            
            with open(result_file_path, 'a+', encoding='utf-8') as result_file:
                result_file.write(json.dumps(model_answer_dict, ensure_ascii=False) + '\n')
