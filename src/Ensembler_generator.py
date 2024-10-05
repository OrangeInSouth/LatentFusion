import pdb

import torch

from src.Model_generator import generate_one


class Ensembler_generator():
    def __init__(self, model_list, tokenizer_list,device_list, ensembler ):

        self.model_list = model_list
        self.tokenizer_list = tokenizer_list
        self.device_list = device_list
        self.ensembler = ensembler


    # def ensemble_generate(self, llm_input_text_list, max_new_tokens=400, learning_rate=0):
    #     current_input_string = ""
    #     llm_input_ids_list = []
    #     for index, llm_input_text in enumerate(llm_input_text_list):
    #         llm_input_ids = self.tokenizer_list[index].encode(llm_input_text, return_tensors="pt").to(self.device)
    #         llm_input_ids_length = llm_input_ids.shape[1]
    #         llm_input_ids_list.append(llm_input_ids)
    #
    #     for _ in range(max_new_tokens):
    #         llm_output_logits_list = []
    #         for index in range(len(llm_input_ids_list)):
    #             llm_output_logits = generate_one(self.model_list[index], llm_input_ids_list[index]).to(self.device)
    #             llm_output_logits_list.append(llm_output_logits)
    #
    #         preliminary_next_token_idx = torch.argmax(llm_output_logits_list[0], dim=-1)
    #         if preliminary_next_token_idx.item() == self.tokenizer_list[0].eos_token_id:
    #             break
    #
    #         # preliminary_next_token_to_string = self.tokenizer_list[0].convert_ids_to_tokens(preliminary_next_token_idx)[
    #         #     0].replace(
    #         #     "▁",
    #         #     " ").strip()
    #
    #         ensemble_output = self.ensembler.ensemble_base(llm_output_logits_list,
    #                                                        learning_rate=learning_rate)
    #         next_token_idx = torch.argmax(ensemble_output, dim=-1)
    #
    #         llm_input_ids = torch.cat((llm_input_ids_list[0], next_token_idx.view(1, -1)), dim=1)
    #         current_input_string = self.tokenizer_list[0].decode(llm_input_ids.tolist()[0])
    #
    #         if next_token_idx.item() == self.tokenizer_list[0].eos_token_id:
    #             break
    #
    #         for index in range(len(llm_input_ids_list)):
    #             llm_input_ids_list[index] = self.tokenizer_list[index].encode(current_input_string,
    #                                                                           return_tensors="pt").to(self.device)
    #
    #     return current_input_string

    def ensemble_generate_select_decode_0601(self, llm_input_text_list, max_new_tokens=400, learning_rate=0):
        current_input_string = ""
        llm_input_ids_list = []
        for index, llm_input_text in enumerate(llm_input_text_list):
            llm_input_ids = self.tokenizer_list[index].encode(llm_input_text, return_tensors="pt").to(self.device_list[index])
            llm_input_ids_length = llm_input_ids.shape[1]
            llm_input_ids_list.append(llm_input_ids)

        for _ in range(max_new_tokens):
            llm_output_logits_list = []
            for index in range(len(llm_input_ids_list)):
                llm_output_logits = generate_one(self.model_list[index], llm_input_ids_list[index]).to(self.device_list[0])
                llm_output_logits_list.append(llm_output_logits)

            preliminary_next_token_idx = torch.argmax(llm_output_logits_list[0], dim=-1)
            if preliminary_next_token_idx.item() == self.tokenizer_list[0].eos_token_id:
                break

            ensemble_output, main_model_id = self.ensembler.ensemble_select_decode_0601(llm_output_logits_list,
                                                                                        learning_rate=learning_rate)
            next_token_idx = torch.argmax(ensemble_output, dim=-1)


            llm_input_ids = torch.cat((llm_input_ids_list[main_model_id], next_token_idx.view(1, -1).to(llm_input_ids_list[main_model_id].device)), dim=1)
            current_input_string = self.tokenizer_list[main_model_id].decode(llm_input_ids.tolist()[0])
            # pdb.set_trace()
            print(main_model_id)
            if next_token_idx.item() == self.tokenizer_list[main_model_id].eos_token_id:
                break

            for index in range(len(llm_input_ids_list)):
                llm_input_ids_list[index] = self.tokenizer_list[index].encode(current_input_string,
                                                                              return_tensors="pt").to("cuda:0")

        return current_input_string
