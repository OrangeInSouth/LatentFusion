import pdb

import torch
from torch import nn
from torch.optim import lr_scheduler


class Ensembler():
    def __init__(self, probability_transfer_matrix_list, device_compute="cuda:0"):
        self.probability_transfer_matrix_list = probability_transfer_matrix_list
        self.device_compute = device_compute

    def euclidean_distance(self, tensor1, tensor2):
        # 确保两个张量的形状相同
        assert tensor1.shape == tensor2.shape, "张量的形状必须相同"

        # 计算两个张量之间的差的平方
        diff_squared = (tensor1 - tensor2) ** 2

        # 对差的平方进行求和，然后开平方根得到欧几里得距离
        distance = torch.sqrt(diff_squared.sum())

        return distance

    def kl_divergence(self, p_probs, q_probs):
        kl_div = torch.nn.functional.kl_div(torch.log(p_probs), q_probs, reduction='batchmean')
        return kl_div

    # def ensemble_base(self, model_generate_ids_logits_list, ensemble_flag, learning_epochs_nums=5, learning_rate=0,
    #                   log_file_path=None):
    #
    #     if abs(learning_rate) > 1e-6 and len(self.probability_transfer_matrix_list) > 1 and ensemble_flag:
    #
    #         assert len(model_generate_ids_logits_list) == len(
    #             self.probability_transfer_matrix_list), "输入logits数量与相对表示矩阵数量不匹配"
    #
    #         model_generate_ids_probs_list = []
    #         with torch.no_grad():
    #
    #             for model_generate_ids_logits, probability_transfer_matrix in zip(model_generate_ids_logits_list,
    #                                                                               self.probability_transfer_matrix_list):
    #                 model_generate_ids_probs = nn.functional.softmax(model_generate_ids_logits, dim=-1).float()
    #                 model_relative_representation_probs = torch.mm(model_generate_ids_probs.to(self.device_compute),
    #                                                                probability_transfer_matrix)
    #
    #                 model_generate_ids_probs_list.append(model_relative_representation_probs)
    #
    #             stacked_model_generate_ids_probs = torch.stack(model_generate_ids_probs_list, dim=0)
    #
    #             average_probs = torch.mean(stacked_model_generate_ids_probs, dim=0)
    #
    #         torch.set_grad_enabled(True)
    #         main_model_generate_ids_logits = model_generate_ids_logits_list[0].detach().clone().to(self.device_compute)
    #
    #         main_model_generate_ids_logits.requires_grad_(True)
    #
    #         criterion = nn.KLDivLoss()
    #         optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
    #                                       lr=learning_rate,
    #                                       betas=(0.9, 0.999))
    #
    #         scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=learning_rate / 4)
    #         loss_list = []
    #         for _ in range(learning_epochs_nums):
    #             main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits, dim=-1).float()
    #             main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
    #                                                                 self.probability_transfer_matrix_list[0])
    #
    #             log_main_probs = torch.log(main_model_relative_representation_probs)
    #             loss = criterion(log_main_probs, average_probs)
    #             loss_list.append(loss.item())
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             scheduler.step()
    #         if log_file_path is not None:
    #             with open(log_file_path, "a+", encoding="utf-8") as process_file:
    #                 process_file.write("\n==weights_list:\n")
    #                 process_file.write("average_probs" + "\n")
    #                 process_file.write("\n==loss_list:\n")
    #                 process_file.write(str(loss_list) + "\n")
    #         torch.set_grad_enabled(False)
    #         del model_generate_ids_logits_list
    #         return main_model_generate_ids_logits
    #     else:
    #         return model_generate_ids_logits_list[0]

    def weighted_sum(self, weights, prob_list):
        weights_tensor = torch.tensor(weights)
        weighted_sum = torch.zeros_like(prob_list[0])
        for tensor, weight in zip(prob_list, weights_tensor):
            weighted_sum += tensor * weight
        return weighted_sum

    def ensemble_select_decode_0601(self, model_generate_ids_logits_list, ensemble_flag=True, learning_epochs_nums=5,
                                    learning_rate=0,
                                    log_file_path=None):

        if abs(learning_rate) > 1e-6 and len(self.probability_transfer_matrix_list) > 1 and ensemble_flag:

            assert len(model_generate_ids_logits_list) == len(
                self.probability_transfer_matrix_list), "输入logits数量与相对表示矩阵数量不匹配"

            model_generate_ids_probs_list = []
            with torch.no_grad():
                for model_generate_ids_logits, probability_transfer_matrix in zip(model_generate_ids_logits_list,
                                                                                  self.probability_transfer_matrix_list):
                    model_generate_ids_probs = nn.functional.softmax(model_generate_ids_logits, dim=-1).to(
                        torch.float32)

                    model_relative_representation_probs = torch.mm(
                        model_generate_ids_probs.to(probability_transfer_matrix.device),
                        probability_transfer_matrix).to(self.device_compute)

                    model_generate_ids_probs_list.append(model_relative_representation_probs)

                stacked_model_generate_ids_probs = torch.stack(model_generate_ids_probs_list, dim=0)
                average_probs = torch.mean(stacked_model_generate_ids_probs, dim=0)

            kl_list = []
            for index in range(len(model_generate_ids_probs_list)):
                kl_distance = self.kl_divergence(model_generate_ids_probs_list[index].squeeze(dim=0), average_probs.squeeze(dim=0))
                kl_list.append(kl_distance.item())
            print(kl_list)
            main_model_id = kl_list.index(min(kl_list))

            torch.set_grad_enabled(True)
            main_model_generate_ids_logits = model_generate_ids_logits_list[main_model_id].detach().clone().to(
                self.device_compute)

            main_model_generate_ids_logits.requires_grad_(True)

            criterion = nn.KLDivLoss()
            optimizer = torch.optim.AdamW(params=[main_model_generate_ids_logits],
                                          lr=learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=learning_rate / 4)
            loss_list = []
            for _ in range(learning_epochs_nums):
                main_model_generate_ids_probs = nn.functional.softmax(main_model_generate_ids_logits,
                                                                      dim=-1).float().to(
                    self.probability_transfer_matrix_list[main_model_id].device)
                main_model_relative_representation_probs = torch.mm(main_model_generate_ids_probs,
                                                                    self.probability_transfer_matrix_list[
                                                                        main_model_id])

                log_main_probs = torch.log(main_model_relative_representation_probs)
                loss = criterion(log_main_probs, average_probs.to(log_main_probs.device))
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            print(loss_list)

            if log_file_path is not None:
                with open(log_file_path, "a+", encoding="utf-8") as process_file:
                    process_file.write("\n==weights_list:\n")
                    process_file.write("average_probs" + "\n")
                    process_file.write("\n==loss_list:\n")
                    process_file.write(str(loss_list) + "\n")
            torch.set_grad_enabled(False)
            del model_generate_ids_logits_list
            return main_model_generate_ids_logits, main_model_id
        else:
            return model_generate_ids_logits_list[0], 0
