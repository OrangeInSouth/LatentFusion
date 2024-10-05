import pdb

import torch
from torch import nn
from torch.optim import lr_scheduler
from utils.block_operations import block_dot, block_cosine_similarity
from utils import avg
import random

class RelativeFuser():
    """
    DeepFuser fuses the hidden states of multiple models in the relative space 
        and returns the absolute representation of the fusion result.
    """
    def __init__(self, 
                 relative_transformation_matrix_list, 
                 model_list,
                 device_compute="cuda:0", 
                 learning_epochs_nums=5,
                 learning_rate=0,
                 temperatures=None,
                 main_model_selection_strategy=None,
                 ensembel_weights=None,
                 beta=0,
                 p=1,
                 l1_alpha=0):
        """
        relative_transformation_matrix_list: with shape (layer_num + 1, anchor_num, dimension)
        """
        model_num = len(relative_transformation_matrix_list)
        if temperatures is None:
            self.temperature_list = [10] * model_num
        else:
            self.temperature_list = temperatures

        # self.relative_transformation_matrix_list = [m.to(device_compute).to(torch.float32) for m in relative_transformation_matrix_list]
        self.relative_transformation_matrix_list = [m.to(device_compute) for m in relative_transformation_matrix_list]
        self.learning_rate = learning_rate
        self.model_list = model_list
        self.device_compute = device_compute
        self.learning_epochs_nums = learning_epochs_nums
        self.main_model_selection_strategy = main_model_selection_strategy

        if ensembel_weights is None:
            ensembel_weights = [1 / model_num] * model_num
        self.ensembel_weights = torch.tensor(ensembel_weights).to(device_compute)

        # self.rep_dist_type = "MultivariateNorm" # Norm
        self.rep_dist_type = "Norm" # Norm
        self.rep_dist = None

        self.beta = beta
        self.p = p
        self.l1_alpha = l1_alpha

        # test code:
        self.ori_state = []
        self.new_state = []

    def transform_to_relative_embedding(self, hidden_state, model_id, layer, main_model_id=0):
        """
        hidden state: (T, d)  
        transformation_matrix: (A, d)
        return: (T, A)
        """
        transformation_matrix = self.relative_transformation_matrix_list[model_id][layer]
        temperature = self.temperature_list[model_id]
        # pdb.set_trace()

        main_model = self.model_list[main_model_id].model
        if model_id == main_model_id and layer == len(main_model.layers):
            hidden_state = main_model.norm(hidden_state)

        # relative_embeds = block_dot(hidden_state, transformation_matrix, block_size=80)
        relative_embeds = block_cosine_similarity(hidden_state, transformation_matrix, block_size=80)
        # pdb.set_trace()
        relative_embeds = (relative_embeds * temperature).softmax(dim=-1)
        
        return relative_embeds

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

    def weighted_sum(self, weights, prob_list):
        weights_tensor = torch.tensor(weights)
        weighted_sum = torch.zeros_like(prob_list[0])
        for tensor, weight in zip(prob_list, weights_tensor):
            weighted_sum += tensor * weight
        return weighted_sum

    def weighted_sum(self, relative_embed_list):
        """
        relative_embed_list: (N, T, A)

        return: (T, A)
        """
        relative_embed_list = torch.stack(relative_embed_list)
        relative_embed_list = relative_embed_list * self.ensembel_weights.unsqueeze(dim=-1).unsqueeze(dim=-1)
        aggregated_relative_embed = relative_embed_list.sum(dim=0)
        return aggregated_relative_embed


    def get_hidden_likelihood(self, hidden_state, layer, main_model_id=0):
        """
        return a probability
        """
        
        if self.rep_dist is None:
            data = self.relative_transformation_matrix_list[main_model_id][layer].to(torch.float32)
            if self.rep_dist_type == "MultivariateNorm":
                mean = torch.mean(data, dim=0)
                # 计算协方差矩阵
                cov_matrix = torch.cov(data.T)
                epsilon = 1e-6
                cov_matrix += epsilon * torch.eye(cov_matrix.size(0)).to(data.device)
                self.rep_dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
            elif self.rep_dist_type == "Norm":
                mean = data.mean(dim=0)
                std = data.std(dim=0)
                self.rep_dist = torch.distributions.Normal(mean, std)
            else:
                raise Exception("Unsupported distribution")
        
        main_model = self.model_list[main_model_id].model
        if layer == len(main_model.layers):
            hidden_state = main_model.norm(hidden_state)
            
        if self.rep_dist_type == "MultivariateNorm":
            scale = hidden_state.size(-1)
            return (self.rep_dist.log_prob(hidden_state.to(torch.float32)) / scale).exp().mean()
        elif self.rep_dist_type == "Norm":
            return self.rep_dist.log_prob(hidden_state.to(torch.float32)).exp().mean()
        else:
            raise Exception("Unsupported distribution")
    
    def get_hidden_MLE_loss(self, hidden_state, layer, main_model_id=0):
        """
        return a tensor of each token's MLE loss.
        """
        
        if self.rep_dist is None:
            data = self.relative_transformation_matrix_list[main_model_id][layer].to(torch.float32)
            if self.rep_dist_type == "MultivariateNorm":
                mean = torch.mean(data, dim=0)
                # 计算协方差矩阵
                cov_matrix = torch.cov(data.T)
                epsilon = 1e-6
                cov_matrix += epsilon * torch.eye(cov_matrix.size(0)).to(data.device)
                self.rep_dist = torch.distributions.MultivariateNormal(mean, cov_matrix)
            elif self.rep_dist_type == "Norm":
                mean = data.mean(dim=0)
                std = data.std(dim=0)
                self.rep_dist = torch.distributions.Normal(mean, std)
            else:
                raise Exception("Unsupported distribution")
        
        main_model = self.model_list[main_model_id].model
        if layer == len(main_model.layers):
            hidden_state = main_model.norm(hidden_state)

        if self.rep_dist_type == "MultivariateNorm":
            scale = hidden_state.size(-1) ** 2

            return self.rep_dist.log_prob(hidden_state.to(torch.float32)) / scale * -1
        elif self.rep_dist_type == "Norm":
            return self.rep_dist.log_prob(hidden_state.to(torch.float32)).mean() * -1
        else:
            raise Exception("Unsupported distribution")
        
 
    def grad_prune(self, grad):
        # if self.p == 1:
        #     return grad
        indices = grad.topk(int(grad.size(-1) * self.p))[1]
        mask = torch.zeros_like(grad, dtype=torch.bool)
        mask.scatter_(1, indices, 1)
        grad = grad * mask

        return grad, indices

    def fuse(self, hidden_state_list, layer_alignment, ensemble_flag=True,
                                    log_file_path=None):

        learning_rate = self.learning_rate
        if  learning_rate > 1e-9 and len(self.relative_transformation_matrix_list) > 1 and ensemble_flag:

            assert len(hidden_state_list) == len(
                self.relative_transformation_matrix_list), "输入logits数量与相对表示矩阵数量不匹配"

            relative_embedding_list = []
            
            with torch.no_grad():
                # 1. 相对表示转换
                for model_id, hidden_state in enumerate(hidden_state_list):
                    relative_embedding = self.transform_to_relative_embedding(hidden_state, model_id, layer_alignment[model_id])
                    relative_embedding = relative_embedding.to(self.device_compute)
                    relative_embedding_list.append(relative_embedding)

                # 2. 相对表示聚合
                aggregated_relative_embedding = self.weighted_sum(relative_embedding_list)
            main_model_id = 0
            
            # 3. 相对表示逆映射
            torch.set_grad_enabled(True)
            ori_dtype = hidden_state_list[main_model_id].dtype
            main_model_states = hidden_state_list[main_model_id].detach().clone().to(
                self.device_compute).to(torch.float32)

            main_model_states.requires_grad_(True)

            criterion = nn.KLDivLoss()
            # criterion2 = nn.MSELoss()
            # criterion = nn.CrossEntropyLoss()
            
            optimizer = torch.optim.Adam(params=[main_model_states],
                                          lr=learning_rate,
                                          betas=(0.9, 0.999))

            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=learning_rate / 4)
            loss_list = []
            KNN_assess_list = [1,5,10,100]
            MuKNN_dict = {}
            for k in KNN_assess_list:
                MuKNN_dict[k] = []
            grad_norm_list = []

            ori_likelihood = self.get_hidden_likelihood(main_model_states, layer_alignment[0]).item()
            ori_MSE_loss = self.get_hidden_MLE_loss(main_model_states, layer_alignment[0]).detach().clone().to(
            self.device_compute).to(torch.float32)
            
            # # 检测神经元变化
            # watched_neurons, watched_neurons_index = main_model_states[-1].topk(5)
            # watched_neurons_history = [watched_neurons.tolist()]
            all_changed_neuron_index = []

            for _ in range(self.learning_epochs_nums):
                main_relative_embed = self.transform_to_relative_embedding(main_model_states, main_model_id, layer_alignment[main_model_id])
                main_relative_embed.retain_grad()
                
                main_relative_embed = main_relative_embed.float().to(self.relative_transformation_matrix_list[main_model_id].device)
                main_relative_embed = main_relative_embed.log()

                relative_fuse_loss = criterion(main_relative_embed, aggregated_relative_embedding)

                for k in KNN_assess_list:
                    MuKNN_dict[k].append(cal_MuKNN_consistency(main_relative_embed, aggregated_relative_embedding, knn=k))
                if self.beta > 0:
                    cur_MSE_loss = self.get_hidden_MLE_loss(main_model_states, layer_alignment[0])
                    
                    ppl_loss = ((torch.relu(cur_MSE_loss - ori_MSE_loss))**2).mean()
                    # ppl_loss = cur_MSE_loss.mean()
                    # pdb.set_trace()
                    loss = relative_fuse_loss + ppl_loss * self.beta
                else:
                    loss = relative_fuse_loss

                if self.l1_alpha > 0:
                    l1_loss = (main_model_states - hidden_state_list[main_model_id]).abs().mean()
                    # l1_loss = ((main_model_states.norm(dim=-1).to(torch.float32) - hidden_state_list[main_model_id].norm(dim=-1).to(torch.float32))**2).mean()
                    loss += l1_loss * self.l1_alpha

                loss_list.append(relative_fuse_loss.item())
                optimizer.zero_grad()
                loss.backward()

                main_model_states.grad, changed_neuron_index = self.grad_prune(main_model_states.grad)

                all_changed_neuron_index += changed_neuron_index[-1].tolist()
                grad_norm_list.append(main_model_states.grad.norm().item())
                # watched_neurons_history.append(main_model_states[-1][watched_neurons_index].tolist())

                # pdb.set_trace()
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

            main_model_states = main_model_states.to(ori_dtype)
            print(f"Norm: {hidden_state_list[main_model_id].norm(dim=-1).mean().item()} =>  {main_model_states[-1].norm(dim=-1).mean().item()}")
            print(f"Prob: {ori_likelihood} =>  {self.get_hidden_likelihood(main_model_states, layer_alignment[0]).item()}")
            for k in KNN_assess_list:
                    print(f"MuKNN (k={k})", MuKNN_dict[k])
            # print(f"Grad Norm:", grad_norm_list)
            # for i in range(len(watched_neurons_history[0])):
            #     print(f"top-{i+1} neuron:", [v[i] for v in watched_neurons_history])
            n = 10
            print(f"随机抽{n}个发生了变化的神经元：")
            all_changed_neuron_index = list(set(all_changed_neuron_index))
            random.shuffle(all_changed_neuron_index)
            for i in range(n):
                index = all_changed_neuron_index[i]
                print(f"neuron-{i+1}: {hidden_state_list[main_model_id][-1][index]} => {main_model_states[-1][index]}")

            print(f"看{n}个发生了最大变化的神经元")
            changed_neuron_value, changed_neuron_index = (main_model_states[-1] - hidden_state_list[main_model_id][-1]).abs().topk(n)
            for i in range(n):
                index = changed_neuron_index[i]
                print(f"neuron-{i+1}: {hidden_state_list[main_model_id][-1][index]} => {main_model_states[-1][index]}")

            print(f"看{n}个增加量最大的神经元")
            changed_neuron_value, changed_neuron_index = (main_model_states[-1] - hidden_state_list[main_model_id][-1]).topk(n)
            for i in range(n):
                index = changed_neuron_index[i]
                print(f"neuron-{i+1}: {hidden_state_list[main_model_id][-1][index]} => {main_model_states[-1][index]}")
            
            # print(f"前10个神经元")
            # print(hidden_state_list[main_model_id][-1][0])
            # print()
            
            print(f"delta sum: {(main_model_states[-1] - hidden_state_list[main_model_id][-1]).sum()}, max: {(main_model_states[-1] - hidden_state_list[main_model_id][-1]).max()}, min: {(main_model_states[-1] - hidden_state_list[main_model_id][-1]).min()}")

            # test code:
            self.ori_state.append(main_model_states[-1])
            self.new_state.append(hidden_state_list[main_model_id][-1])
            if len(self.ori_state) > 10000:
                torch.save(torch.stack(self.ori_state, dim=0), "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/NQ/dev/Debug_llama2-13b_mistral-7b_10000anchors/ori_state_llama.pt")
                torch.save(torch.stack(self.new_state, dim=0), "/data/home/cpfu/ychuang/DeepEN_v0601_ychuang/experiments/NQ/dev/Debug_llama2-13b_mistral-7b_10000anchors/new_state_llama.pt")
                exit(0)
            # pdb.set_trace()

            return main_model_states
        else:
            return hidden_state_list[0]


def cal_MuKNN_consistency(embedding1, embedding2, knn=10):

    assert embedding1.size(0) == embedding2.size(0)

    s1_knn = embedding1.topk(knn, dim=-1)[1].tolist()
    s2_knn = embedding2.topk(knn, dim=-1)[1].tolist()

    res_list = []
    for i in range(len(s1_knn)):
        res_list.append(len(set(s1_knn[i]) & set(s2_knn[i]))/ knn)

    return avg(res_list)

