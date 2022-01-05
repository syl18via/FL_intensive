from __future__ import absolute_import, division, print_function

import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import argparse
import json
import os
import pprint

### self-defined module
import policy
from dataloader import *
import util
import tff_util
from task import Task

parser = argparse.ArgumentParser(description='FL')
parser.add_argument("--distribution", type=str, default="mix", help="Data distribution")
parser.add_argument("--lr", type=float, default=0.1, help="Initialized learning rate")
parser.add_argument("--policy", type=str, default="my", help="Client Assignment Policy")
parser.add_argument("--trade_once", action="store_true", help="Set to update clients selection only after the first epoch")
args = parser.parse_args()

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

# NUM_EXAMPLES_PER_USER = 1000

### Experiment Configs
MIX_RATIO = 0.8
SIMULATE = False
EPOCH_NUM = 50
TRIAL_NUM = 20
TASK_NUM = 2
bid_per_loss_delta_space = [1]
required_client_num_space = [2,1]
target_labels_space = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]

START_TIME = time.time()

# class Client:
#     def __init__(self, ...):
#         self.cost = xxx
#         self.id = xxx
#         self.idlecost = xxx
#         self.data = xxx

def pick_client_based_on_index(task, epoch, selected_client_idx, all_client_data, all_client_data_full):
    clients_data = []
    # if epoch == 0:
    #      return all_client_data_full
    
    # else:
    for idx in selected_client_idx:
        batch= next(all_client_data[idx])
        if task.target_labels is None:
            new_batch = batch
        else:
            # filter batch according to required labels of tasks
            new_batch = {"x": [], "y": []}
            for idx, y in enumerate(batch["y"]):
                if y in task.target_labels:
                    new_batch["x"].append(batch["x"][idx])
                    new_batch["y"].append(batch["y"][idx])

        clients_data.append([new_batch])

    return clients_data

def train_one_round(
        task,
        round_idx,
        learning_rate,
        epoch,
        all_client_data,
        all_client_data_full,
        test_data,
        ckpt=False, evaluate_each_client=False):
    if task.selected_client_idx is None:
        assert args.trade_once
        task.totoal_loss_delta = 0
        return
    clients_data = pick_client_based_on_index(task, epoch, task.selected_client_idx, all_client_data, all_client_data_full)
    task.params_per_client = [None] * len(task.selected_client_idx)

    ### Train
    if SIMULATE:
        local_models = [(task.model['weights'], task.model['bias']) for _ in task.selected_client_idx]
    else:
        local_models = tff_util.federated_train(task.model, learning_rate, clients_data)
    
    ### Output model parameters of all selected agents of this task
    ### Store the model parameters of this round if ckpt is True
    for local_index in range(len(local_models)):
        client_id = task.selected_client_idx[local_index]
        if epoch == 0:
            if task.params_per_client[local_index] is None:
                task.params_per_client[local_index] = []
            task.params_per_client[local_index].append(
                (local_models[local_index], learning_rate))
        task.params_per_client[local_index] = (local_models[local_index], learning_rate)
        if ckpt:
            f = open(os.path.join(os.path.dirname(__file__), "weights_"+str(client_id)+".txt"),"a",encoding="utf-8")
            for i in local_models[local_index][0]:
                line = ""
                arr = list(i)
                for j in arr:
                    line += (str(j)+"\t")
                print(line, file=f)
            print("***"+str(learning_rate)+"***",file=f)
            print("-"*50,file=f)
            f.close()

            f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(client_id) + ".txt"), "a", encoding="utf-8")
            line = ""
            for i in local_models[local_index][1]:
                line += (str(i) + "\t")
            print(line, file=f)
            print("***" + str(learning_rate) + "***",file=f)
            print("-"*50,file=f)
            f.close()
    
    ### Store the model before updating
    if task.model_epoch_start is None:
        task.model_epoch_start = task.model

    ### Update the global model parameters
    task.model = tff_util.weighted_average_model(local_models, data_num, 
        task.selected_client_idx, evaluate_each_client, test_data=None)
    task.learning_rate = learning_rate

    ### evaluate the loss
    loss = tff_util.evaluate_loss_on_server(task.model, test_data)
    task.totoal_loss_delta = 1000*float(task.prev_loss - loss)
    task.prev_loss = loss
    task.log("Epoch {} Round {} at {:.3f} s, selected_client_idx: {}, learning rate: {:.3f}, loss: {:.3f}, loss_delta: {:.3f}".format(
        epoch, round_idx, time.time()-START_TIME, task.selected_client_idx, learning_rate, loss, task.totoal_loss_delta))

    #raise NotImplementedError("TODO, update task.total_loss_delta")
    
def calculate_feedback(task, test_data):
    ### TODO: comment the process to calculate the shapley value
    ### Substitute with our own algorithm
    
    if task.selected_client_idx is None:
        return []

    ### Calculate the Feedback
    selected_client_num = len(task.selected_client_idx)
    all_sets = util.PowerSetsBinary([i for i in range(selected_client_num)])
    group_shapley_value = []

    # print(train_with_gradient_and_valuation(task, [0], test_data, data_num))
    # print(train_with_gradient_and_valuation(task, [9], test_data, data_num))
    # print(train_with_gradient_and_valuation(task, [0, 9], test_data, data_num))

    # raise

    for s in all_sets:
        _loss = tff_util.train_with_gradient_and_valuation(task, s, test_data, data_num)
        contrib = task.prev_loss - _loss
        group_shapley_value.append(contrib)
        # task.log(str(s)+"\t"+str(group_shapley_value[len(group_shapley_value)-1]))

    agent_shapley = []
    for index in range(selected_client_num):
        shapley = 0.0
        for set_idx, j in enumerate(all_sets):
            if index in j:
                remove_list_index = util.remove_list_indexed(index, j, all_sets)
                if remove_list_index != -1:

                    shapley += (group_shapley_value[set_idx] - group_shapley_value[
                        remove_list_index]) / (comb(selected_client_num - 1, len(all_sets[remove_list_index])))

        agent_shapley.append(shapley)
    # for ag_s in agent_shapley:
    #     print(ag_s)
    # task.select_clients(agent_shapley, free_client)
    # if sum(agent_shapley) == 0:
    #     import code
    #     code.interact(local=locals())
    return agent_shapley

def run_one_trial():

    ### Load data
    all_client_data, test_data, all_client_data_full = main_load(args)

    ############################### client ###########################################
    ### 0 denotes free, 1 denotes being occupied
    free_client = [0] * NUM_AGENT

    cost_list = []
    for client_idx in range(NUM_AGENT):
        # cost_list.append(random.randint(1,10)/10)
        cost_list.append(0)
    
    idlecost_list = []
    for client_idx in range(NUM_AGENT):
        idlecost_list.append(0)

    client_feature_list = list(zip( cost_list, idlecost_list))

    # client_list = []
    # for client_idx in range(NUM_AGENT):
    #     client = Client(....)
    #     client_list.append(client)
    ############################### client end ###########################################
    
    ### Read inital model parameters from files
    w_initial, b_initial = tff_util.init_model()
    learning_rate = args.lr
    
    ############################### Task ###########################################
    ### Initialize the global model parameters for both tasks
    ### At the first epoch, both tasks select all clients
    task_list = []
    def create_task(selected_client_idx, init_model, required_client_num, bid_per_loss_delta, target_labels=None):
        task = Task(
            task_id = len(task_list),
            selected_client_idx=selected_client_idx,
            model = init_model,
            required_client_num=required_client_num,
            bid_per_loss_delta=bid_per_loss_delta,
            target_labels=target_labels)
        task_list.append(task)

        ### Init the loss
        task.prev_loss = tff_util.evaluate_loss_on_server(task.model, test_data)
    
    for task_id in range(TASK_NUM):
        create_task(
            selected_client_idx=list(range(NUM_AGENT)),
            init_model = {
                    'weights': w_initial,
                    'bias': b_initial
            },
            required_client_num=util.sample_config(required_client_num_space, task_id, use_random=True),
            bid_per_loss_delta=util.sample_config(bid_per_loss_delta_space, task_id, use_random=True),
            target_labels=util.sample_config(target_labels_space,task_id, use_random=True)
        )

    ### Initialize the price_table and bid table
    price_table = None
    def init_price_table(price_table):
        price_table = []
        for client_idx in range(NUM_AGENT):
            init_price_list = []
            for taks_idx in range(len(task_list)):
                init_price_list.append(0)
            price_table.append(init_price_list)
        return price_table
    
    price_table = init_price_table(price_table)
    bid_table = np.zeros((NUM_AGENT, len(task_list)))

    ############################### Main process of FL ##########################################
    total_reward_list = []
    succ_cnt_list = []
    reward_sum=[]
    for epoch in range(EPOCH_NUM):
        for task in task_list:
            task.epoch = epoch

        for round_idx in range(1):
            ### Train the model parameters distributedly
            #   return a list of model parameters
            #       local_models[0][0], weights of the 0-th agent
            #       local_models[0][1], bias of the 0-th agent

            for task in task_list:
                train_one_round(
                    task,
                    round_idx,
                    learning_rate,
                    epoch,
                    all_client_data,
                    all_client_data_full,
                    test_data,
                    ckpt=False,
                    evaluate_each_client=False)
            
            learning_rate = learning_rate * 0.7

        ### At the end of this epoch
        ### At the first epoch, calculate the Feedback and update clients for each task
        
        print("Start to update client assignment ... ")
  
        shapely_value_table = [calculate_feedback(task, test_data) for task in task_list]
        ### Normalize using sigmoid
        shapely_value_table = [
            util.sigmoid(np.array(elem)) if len(elem) > 0 else elem 
                for elem in shapely_value_table]
        shapely_value_table = np.array(shapely_value_table)
        print(shapely_value_table)

        ### Update price table
        for task_idx in range(len(task_list)):
            if task_list[task_idx].selected_client_idx is None:
                continue
            selected_client_index = task_list[task_idx].selected_client_idx
            for idx in range(len(selected_client_index)):
                client_idx = selected_client_index[idx]
                shapley_value = shapely_value_table[task_idx][idx]
                shapely_value_scaled = shapley_value * len(selected_client_index) / NUM_AGENT
                # price_table[client_idx][task_idx] = ((epoch / (epoch + 1)) * price_table[client_idx][task_idx] + (1 / (epoch + 1)) * shapely_value_scaled) 
                price_table[client_idx][task_idx] = shapely_value_scaled 
        
        total_cost = 0
        bid_list = [task.totoal_loss_delta * task.bid_per_loss_delta for task in task_list]
        total_bid = sum(bid_list)
        
        for task in task_list:
            if task.selected_client_idx is None:
                continue
            for client_idx in task.selected_client_idx :
                total_cost += cost_list[client_idx]

        assert price_table is not None
    
        ### Update bid table
        for task_idx in range(len(task_list)):
            if task_list[task_idx].selected_client_idx is None:
                continue
            selected_client_index = task_list[task_idx].selected_client_idx
            for idx in range(len(selected_client_index)):
                client_idx = selected_client_index[idx]
                shapley_value = shapely_value_table[task_idx][idx]
                bid_table[client_idx][task_idx] = shapley_value * bid_list[task_idx]

        # reward_list = [task.totoal_loss_delta * task.bid_per_loss_delta for task in task_list]
        # reward_list = [task.totoal_loss_delta * task.bid_per_loss_delta - total_cost for task in task_list]
        #print ('reward list', reward_list)

        print("Start to select clients ... ")
        if epoch == 0 or not args.trade_once:
            ask_table = util.calcualte_client_value(price_table, client_feature_list)
            norm_ask_table = util.normalize_data(ask_table)
            norm_bid_table = util.normalize_data(bid_table)
            if args.policy == "my":
                succ_cnt, reward = policy.my_select_clients(
                    norm_ask_table,
                    client_feature_list,
                    task_list,
                    norm_bid_table)
            elif args.policy == "random":
                succ_cnt, reward = policy.random_select_clients(
                    norm_ask_table,
                    client_feature_list,
                    task_list,
                    norm_bid_table)
            elif args.policy == "simple":
                succ_cnt, reward = policy.simple_select_clients(task_list, NUM_AGENT)
            elif args.policy == "mcafee":
                if epoch == 0:
                    succ_cnt, reward = policy.mcafee_select_clients(
                        norm_ask_table,
                        client_feature_list,
                        task_list,
                        norm_bid_table)
                    norm_bid_table_first_epoch = norm_bid_table.copy()
            else:
                raise
        print("Client assignment Done ")
        
        for task in task_list:
            task.end_of_epoch()

        ### caclulate reward
        if epoch > 0:
            if args.policy == "mcafee":
                _, reward  = policy.mcafee_select_clients(
                        norm_ask_table,
                        client_feature_list,
                        task_list,
                        norm_bid_table_first_epoch,
                        update=False)
                bid_list = [task.totoal_loss_delta * reward for task in task_list]
                print ([task.totoal_loss_delta for task in task_list])
                total_bid = sum(bid_list)
                total_reward = total_bid - total_cost
                total_reward_list.append(total_reward)
                reward_sum.append(sum(total_reward_list))
                
                # raise NotImplementedError("Current implementation is wrong")
                # client_value_table = policy.calcualte_client_value(price_table, client_feature_list)
                # task_price_list = np.sum(bid_table, axis=0)
                # sorted_task_with_index = sorted(enumerate(task_price_list), key=lambda x: x[1], reverse=True)
                # client_value_list = np.sum((client_value_table), axis=1)
                # client_value_list_sorted = sorted(enumerate(client_value_list), key=lambda x: x[1], reverse=False)
                # for j in selected_client_index:
                #     b = client_value_list_sorted[j][1]
                # bid_list = [task.totoal_loss_delta * 1/2* (task.bid_per_loss_delta +b ) for task in task_list]
            else:     
                total_reward = total_bid - total_cost
                total_reward_list.append(total_reward)
                reward_sum.append(sum(total_reward_list))
                print(reward_sum[-1])

        ### Count successful matching
        succ_cnt_list.append(succ_cnt)

    ### end of trianing
    with open("total_reward_list_{}.json".format(args.policy), 'w') as fp:
        json.dump({"total_reward)=_list": total_reward_list}, fp, indent=4)
    print(reward_sum)
    # print(f"Count for succcessful matching: {succ_cnt_list}")
    return succ_cnt_list

if __name__ == "__main__":
    succ_cnt_table = [] ###  shape = (TRIAL_NUM, EPOCH_NUM)
    for _ in range(TRIAL_NUM):
        succ_cnt_list = run_one_trial()
        succ_cnt_table.append(succ_cnt_list)
    
    pprint(succ_cnt_table)