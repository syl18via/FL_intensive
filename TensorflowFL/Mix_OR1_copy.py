from __future__ import absolute_import, division, print_function
from typing import List
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import random
import argparse
import json
import math
import os

### self-defined module
import policy
from dataloader import *

parser = argparse.ArgumentParser(description='FL')
parser.add_argument("--distribution", type=str, default="mix", help="Data distribution")
parser.add_argument("--lr", type=float, default=0.1, help="Initialized learning rate")
parser.add_argument("--policy", type=str, default="my", help="Client Assignment Policy")
parser.add_argument("--trade_once", action="store_true", help="Set to update clients selection only after the first epoch")
args = parser.parse_args()

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

# NUM_EXAMPLES_PER_USER = 1000


MIX_RATIO = 0.8
SIMULATE = False

BATCH_TYPE = tff.NamedTupleType([
    ('x', tff.TensorType(tf.float32, [None, 784])),
    ('y', tff.TensorType(tf.int32, [None]))])

MODEL_TYPE = tff.NamedTupleType([
    ('weights', tff.TensorType(tf.float32, [784, 10])),
    ('bias', tff.TensorType(tf.float32, [10]))])


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE)
def batch_loss(model, batch):
    predicted_y = tf.nn.softmax(tf.matmul(batch.x, model.weights) + model.bias)
    return -tf.reduce_mean(tf.reduce_sum(
        tf.one_hot(batch.y, 10) * tf.log(predicted_y), axis=[1]))


@tff.tf_computation(MODEL_TYPE, BATCH_TYPE, tf.float32)
def batch_train(initial_model, batch, learning_rate):
    # Define a group of model variables and set them to `initial_model`.
    model_vars = tff.utils.create_variables('v', MODEL_TYPE)
    init_model = tff.utils.assign(model_vars, initial_model)

    # Perform one step of gradient descent using loss from `batch_loss`.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    with tf.control_dependencies([init_model]):
        train_model = optimizer.minimize(batch_loss(model_vars, batch))

    # Return the model vars after performing this gradient descent step.
    with tf.control_dependencies([train_model]):
        return tff.utils.identity(model_vars)


LOCAL_DATA_TYPE = tff.SequenceType(BATCH_TYPE)


@tff.federated_computation(MODEL_TYPE, tf.float32, LOCAL_DATA_TYPE)
def local_train(initial_model, learning_rate, all_batches):
    # Mapping function to apply to each batch.
    @tff.federated_computation(MODEL_TYPE, BATCH_TYPE)
    def batch_fn(model, batch):
        return batch_train(model, batch, learning_rate)

    l = tff.sequence_reduce(all_batches, initial_model, batch_fn)
    return l


@tff.federated_computation(MODEL_TYPE, LOCAL_DATA_TYPE)
def local_eval(model, all_batches):
    return tff.sequence_sum(
        tff.sequence_map(
            tff.federated_computation(lambda b: batch_loss(model, b), BATCH_TYPE),
            all_batches))


SERVER_MODEL_TYPE = tff.FederatedType(MODEL_TYPE, tff.SERVER, all_equal=True)
CLIENT_DATA_TYPE = tff.FederatedType(LOCAL_DATA_TYPE, tff.CLIENTS)


@tff.federated_computation(SERVER_MODEL_TYPE, CLIENT_DATA_TYPE)
def federated_eval(model, data):
    return tff.federated_mean(
        tff.federated_map(local_eval, [tff.federated_broadcast(model), data]))


SERVER_FLOAT_TYPE = tff.FederatedType(tf.float32, tff.SERVER, all_equal=True)


@tff.federated_computation(
    SERVER_MODEL_TYPE, SERVER_FLOAT_TYPE, CLIENT_DATA_TYPE)
def federated_train(model, learning_rate, data):
    l = tff.federated_map(
        local_train,
        [tff.federated_broadcast(model),
         tff.federated_broadcast(learning_rate),
         data])
    return l
    # return tff.federated_mean()

def remove_list_indexed(removed_ele, original_l, ll):
    new_original_l = []
    for i in original_l:
        new_original_l.append(i)
    for i in new_original_l:
        if i == removed_ele:
            new_original_l.remove(i)
    for i in range(len(ll)):
        if set(ll[i]) == set(new_original_l):
            return i
    return -1


def shapley_list_indexed(original_l, ll):
    for i in range(len(ll)):
        if set(ll[i]) == set(original_l):
            return i
    return -1

def PowerSetsBinary(items):
    N = len(items)
    set_all = []
    for i in range(2 ** N):
        combo = []
        for j in range(N):
            if (i >> j) % 2 == 1:
                combo.append(items[j])
        set_all.append(combo)
    return set_all


def weighted_average_model(local_models, all_data_num, client_index_list, evaluate_each_client, test_data=None):
    ### Calculate the global model parameters by weighted averaging all local models
    _data_num = []
    for local_model_index in range(len(local_models)):
        client_id = client_index_list[local_model_index]
        _data_num.append(all_data_num[client_id])
    _data_num = np.array(_data_num)
    _agents_weights = np.divide(_data_num, _data_num.sum())

    m_w = np.zeros([784, 10], dtype=np.float32)
    m_b = np.zeros([10], dtype=np.float32)
    # try:
    for local_model_index in range(len(local_models)):
        client_id = client_index_list[local_model_index]
        m_w = np.add(np.multiply(local_models[local_model_index][0], _agents_weights[local_model_index]), m_w)
        m_b = np.add(np.multiply(local_models[local_model_index][1], _agents_weights[local_model_index]), m_b)

        if (evaluate_each_client) and test_data is not None:
            ### Evaluate the model parameters of current agent if `evaluate_each_client` is True
            loss = evaluate_loss_on_server(
                {
                'weights': local_models[local_model_index][0],
                'bias': local_models[local_model_index][1]
                }, 
                test_data)
            print(' - agent {}, loss={}'.format(client_id, loss))
    return {'weights': m_w, "bias": m_b}

def test_accuracy(model, test_data):
    test_images, test_labels_onehot = test_data 
    test_result = np.dot(test_images, model['weights']) + model['bias']
    y = tf.nn.softmax(test_result)
    
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction[:892], tf.float32))
    return accuracy.numpy()

def cross_entropy(model, test_batch):
    predicted_y = tf.nn.softmax(tf.matmul(test_batch[0], model["weights"]) + model["bias"])
    label = test_batch[1]
    
    
    if len(np.shape(label)) == 1:
        label = tf.one_hot(label, 10)
    return -tf.reduce_mean(tf.reduce_sum(label * tf.math.log(predicted_y), axis=[1]))

def evaluate_loss_on_server(model, test_batch):
    return cross_entropy(model, test_batch)

def train_with_gradient_and_valuation(task, client_set, test_data, all_data_num):
    client_index_list = []
    local_models = []
    # if len(client_set) > 2:
    #     import code
    #     code.interact(local=locals())
    for idx in client_set:
        client_index_list.append(task.selected_client_idx[idx])
        local_models.append(task.params_per_client[idx][0])
    model = weighted_average_model(local_models, all_data_num, client_index_list, False, test_data=test_data)
    return np.array(evaluate_loss_on_server(model, test_data))

class Task:
    def __init__(self, task_id, selected_client_idx, model,
            required_client_num=None,
            bid_per_loss_delta=None,
            target_labels=None):
        self.selected_client_idx = selected_client_idx    # a list of client indexes of selected clients
        self.model = model  # model parameters
        self.model_epoch_start = None # model parameters at the start of an epoch
        self.task_id = task_id
        self.learning_rate = None
        self.epoch = 0
        self.required_client_num = required_client_num
        self.bid_per_loss_delta = bid_per_loss_delta

        self.prev_loss = None
        self.totoal_loss_delta = None
        self.params_per_client = None

        self.target_labels = None
    
    def log(self, *args, **kwargs):
        print("[Task {} - epoch {}]: ".format(self.task_id, self.epoch), *args, **kwargs)

    def end_of_epoch(self):
        self.params_per_client = None
        self.model_epoch_start = None

    def select_clients(self, agent_shapley, free_client):
        # zip([1, 2, 3], [a, b, c]) --> [(1, a), (2, b), (3, c)]
        # enumerate([a, b, c])  --> [(1, a), (2, b), (3, c)]
        # agent_shapley = list(enumerate(agent_shapley))
        agent_shapley = zip(list(range(NUM_AGENT)), agent_shapley) ### shapley value of all clients, a list of (client_idx, value)
        #sorted_shapley_value = sorted(agent_shapley, key=lambda x: x[1], reverse=True)
        #self.log("Sorted shapley value: {}".format(sorted_shapley_value))
        self.selected_client_idx = []
        for client_idx, _ in agent_shapley:
            if free_client[client_idx] == 0:
                self.selected_client_idx.append(client_idx)
                if self.required_client_num and len(self.selected_client_idx) >= self.required_client_num:
                    break
        
        # ### !!! Select different clients for different tasks
        # ### TODO: the agent_shapley value should be considered 
        # ### E.g., Top-K
        # if task.task_id == 0:
        #     task.selected_client_idx = [0, 1, 2]
        # else:
        #     task.selected_client_idx = [3, 4]

        ### Update the client table
        for idx in self.selected_client_idx:
            free_client[idx] = 1

        self.log("Clients {} are selected.".format(self.selected_client_idx))

# class Client:
#     def __init__(self, ...):
#         self.cost = xxx
#         self.id = xxx
#         self.idlecost = xxx
#         self.data = xxx


if __name__ == "__main__":
    start_time = time.time()

    ############################### client ###########################################
    data_num = np.asarray([5421 * 2] * NUM_AGENT)
    agents_weights = np.divide(data_num, data_num.sum())

    for index in range(NUM_AGENT):
        f = open(os.path.join(os.path.dirname(__file__), "weights_"+str(index)+".txt"), "w")
        f.close()
        f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(index) + ".txt"), "w")
        f.close()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    if args.distribution.lower() == "mix":
        all_client_data_divide, all_client_data_full = get_data_for_digit_mix(mnist_train)
    elif args.distribution.lower() == "noisey":
        all_client_data_divide = get_data_for_digit_noiseY(mnist_train)
    elif args.distribution.lower() == "noisex":
        all_client_data_divide, all_client_data_full = get_data_for_digit_noiseX(mnist_train)
    elif args.distribution.lower() == "same":
        all_client_data_divide = get_data_for_digit_same(mnist_train)
    elif args.distribution.lower() == "even":
        all_client_data_divide = get_data_evenly(mnist_train)
    elif args.distribution.lower() == "old":
        all_client_data_divide = get_data_for_federated_agents(mnist_train)
    elif args.distribution.lower() == "mix2":
        all_client_data_divide, all_client_data_full = get_data_for_digit_mix2(mnist_train)
    else:
        raise ValueError("Not implemented data distribution {}".format(args.distribution))
    all_client_data = all_client_data_divide
    test_data = get_test_images_labels("SAME")
    
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

    def pick_client_based_on_index(task, epoch, selected_client_idx):
        clients_data = []
        # if epoch == 0:
        #      return all_client_data_full
        
        # else:
        for idx in selected_client_idx:
            batch= next(all_client_data[idx])
            if task.target_labels is None:
                new_batch = batch
            else:
                new_batch = {"x": [], "y": []}
                for idx, y in enumerate(batch["y"]):
                    if y in task.target_labels:
                        new_batch["x"].append(batch["x"][idx])
                        new_batch["y"].append(batch["y"][idx])


            clients_data.append([new_batch])
  

            # filter batch according to required labels of tasks
            
         

        return clients_data

    
    ### Read inital model parameters from files
    f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
    para_lines = f_ini_p.readlines()
    w_paras = para_lines[0].split("\t")
    w_paras = [float(i) for i in w_paras]
    b_paras = para_lines[1].split("\t")
    b_paras = [float(i) for i in b_paras]
    w_initial = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
    b_initial = np.asarray(b_paras, dtype=np.float32).reshape([10])
    f_ini_p.close()

    learning_rate = args.lr
    

    ############################### Task ###########################################
    ### Initialize the global model parameters for both tasks
    ### At the first epoch, both tasks select all clients
    
    bid_per_loss_delta_list = [1]
    required_client_num_list = [1]
    target_labels_list = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    def sample_config(l, id, use_random=True):
        if use_random:
            return random.sample(l, 1)[0]
        else:
            return l[id%len(l)]

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
        task.prev_loss = evaluate_loss_on_server(task.model, test_data)
    
    TASK_NUM = 2
    for task_id in range(TASK_NUM):
        create_task(
            selected_client_idx=list(range(NUM_AGENT)),
            init_model = {
                    'weights': w_initial,
                    'bias': b_initial
            },
            required_client_num=sample_config(required_client_num_list, task_id, use_random=True),
            bid_per_loss_delta=sample_config(bid_per_loss_delta_list, task_id, use_random=True),
            target_labels=sample_config(target_labels_list,task_id, use_random=True)
        )

    

    ### Initialize the price_table
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
    
    def train_one_round(task, round_idx, learning_rate, epoch, ckpt=False, evaluate_each_client=False):
        clients_data = pick_client_based_on_index(task, epoch, task.selected_client_idx)
        task.params_per_client = [None] * len(task.selected_client_idx)

        ### Train
        if SIMULATE:
            local_models = [(task.model['weights'], task.model['bias']) for _ in task.selected_client_idx]
        else:
            local_models = federated_train(task.model, learning_rate, clients_data)
        
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
        task.model = weighted_average_model(local_models, data_num, 
            task.selected_client_idx, evaluate_each_client, test_data=None)
        task.learning_rate = learning_rate

        ### evaluate the loss
        loss = evaluate_loss_on_server(task.model, test_data)
        task.totoal_loss_delta = 1000*float(task.prev_loss - loss)
        task.prev_loss = loss
        task.log("Epoch {} Round {} at {:.3f} s, selected_client_idx: {}, learning rate: {:.3f}, loss: {:.3f}, loss_delta: {:.3f}".format(
            epoch, round_idx, time.time()-start_time, task.selected_client_idx, learning_rate, loss, task.totoal_loss_delta))

        #raise NotImplementedError("TODO, update task.total_loss_delta")

    def calculate_feedback(task):
        ### TODO: comment the process to calculate the shapley value
        ### Substitute with our own algorithm
        
        ### Calculate the Feedback
        selected_client_num = len(task.selected_client_idx)
        all_sets = PowerSetsBinary([i for i in range(selected_client_num)])
        group_shapley_value = []

        # print(train_with_gradient_and_valuation(task, [0], test_data, data_num))
        # print(train_with_gradient_and_valuation(task, [9], test_data, data_num))
        # print(train_with_gradient_and_valuation(task, [0, 9], test_data, data_num))

        # raise

        for s in all_sets:
            _loss = train_with_gradient_and_valuation(task, s, test_data, data_num)
            contrib = task.prev_loss - _loss
            group_shapley_value.append(contrib)
            # task.log(str(s)+"\t"+str(group_shapley_value[len(group_shapley_value)-1]))

        agent_shapley = []
        for index in range(selected_client_num):
            shapley = 0.0
            for set_idx, j in enumerate(all_sets):
                if index in j:
                    remove_list_index = remove_list_indexed(index, j, all_sets)
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
        
    EPOCH_NUM = 50
    ### Main process of FL
    total_reward_list = []
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
                train_one_round(task, round_idx, learning_rate, epoch, ckpt=False, evaluate_each_client=False)
            
            learning_rate = learning_rate * 0.7

        ### At the end of this epoch
        ### At the first epoch, calculate the Feedback and update clients for each task
        
        print("Start to update client assignment ... ")

        bid_list = [task.totoal_loss_delta * task.bid_per_loss_delta for task in task_list]
        total_bid = sum(bid_list)
        total_cost = 0
        
        for task in task_list:
            for client_idx in task.selected_client_idx :
                total_cost += cost_list[client_idx]
        if epoch > 0:       
            total_reward = total_bid - total_cost
            total_reward_list.append(total_reward)
            reward_sum.append(sum(total_reward_list))
            print(reward_sum[-1])
        
        shapely_value_table = [calculate_feedback(task) for task in task_list]
        ### Normalize by task
        print(shapely_value_table)
        shapely_value_table = np.array(shapely_value_table)
        if np.sum(shapely_value_table) < 0:
            # shapely_value_table += shapely_value_table - np.min(shapely_value_table)
            shapely_value_table = np.array([np.array(s_list) / (sum(s_list) if sum(s_list) !=0 else 0.1) for s_list in shapely_value_table])
            shapely_value_table = - shapely_value_table
        else:
            shapely_value_table = [np.array(s_list) / (sum(s_list) if sum(s_list) !=0 else 0.1) for s_list in shapely_value_table]

        print(shapely_value_table)

        ### Update price table
        for task_idx in range(len(task_list)):
            selected_client_index = task_list[task_idx].selected_client_idx
            for idx in range(len(selected_client_index)):
                client_idx = selected_client_index[idx]
                shapley_value = shapely_value_table[task_idx][idx]
                shapely_value_scaled = shapley_value * len(selected_client_index) / NUM_AGENT
                # price_table[client_idx][task_idx] = ((epoch / (epoch + 1)) * price_table[client_idx][task_idx] + (1 / (epoch + 1)) * shapely_value_scaled) 
                price_table[client_idx][task_idx] = shapely_value_scaled 


        assert price_table is not None
    
        ### Update bid table
        bid_table = np.zeros((NUM_AGENT, len(task_list)))

        for task_idx in range(len(task_list)):
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
            if args.policy == "my":
                policy.my_select_clients(price_table, client_feature_list, task_list, bid_table)
            elif args.policy == "random":
                policy.random_select_clients(task_list,NUM_AGENT)
            elif args.policy == "simple":
                policy.simple_select_clients(task_list,NUM_AGENT)
            elif args.policy == "mcafee":
                if epoch == 0:
                    policy.mcafee_select_clients(price_table, client_feature_list, task_list, bid_table)
            else:
                raise
        print("Client assignment Done ")
        
        for task in task_list:
            task.end_of_epoch()

    ### end of trianing
    with open("total_reward_list_{}.json".format(args.policy), 'w') as fp:
        json.dump({"total_reward)=_list": total_reward_list}, fp, indent=4)
    print(reward_sum)
    