from __future__ import absolute_import, division, print_function
import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import time
from scipy.special import comb, perm
import random

import os

### self-defined module
import policy

# tf.compat.v1.enable_v2_behavior()
# tf.compat.v1.enable_eager_execution()

# NUM_EXAMPLES_PER_USER = 1000
BATCH_SIZE = 100
NUM_AGENT = 5
MIX_RATIO = 0.8
SIMULATE = True

def get_data_for_digit(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    return output_sequence

def get_data_for_digit_mix(source):
    output_sequence = [[] for _ in range(NUM_AGENT)]
    # print(source[0].shape, source[1].shape)
    # (60000, 28, 28) (60000,)

    ### Samples is a list, each element is a list of sample indexs
    ### E.g, Samples[i] is the list of indexs whose label is i
    label2indexs = []
    for digit in range(0, 10):
        indexs = [i for i, label in enumerate(source[1]) if label == digit]
        indexs = indexs[0:5421]
        label2indexs.append(indexs)

    ### Construct an imbalanced dataset
    ### For the agent with `agent_id`, it corresponds to 
    #   * 2 major classes with labels of agent_id*2 and agent_id*2+1
    #   * other 8 minor classes
    # For a major classe and minor class, the data ratio is 80:5
    all_samples = [[] for _ in range(NUM_AGENT)]
    for label, indexs in enumerate(label2indexs):
        left =0
        for agent_id in range(NUM_AGENT):
            if label == agent_id*2 or label == agent_id*2+1:
                for sample_index in range(left, left+int(len(indexs)*0.8)):
                    all_samples[agent_id].append(indexs[sample_index])
                left=left+int(len(indexs)*0.8)
            else:
                for sample_index in range(left, left + int(len(indexs) * 0.05)):
                    all_samples[agent_id].append(indexs[sample_index])
                left = left + int(len(indexs) * 0.05)

    ### Constuct batched dataset and normalize x
    for agent_id, sample_idxs in enumerate(all_samples):
        for i in range(0, len(sample_idxs), BATCH_SIZE):
            batch_sample_idxs = sample_idxs[i:i + BATCH_SIZE]
            output_sequence[agent_id].append({
                'x': np.array([source[0][idx].flatten() / 255.0 for idx in batch_sample_idxs],
                              dtype=np.float32),
                'y': np.array([source[1][idx] for idx in batch_sample_idxs], dtype=np.int32)})
    return output_sequence

def get_data_for_digit_test(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

def get_data_for_federated_agents(source, num):
    output_sequence = []
    all_samples = [i for i in
                   range(int(num * (len(source[1]) / NUM_AGENT)), int((num + 1) * (len(source[1]) / NUM_AGENT)))]

    '''all_samples = None
    if num == 0:
        all_samples = [i for i in range(0, 1000)]
    elif num == 1:
        all_samples = [i for i in range(3000, 4000)]
    else:
        all_samples = [i for i in range(9000, 10000)]'''

    for i in range(0, len(all_samples), BATCH_SIZE):
        batch_samples = all_samples[i:i + BATCH_SIZE]
        output_sequence.append({
            'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                          dtype=np.float32),
            'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
    return output_sequence


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


def readTestImagesFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_images1_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split("\t")
        for i in p:
            if i != "":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)

def readTestLabelsFromFile(distr_same):
    ret = []
    if distr_same:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    else:
        f = open(os.path.join(os.path.dirname(__file__), "test_labels_.txt"), encoding="utf-8")
    lines = f.readlines()
    for line in lines:
        tem_ret = []
        p = line.replace("[", "").replace("]", "").replace("\n", "").split(" ")
        for i in p:
            if i!="":
                tem_ret.append(float(i))
        ret.append(tem_ret)
    return np.asarray(ret)


def getParmsAndLearningRate(agent_no):
    f = open(os.path.join(os.path.dirname(__file__), "weights_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    parm_local = []
    learning_rate_list = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0:784]
            learning_rate_list.append(float(line[784].replace("*", "").replace("\n", "")))
        else:
            weights_line = line[1:785]
            learning_rate_list.append(float(line[785].replace("*", "").replace("\n", "")))
        valid_weights_line = []
        for l in weights_line:
            w_list = l.split("\t")
            w_list = w_list[0:len(w_list) - 1]
            w_list = [float(i) for i in w_list]
            valid_weights_line.append(w_list)
        parm_local.append(valid_weights_line)
    f.close()

    f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(agent_no) + ".txt"))
    content = f.read()
    g_ = content.split("***\n--------------------------------------------------")
    bias_local = []
    for j in range(len(g_) - 1):
        line = g_[j].split("\n")
        if j == 0:
            weights_line = line[0]
        else:
            weights_line = line[1]
        b_list = weights_line.split("\t")
        b_list = b_list[0:len(b_list) - 1]
        b_list = [float(i) for i in b_list]
        bias_local.append(b_list)
    f.close()
    ret = {
        'weights': np.asarray(parm_local),
        'bias': np.asarray(bias_local),
        'learning_rate': np.asarray(learning_rate_list)
    }
    return ret


def train_with_gradient_and_valuation(agent_list, grad, bi, lr, distr_type, datanum):
    f_ini_p = open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r")
    para_lines = f_ini_p.readlines()
    w_paras = para_lines[0].split("\t")
    w_paras = [float(i) for i in w_paras]
    b_paras = para_lines[1].split("\t")
    b_paras = [float(i) for i in b_paras]
    w_initial_g = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
    b_initial_g = np.asarray(b_paras, dtype=np.float32).reshape([10])
    f_ini_p.close()
    model_g = {
        'weights': w_initial_g,
        'bias': b_initial_g
    }
    data_sum = 0
    for i in agent_list:
        data_sum += datanum[i]
    agents_w = [0 for _ in range(NUM_AGENT)]
    for i in agent_list:
        agents_w[i] = datanum[i]/data_sum
    for i in range(len(grad[0])):
        # i->迭代轮数
        gradient_w = np.zeros([784, 10], dtype=np.float32)
        gradient_b = np.zeros([10], dtype=np.float32)
        for j in agent_list:
            gradient_w = np.add(np.multiply(grad[j][i], agents_w[j]), gradient_w)
            gradient_b = np.add(np.multiply(bi[j][i], agents_w[j]), gradient_b)
        model_g['weights'] = np.subtract(model_g['weights'], np.multiply(lr[0][i], gradient_w))
        model_g['bias'] = np.subtract(model_g['bias'], np.multiply(lr[0][i], gradient_b))

    test_images = None
    test_labels_onehot = None
    if distr_type == "SAME":
        test_images = readTestImagesFromFile(True)
        test_labels_onehot = readTestLabelsFromFile(True)
    else:
        test_images = readTestImagesFromFile(False)
        test_labels_onehot = readTestLabelsFromFile(False)
    m = np.dot(test_images, np.asarray(model_g['weights']))
    # print(m.shape)
    test_result = m + np.asarray(model_g['bias'])
    y = tf.nn.softmax(test_result)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(test_labels_onehot, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.numpy()


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

class Task:
    def __init__(self, task_id, selected_client_idx, model,
            required_client_num=None,
            bid_per_loss_delta=None):
        self.selected_client_idx = selected_client_idx    # a list of client indexes of selected clients
        self.model = model  # model parameters
        self.model_epoch_start = None # model parameters at the start of an epoch
        self.task_id = task_id
        self.learning_rate = None
        self.epoch = 0
        self.required_client_num = required_client_num
        self.bid_per_loss_delta = bid_per_loss_delta

        self.totoal_loss_delta = None
        self.params_per_client = [None] * len(selected_client_idx)
    
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

if __name__ == "__main__":
    start_time = time.time()

    data_num = np.asarray([5421 * 2] * NUM_AGENT)
    agents_weights = np.divide(data_num, data_num.sum())

    for index in range(NUM_AGENT):
        f = open(os.path.join(os.path.dirname(__file__), "weights_"+str(index)+".txt"), "w")
        f.close()
        f = open(os.path.join(os.path.dirname(__file__), "bias_" + str(index) + ".txt"), "w")
        f.close()
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data()

    DISTRIBUTION_TYPE = "DIFF"

    all_client_data_divide = None
    all_client_data = None
    if DISTRIBUTION_TYPE == "SAME":
        exit(0)
        #all_client_data_divide = [get_data_for_federated_agents(mnist_train, d) for d in range(NUM_AGENT)]
        #test_images = readTestImagesFromFile(True)
        #test_labels_onehot = readTestLabelsFromFile(True)
    else:
        all_client_data_divide = get_data_for_digit_mix(mnist_train)
        all_client_data = all_client_data_divide
    
    ### 0 denotes free, 1 denotes being occupied
    free_client = [0] * NUM_AGENT
    
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

    learning_rate = 0.1

    ### Initialize the global model parameters for both tasks
    ### At the first epoch, both tasks select all clients
    task0 = Task(
        task_id = 0,
        selected_client_idx=list(range(NUM_AGENT)),
        model = {
                'weights': w_initial,
                'bias': b_initial
        },
        required_client_num=3,
        bid_per_loss_delta=2)
    task1 = Task(
        task_id = 1,
        selected_client_idx=list(range(NUM_AGENT)),
        model = {
                'weights': w_initial,
                'bias': b_initial
        },
        required_client_num=2,
        bid_per_loss_delta=1
        )


    cost_list = []
    for client_idx in range(NUM_AGENT):
        cost_list.append(random.randint(1,10))
    
    
    idlecost_list = []
    for client_idx in range(NUM_AGENT):
        idlecost_list.append(random.random())

    client_feature_list = list(zip( cost_list, idlecost_list))

    task_list = [task0, task1]

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

    def pick_client_based_on_index(selected_client_idx):
        clients_data = []
        for idx in selected_client_idx:
            clients_data.append(all_client_data[idx])
        return clients_data
    
    def train_one_round(task, round_idx, learning_rate, epoch, ckpt=False, evaluate_each_client=False):
        clients_data = pick_client_based_on_index(task.selected_client_idx)
        if SIMULATE:
            local_models = [(task.model['weights'], task.model['bias']) for _ in task.selected_client_idx]
        else:
            local_models = federated_train(task.model, learning_rate, clients_data)
        
        ### Output model parameters of all selected agents of this task
        ### Store the model parameters of this round if ckpt is True
        
        for local_index in range(len(local_models)):
            client_id = task.selected_client_idx[local_index]
            if epoch == 0:
                if task.params_per_client[client_id] is None:
                    task.params_per_client[client_id] = []
                task.params_per_client[client_id].append(
                    (local_models[local_index], learning_rate))
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
        
        ### Calculate the global model parameters
        m_w = np.zeros([784, 10], dtype=np.float32)
        m_b = np.zeros([10], dtype=np.float32)
        task.log("Round {} at {:.3f} s, learning rate: {:.3f}".format(round_idx, time.time()-start_time, learning_rate))
        for local_model_index in range(len(local_models)):
            client_id = task.selected_client_idx[local_model_index] 
            m_w = np.add(np.multiply(local_models[local_model_index][0], agents_weights[local_model_index]), m_w)
            m_b = np.add(np.multiply(local_models[local_model_index][1], agents_weights[local_model_index]), m_b)

            if evaluate_each_client:
            ### Evaluate the model parameters of current agent if `evaluate_each_client` is True
                loss = federated_eval(
                    {
                    'weights': local_models[local_model_index][0],
                    'bias': local_models[local_model_index][1]
                    }, 
                    clients_data)
                print(' - agent {}, loss={}'.format(client_id, loss))
        
        ### Update the global model parameters
        if task.model_epoch_start is None:
            task.model_epoch_start = task.model
        task.model = {
                'weights': m_w,
                'bias': m_b
        }
        task.learning_rate = learning_rate
        loss = federated_eval(task.model, clients_data)
        
        #raise NotImplementedError("TODO, update task.total_loss_delta")

    def calculate_feedback(task):
        ### TODO: comment the process to calculate the shapley value
        ### Substitute with our own algorithm
        
        ### Calculate the Feedback
        gradient_weights = []
        gradient_biases = []
        gradient_lrs = []

        assert len(task.params_per_client) == NUM_AGENT
        for client_id in range(NUM_AGENT):
            # model_ = getParmsAndLearningRate(client_id)
            # round_num = len(model_['learning_rate'])

            params_learning_rate_list = task.params_per_client[client_id]
            round_num = len(params_learning_rate_list)

            gradient_weights_local = []
            gradient_biases_local = []
            learning_rate_local = []

            for round_idx in range(round_num):
                # _weight = model_['weights'][round_idx]
                # _bias = model_['bias'][round_idx]
                # _prev_weight = initial_model['weights'] if round_idx == 0 else model_['weights'][round_idx-1]
                # _prev_bias = initial_model['bias'] if round_idx == 0 else model_['bias'][round_idx-1]
                # _learning_rate = model_['learning_rate'][round_idx]

                _weight = params_learning_rate_list[round_idx][0][0]
                _bias = params_learning_rate_list[round_idx][0][1]
                _prev_weight = task.model_epoch_start['weights'] if round_idx == 0 else params_learning_rate_list[round_idx-1][0][0]
                _prev_bias = task.model_epoch_start['bias'] if round_idx == 0 else params_learning_rate_list[round_idx-1][0][1]
                _learning_rate = params_learning_rate_list[round_idx][1]

                gradient_weight = np.divide(np.subtract(_prev_weight, _weight), _learning_rate)
                gradient_bias = np.divide(np.subtract(_prev_bias, _bias), _learning_rate)

                gradient_weights_local.append(gradient_weight)
                gradient_biases_local.append(gradient_bias)
                learning_rate_local.append(_learning_rate)

            gradient_weights.append(gradient_weights_local)
            gradient_biases.append(gradient_biases_local)
            gradient_lrs.append(learning_rate_local)

        all_sets = PowerSetsBinary([i for i in range(NUM_AGENT)])
        group_shapley_value = []
        for s in all_sets:
            if SIMULATE:
                contrib = 1
            else:
                contrib = train_with_gradient_and_valuation(s, gradient_weights, gradient_biases, gradient_lrs, DISTRIBUTION_TYPE, data_num)

            group_shapley_value.append(contrib)
            # task.log(str(s)+"\t"+str(group_shapley_value[len(group_shapley_value)-1]))

        agent_shapley = []
        for index in range(NUM_AGENT):
            shapley = 0.0
            for j in all_sets:
                if index in j:
                    remove_list_index = remove_list_indexed(index, j, all_sets)
                    if remove_list_index != -1:
                        shapley += (group_shapley_value[shapley_list_indexed(j, all_sets)] - group_shapley_value[
                            remove_list_index]) / (comb(NUM_AGENT - 1, len(all_sets[remove_list_index])))
            agent_shapley.append(shapley)
        # for ag_s in agent_shapley:
        #     print(ag_s)

        # task.select_clients(agent_shapley, free_client)
        return agent_shapley
        
    EPOCH_NUM = 50
    ### Main process of FL
    for epoch in range(EPOCH_NUM):
        task0.epoch = task1.epoch = epoch
        for round_idx in range(1):
            ### Train the model parameters distributedly
            #   return a list of model parameters
            #       local_models[0][0], weights of the 0-th agent
            #       local_models[0][1], bias of the 0-th agent

            ### Task 0
            train_one_round(task0, round_idx, learning_rate, epoch, ckpt=False, evaluate_each_client=False)
            
            ### Task1
            train_one_round(task1, round_idx, learning_rate, epoch, ckpt=False, evaluate_each_client=False)
    
            learning_rate = learning_rate * 0.9

        ### At the end of this epoch
        ### At the first epoch, calculate the Feedback and update clients for each task
        
        print("Start to update client assignment ... ")
        ### Update price table
        price_table = init_price_table(price_table)
        assert price_table is not None

        shapely_value_table = [calculate_feedback(task) for task in task_list]
        for task_idx in range(len(task_list)):
            selected_client_index = task_list[task_idx].selected_client_idx
            for idx in range(len(selected_client_index)):
                client_idx = selected_client_index[idx]
                shapley_value = shapely_value_table[task_idx][idx]
                price_table[client_idx][task_idx] = shapley_value


        ### TOD  
        task_price_list = [random.random() for task in task_list]
        ### task_price_list = [total_bid / task.required_client_num for task in task_list]
        # total_bid_list = [task.totoal_loss_delta * task.bid_per_loss_delta for task in task_list]

        print("Start to select clients ... ")
        policy.select_clients(price_table, client_feature_list, task_list, task_price_list)
        print("Client assignment Done ")
        
        task0.end_of_epoch()
        task1.end_of_epoch()
        

        
