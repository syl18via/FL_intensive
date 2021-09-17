from __future__ import absolute_import, division, print_function
import numpy as np
import math
import os


NUM_AGENT = 10
BATCH_SIZE = 100
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

def get_data_for_digit_same(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

def get_data_for_digit_mix(source):

    output_sequence = [[] for _ in range(NUM_AGENT)]
    output_sequence_full = [[] for _ in range(NUM_AGENT)]
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
            if label == agent_id:
                for sample_index in range(left, left+int(len(indexs)*0.4)):
                    all_samples[agent_id].append(indexs[sample_index])
                left=left+int(len(indexs)*0.4)
            else:
                for sample_index in range(left, left + int(len(indexs) * 0.05)):
                    all_samples[agent_id].append(indexs[sample_index])
                left = left + int(len(indexs) * 0.05)

    ### Constuct batched dataset and normalize x
    for agent_id, sample_idxs in enumerate(all_samples):
        ### shuffle
        sample_idxs = np.array(sample_idxs)
        # np.random.shuffle(sample_idxs)
        output_sequence_full[agent_id].append({
                'x': np.array([source[0][idx].flatten() / 255.0 for idx in sample_idxs],
                              dtype=np.float32),
                'y': np.array([source[1][idx] for idx in sample_idxs], dtype=np.int32)})

        for i in range(0, len(sample_idxs), BATCH_SIZE):
            batch_sample_idxs = sample_idxs[i:i + BATCH_SIZE]
            output_sequence[agent_id].append({
                'x': np.array([source[0][idx].flatten() / 255.0 for idx in batch_sample_idxs],
                              dtype=np.float32),
                'y': np.array([source[1][idx] for idx in batch_sample_idxs], dtype=np.int32)})
    
    def iterator(agent_id):
        while True:
            for batch in output_sequence[agent_id]:
                yield batch
                
    all_data = [iterator(agent_id) for agent_id in range(NUM_AGENT)]
    return all_data, output_sequence_full


### NoiseY
NOISE_STEP = 0.05
rand_index = []
rand_label = []
with open(os.path.join(os.path.dirname(__file__), "random_index.txt"), "r") as randomIndex:
    lines = randomIndex.readlines()
    for line in lines:
        # print(line)
        rand_index.append(eval(line))
with open(os.path.join(os.path.dirname(__file__), "random_label.txt"), "r") as randomLabel:
    lines = randomLabel.readlines()
    for line in lines:
        rand_label.append(eval(line))

def get_data_for_digit_noiseY(source):
    output_sequence = [[] for _ in range(NUM_AGENT)]
    Samples = []
    for digit in range(0, 10):
        samples = [i for i, d in enumerate(source[1]) if d == digit]
        samples = samples[0:5421]
        Samples.append(samples)

    for client_id, sequence_per_client in enumerate(output_sequence):
        all_samples = []
        for sample in Samples:
            for sample_index in range(int(client_id * (len(sample) / NUM_AGENT)), int((client_id + 1) * (len(sample) / NUM_AGENT))):
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(client_id*(len(source[1])/NUM_AGENT)), int((client_id+1)*(len(source[1])/NUM_AGENT)))]
        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            sequence_per_client.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})
        # add noise 0%-40%
        ratio = NOISE_STEP * (client_id)
        sum_agent = int(len(all_samples))

        #TODO: write random list into file and change randint to a number
        noiseList = rand_index[client_id][0:int(ratio*sum_agent)]
        noiseLabel = rand_label[client_id][0:int(ratio*sum_agent)]
        # noiseList = random.sample(range(0, sum_agent), int(ratio*sum_agent))
        # noiseLabel = []
        index = 0
        for i in noiseList:
            # noiseHere = random.randint(1, 9)
            # noiseLabel.append(noiseHere)
            noiseHere = noiseLabel[index]
            index = index + 1
            sequence_per_client[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE] = (
                sequence_per_client[int(i/BATCH_SIZE)]['y'][i % BATCH_SIZE]+noiseHere) % 10

    def iterator(agent_id):
        while True:
            for batch in output_sequence[agent_id]:
                yield batch
                
    all_data = [iterator(agent_id) for agent_id in range(NUM_AGENT)]
    return all_data


def checkRange(x):
    for i in range(len(x)):
        if x[i] < 0:
            x[i] = 0

        if x[i] > 1:
            x[i] = 1
    return x

def get_data_for_digit_noiseX(source):
    output_sequence = [[] for _ in range(NUM_AGENT)]
    output_sequence_full = [[] for _ in range(NUM_AGENT)]

    Samples = []
    for digit in range(0, 10):
        samples = [i for i, d in enumerate(source[1]) if d == digit]
        samples = samples[0:5421]
        Samples.append(samples)
    
    for client_id, sample_idxs in enumerate(output_sequence_full):
        all_samples = []
        for sample in Samples:
            for sample_index in range(int(client_id * (len(sample) / NUM_AGENT)), int((client_id + 1) * (len(sample) / NUM_AGENT))):
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]
        sample_idxs = np.array(sample_idxs)
        # np.random.shuffle(sample_idxs)
        output_sequence_full[client_id].append({
                'x': np.array([source[0][idx].flatten() / 255.0 for idx in sample_idxs],
                              dtype=np.float32),
                'y': np.array([source[1][idx] for idx in sample_idxs], dtype=np.int32)})

    for client_id, sequence_per_client in enumerate(output_sequence):
        all_samples = []
        for sample in Samples:
            for sample_index in range(int(client_id * (len(sample) / NUM_AGENT)), int((client_id + 1) * (len(sample) / NUM_AGENT))):
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            sequence_per_client.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})

        # add noise 0x-0.2x
        ratio = client_id * 0.05
        sum_agent = int(len(all_samples))
        index = 0
        for i in range(0, sum_agent):
            noiseHere = ratio * np.random.randn(28*28)
            sequence_per_client[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE] = checkRange(np.add(
                sequence_per_client[int(i/BATCH_SIZE)]['x'][i % BATCH_SIZE], noiseHere))

    def iterator(agent_id):
        while True:
            for batch in output_sequence[agent_id]:
                yield batch
                
    all_data = [iterator(agent_id) for agent_id in range(NUM_AGENT)]
    return all_data,output_sequence_full


def get_data_for_digit_test(source, digit):
    output_sequence = []
    all_samples = [i for i, d in enumerate(source[1]) if d == digit]
    for i in range(0, len(all_samples)):
        output_sequence.append({
            'x': np.array(source[0][all_samples[i]].flatten() / 255.0,
                          dtype=np.float32),
            'y': np.array(source[1][all_samples[i]], dtype=np.int32)})
    return output_sequence

def get_data_for_federated_agents(source):
    output_sequence = [[] for _ in range(NUM_AGENT)]

    num_of_piece = (1 + NUM_AGENT) * NUM_AGENT / 2
    
    digit2samples = []
    for digit in range(0, 10):
        samples = [i for i, d in enumerate(source[1]) if d == digit]
        samples = samples[0:5421]
        digit2samples.append(samples)

    for client_id, sequence_per_client in enumerate(output_sequence):
        left = (client_id)*(client_id+1)/2
        right = left + client_id + 1
        all_samples = []
        for sample in digit2samples:
            len_per_piece = len(sample) / num_of_piece
            for sample_index in range(int(left*len_per_piece), int(right*len_per_piece)):
                all_samples.append(sample[sample_index])

        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            sequence_per_client.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})

    def iterator(agent_id):
        while True:
            for batch in output_sequence[agent_id]:
                yield batch
                
    all_data = [iterator(agent_id) for agent_id in range(NUM_AGENT)]
    return all_data

def get_data_evenly(source):
    output_sequence = [[] for _ in range(NUM_AGENT)]

    sample_indexs = np.array(range(len(source[1])))
    np.random.shuffle(sample_indexs)
    sample_num_per_client = math.floor(len(source[1]) / NUM_AGENT)

    for client_id, sequence_per_client in enumerate(output_sequence):
        left = client_id * sample_num_per_client
        right = left + sample_num_per_client
        all_samples = sample_indexs[left:right]
        # all_samples = [i for i in range(int(num*(len(source[1])/NUM_AGENT)), int((num+1)*(len(source[1])/NUM_AGENT)))]

        for i in range(0, len(all_samples), BATCH_SIZE):
            batch_samples = all_samples[i:i + BATCH_SIZE]
            sequence_per_client.append({
                'x': np.array([source[0][i].flatten() / 255.0 for i in batch_samples],
                            dtype=np.float32),
                'y': np.array([source[1][i] for i in batch_samples], dtype=np.int32)})

    def iterator(agent_id):
        while True:
            for batch in output_sequence[agent_id]:
                yield batch
                
    all_data = [iterator(agent_id) for agent_id in range(NUM_AGENT)]
    return all_data

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
    return np.asarray(ret, dtype=np.float32)

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
    return np.asarray(ret, dtype=np.int32)


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

def get_test_images_labels(distr_type):
    test_images = None
    test_labels_onehot = None
    if distr_type == "SAME":
        test_images = readTestImagesFromFile(True)
        test_labels_onehot = readTestLabelsFromFile(True)
    else:
        test_images = readTestImagesFromFile(False)
        test_labels_onehot = readTestLabelsFromFile(False)
    return test_images, test_labels_onehot
