import tensorflow_federated as tff
import tensorflow.compat.v1 as tf
import numpy as np
import os

### FL config
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

def init_model():
    with open(os.path.join(os.path.dirname(__file__), "initial_model_parameters.txt"), "r") as f_ini_p:
        para_lines = f_ini_p.readlines()
        w_paras = para_lines[0].split("\t")
        w_paras = [float(i) for i in w_paras]
        b_paras = para_lines[1].split("\t")
        b_paras = [float(i) for i in b_paras]
        w_initial = np.asarray(w_paras, dtype=np.float32).reshape([784, 10])
        b_initial = np.asarray(b_paras, dtype=np.float32).reshape([10])
    return w_initial, b_initial