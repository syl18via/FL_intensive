from os import CLD_CONTINUED
import numpy as np


### By default,  IGNORE_BID_ASK is False, only when the buyer bid is larger that the client ask
#   the client can be selected by this task.
#   NOTE: the client number of a task may not satisfy the required client number
IGNORE_BID_ASK = True

def calcualte_client_value(price_table, client_feature_list):
    ''' price_table is a 2-D list, shape=(#client, #task)
    price_table[client_id][task_id] is the price of client_id for task_id
    '''
    idle_cost_list = [feature[1] for feature in client_feature_list]
    value_table = []
    for client_idx in range(len(client_feature_list)):
        client_price_list = price_table[client_idx]
        value_list = [price / idle_cost_list[client_idx] for price in  client_price_list]
        value_table.append(value_list)
    return value_table

def buyer_give_more_money(client_idx, task_idx, price_table, buyer_bid):
    if IGNORE_BID_ASK:
        return True
    ### client ask
    seller_ask = price_table[client_idx][task_idx]
    return buyer_bid >= seller_ask

def my_select_clients(price_table, client_feature_list, task_list, bid_table):
    ''' client_feature_list: list
            a list of (cost, idlecost)
        task_list: list
            a list of class: Task
        bid_table: numpy array
            shape = (client_num, task_num)
    '''
    ### policy
    client_value_table = calcualte_client_value(price_table, client_feature_list)


    ### shape of task_price_list = (task_num)
    task_price_list = np.sum(bid_table, axis=0)

    sorted_task_with_index = sorted(enumerate(task_price_list), key=lambda x: x[1], reverse=True)
    free_client = [True] * len(client_feature_list)
    for task_idx, _ in sorted_task_with_index:
        _task = task_list[task_idx]
        num_of_client = _task.required_client_num
        
        client_value_for_this_task = [client_value_list[task_idx] for client_value_list in client_value_table]
        client_value_list_sorted = sorted(enumerate(client_value_for_this_task), key=lambda x: x[1], reverse=True)

        ### Select clients
        selected_client_index = []
        for client_idx, _ in client_value_list_sorted:
            buyer_bid = bid_table[client_idx][task_idx]
            if free_client[client_idx] and buyer_give_more_money(client_idx, task_idx, price_table, buyer_bid):
                selected_client_index.append(client_idx)
                free_client[client_idx] = False
            if len(selected_client_index) >= num_of_client:
                break
        _task.selected_client_idx = selected_client_index
        print("Clients {} are assined to task {}".format(selected_client_index, task_idx))

def random_select_clients(task_list, client_number):
    free_client = [True] * client_number
    for task_idx, _ in enumerate(task_list):
        _task = task_list[task_idx]
        num_of_client = _task.required_client_num

        ### Select clients
        selected_client_index = []
        for client_idx in range(client_number):
            if free_client[client_idx]:
                selected_client_index.append(client_idx)
                free_client[client_idx] = False
            if len(selected_client_index) >= num_of_client:
                break
        _task.selected_client_idx = selected_client_index
        print("Clients {} are assined to task {}".format(selected_client_index, task_idx)) 