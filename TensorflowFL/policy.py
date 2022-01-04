from dataloader import NUM_AGENT
from os import CLD_CONTINUED
import numpy as np
import random 


### By default,  IGNORE_BID_ASK is False, only when the buyer bid is larger that the client ask
#   the client can be selected by this task.
#   NOTE: the client number of a task may not satisfy the required client number
IGNORE_BID_ASK = False

def buyer_give_more_money(client_idx, task_idx, price_table, bid_table):
    if IGNORE_BID_ASK:
        return True
    ### client ask
    buyer_bid = bid_table[client_idx][task_idx]
    seller_ask = price_table[client_idx][task_idx]
    return buyer_bid >= seller_ask

def mcafee_condition(client_idx, task_idx, price_table, buyer_bid):
    if IGNORE_BID_ASK:
        return True
    ### client ask
    seller_ask = price_table[client_idx][task_idx]
    return buyer_bid >= seller_ask

def select_one_client(client_idx, selected_client_index, free_client, _task):
    ### Return True if one task's requirement is satisfied
    selected_client_index.append(client_idx)
    free_client[client_idx] = False
    return len(selected_client_index) >= _task.required_client_num

def check_trade_success_or_not(selected_client_index, _task, free_client):
    if len(selected_client_index) < _task.required_client_num:
        ### Trade failed
        for client_idx in selected_client_index:
            free_client[client_idx] = True
        _task.selected_client_idx = None
        return False
    else:
        ### Successful trade
        _task.selected_client_idx = selected_client_index
        print("Clients {} are assined to task {}".format(selected_client_index, _task.task_id))
        return True

def my_select_clients(ask_table, client_feature_list, task_list, bid_table):
    ''' client_feature_list: list
            a list of (cost, idlecost)
        task_list: list
            a list of class: Task
        bid_table: numpy array
            shape = (client_num, task_num)
    '''
    ### policy
    

    ### shape of task_price_list = (task_num)
    task_price_list = np.sum(bid_table, axis=0)

    sorted_task_with_index = sorted(enumerate(task_price_list), key=lambda x: x[1], reverse=True)
    free_client = [True] * len(client_feature_list)
    for task_idx, _ in sorted_task_with_index:
        _task = task_list[task_idx]
        
        client_value_for_this_task = [client_value_list[task_idx] for client_value_list in ask_table]
        client_value_list_sorted = sorted(enumerate(client_value_for_this_task), key=lambda x: x[1], reverse=True)

        ### Select clients
        selected_client_index = []
        for client_idx, _ in client_value_list_sorted:
            if free_client[client_idx] and buyer_give_more_money(client_idx, task_idx, ask_table, bid_table):
                is_task_ready = select_one_client(client_idx, selected_client_index, free_client, _task)
                if is_task_ready:
                    break
        
        check_trade_success_or_not(selected_client_index, _task, free_client)


def mcafee_select_clients(ask_table, client_feature_list, task_list, bid_table):
    ''' client_feature_list: list
            a list of (cost, idlecost)
        task_list: list
            a list of class: Task
        bid_table: numpy array
            shape = (client_num, task_num)
    '''
    for task in task_list:
        task.selected_client_idx = []

    ### policy

    ### shape of task_price_list = (task_num)
    task_price_list = np.sum(bid_table, axis=0)
    sorted_task_with_index = sorted(enumerate(task_price_list), key=lambda x: x[1], reverse=True)
    client_value_list = np.sum((ask_table), axis=1)
    client_value_list_sorted = sorted(enumerate(client_value_list), key=lambda x: x[1], reverse=False)
    client_num= len(client_value_list)
    task_num = len(task_price_list)
    print("mb", sorted_task_with_index)
    print("ma",client_value_list_sorted)
    print("task#: ", task_num, " client#: ", client_num)

    i = 0
    free_client = [True] * len(client_feature_list)
    while i < task_num:
        task_id = sorted_task_with_index[i][0]
        _task = task_list[task_id]
        trade_succed = False
        assert client_num > 0
        
        j = 0
        selected_client_index = []
        while j < (client_num-1):
            client_id = client_value_list_sorted[j][0]
            try:
                check = free_client[client_id] and sorted_task_with_index[i][1] >= client_value_list_sorted[j][1] \
                    and sorted_task_with_index[i][1] < client_value_list_sorted[j+1][1]
            except:
                import code
                code.interact(local=locals())
            if check:
                is_task_ready = select_one_client(client_id, selected_client_index, free_client, _task)
                ### check whether the requirement of this taks has been met
                if is_task_ready:
                    # raise NotImplementedError("Remove this task away")
                    sorted_task_with_index.pop(i)
                    task_num -= 1
                    trade_succed = True
                    break
                j = 0
                continue
            j += 1
        if not trade_succed:
            raise ValueError("Fail trading")
        ### end of client selection for one task
        check_trade_success_or_not(selected_client_index, _task, free_client)

    ### The original mecafee algorithm
    # while task_num > 0:
    #     assert client_num > 0
    #     trade_succed = False
    #     for k in range(min(client_num, task_num) - 1):
    #         if sorted_task_with_index[k][1] >= client_value_list_sorted[k][1] \
    #             and sorted_task_with_index[k+1][1] < client_value_list_sorted[k+1][1]:
    #             task_id = sorted_task_with_index[k][0]
    #             client_id = client_value_list_sorted[k][0]
    #             task_list[task_id].selected_client_idx.append(client_id)
    #             # raise NotImplementedError("Remove this client away")
    #             client_value_list_sorted.pop(k)
    #             client_num -= 1

    #             ### check whether the requirement of this taks has been met
    #             if len(task_list[task_id].selected_client_idx) == task_list[task_id].required_client_num:
    #                 # raise NotImplementedError("Remove this task away")
    #                 sorted_task_with_index.pop(k)
    #                 task_num -= 1
                
    #             trade_succed = True
    #             break
    #     if not trade_succed:
    #         raise ValueError("Fail trading")

def simple_select_clients(task_list, client_number):
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

def random_select_clients(ask_table, client_feature_list, task_list, bid_table):
    num_of_client = len(client_feature_list)
    free_client = [True] * len(client_feature_list)

    for task_idx, _ in enumerate(task_list):
        _task = task_list[task_idx]
       
        ### Select clients
        selected_client_index = []
        clients_candidates = list(range(num_of_client))
        while len(clients_candidates) > 0:
            client_idx= random.choice(clients_candidates)
            clients_candidates.remove(client_idx)
            if free_client[client_idx] and buyer_give_more_money(client_idx, task_idx, ask_table, bid_table):
                is_task_ready = select_one_client(client_idx, selected_client_index, free_client, _task)
                if is_task_ready:
                    break
        
        check_trade_success_or_not(selected_client_index, _task, free_client)
