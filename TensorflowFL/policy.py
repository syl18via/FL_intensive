### By default,  IGNORE_BID_ASK is False, only when the buyer bid is larger that the client ask
#   the client can be selected by this task.
#   NOTE: the client number of a task may not satisfy the required client number
IGNORE_BID_ASK = False

def calcualte_client_value(price_table, client_feature_list):
    ''' price_table is a 2-D list, shape=(#client, #task)
    price_table[client_id][task_id] is the price of client_id for task_id
    '''
    cost_list = [feature[1] for feature in client_feature_list]
    value_table = []
    for client_idx in range(len(client_feature_list)):
        client_price_list = price_table(client_idx)
        value_list = [price / cost_list[client_idx] for price in  client_price_list]
        value_table.append(value_list)
    return value_table

def buyer_give_more_money(client_idx, task_idx, price_table, buyer_bid):
    if IGNORE_BID_ASK:
        return True
    ### client ask
    seller_ask = price_table[client_idx][task_idx]
    return buyer_bid >= seller_ask

def select_clients(price_table, client_feature_list, task_list, task_price_list):
    ''' client_feature_list: list
            a list of (quantity, cost, idlecost)
        task_list: list
            a list of class: Task
    '''
    ### policy
    client_value_table = calcualte_client_value(price_table, client_feature_list)

    sorted_task_with_index = sorted(enumerate(task_price_list), key=lambda x: x[1], reverse=True)
    free_client = [True] * len(client_feature_list)
    for task_idx, buyer_bid in sorted_task_with_index:
        _task = task_list[task_idx]
        num_of_client = _task.required_client_num
        
        client_value_for_this_task = [client_value_list[task_idx] for client_value_list in client_value_table]
        client_value_list_sorted = sorted(enumerate(client_value_for_this_task), key=lambda x: x[1], reverse=True)

        ### Select clients
        selected_client_index = []
        for client_idx, _ in client_value_list_sorted:
            if free_client[client_idx] and buyer_give_more_money(client_idx, task_idx, price_table, buyer_bid):
                selected_client_index.append(client_idx)
                free_client[client_idx] = False
            if len(selected_client_index) >= num_of_client:
                break
        _task.selected_client_idx = selected_client_index