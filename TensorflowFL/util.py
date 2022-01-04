import numpy as np

def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def calcualte_client_value(price_table, client_feature_list):
    ''' price_table is a 2-D list, shape=(#client, #task)
    price_table[client_id][task_id] is the price of client_id for task_id
    '''
    idle_cost_list = [feature[1] for feature in client_feature_list]
    value_table = []
    for client_idx in range(len(client_feature_list)):
        client_price_list = price_table[client_idx]
        value_list = [price / (idle_cost_list[client_idx]+1) for price in  client_price_list]
        value_table.append(value_list)
    return value_table