import pandas as pd
import numpy as np 
import time 
import multiprocessing as mp 
from functools import partial


#mc = pd.read_pickle('mc_25k.pkl')
trk = pd.read_csv('trk_processed_25k.csv')



n_events = trk['entry'].nunique() 


def build_graph(event_number): 
    
    #select one event 
    event = trk[trk['entry'] == event_number] 
 
    event['unique_trk_id'] = range(len(event)) 
   
    # 85% of samples are not primary vertex, repeat true vertices to rebalance 
    pv = event[event['is_pv']==1]
    not_pv = event[event['is_pv']==0]

    balanced_df = pd.concat([pd.concat([pv]*6), not_pv])
    
    #fully connected edges 
    combos = balanced_df[['entry', 'unique_trk_id']].merge(
        balanced_df[['unique_trk_id', 'entry']], on='entry', suffixes=('_1', '_2'))

    #remove node connection to itself 
    combos = combos[combos['unique_trk_id_1']!= combos['unique_trk_id_2']]

    #edge indices in correct format 
    connections = np.stack((combos['unique_trk_id_1'].values, combos['unique_trk_id_2'].values))
    X = balanced_df[['trk_pt', 'trk_eta', 'trk_phi', 'trk_z0']].values
    

    np.savez("built_graphs/standard/graph_"+str(event_number), ** dict(x=X, edge_index=connections, y=balanced_df['is_pv'].values)) 

    return 

events = list(range(n_events)) 
n_workers = 4

t0 = time.time()
with mp.Pool(processes=n_workers) as pool:
    process_func = partial(build_graph)
    pool.map(process_func, events)
t1 = time.time()
print("Finished in", t1-t0, "seconds")





    

