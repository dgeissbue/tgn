import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess_edges(edge_data):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(edge_data) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)

def preprocess_nodes(node_data):
  idx_list = []
  feat_l = []

  with open(node_data) as f:
    for idx, line in enumerate(f):
      idx_list.append(idx+1)
      feat = np.array([float(x) for x in line.split(',')])
      feat_l.append(feat)

  return idx_list, np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df

def run(data_name, edge_data, node_data, save_path, bipartite=True):
  Path(save_path).mkdir(parents=True, exist_ok=True)
  PATH_EDGE = edge_data #./data/{}.csv'.format(data_name)
  PATH_NODES = node_data 
  OUT_DF = os.path.join(save_path,'ml_{}.csv'.format(data_name))
  OUT_EDGE_FEAT = os.path.join(save_path,'ml_{}.npy'.format(data_name))
  OUT_NODE_FEAT = os.path.join(save_path,'ml_{}_node.npy'.format(data_name))

  df, edge_feat = preprocess_edges(PATH_EDGE)
  new_df = reindex(df, bipartite)
  empty = np.zeros(edge_feat.shape[1])[np.newaxis, :]
  edge_feat = np.vstack([empty, edge_feat])

  if PATH_NODES is not None:
    idx_list, node_feat = preprocess_nodes(PATH_NODES)
    node_feat = np.vstack([np.zeros(node_feat.shape[1]), node_feat])
  else :
    max_idx = max(new_df.u.max(), new_df.i.max())
    node_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_EDGE_FEAT, edge_feat)
  np.save(OUT_NODE_FEAT, node_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data_name', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--edge_data', type=str, help='Path to edge data',
                    default='./data/wikipedia.csv')
parser.add_argument('--node_data', type=str, help='Path to node data', default=None)
parser.add_argument('--save_path', type=str, help='Path to save preprocessed data', default='./data/')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data_name, args.edge_data, args.node_data, args.save_path, bipartite=args.bipartite)