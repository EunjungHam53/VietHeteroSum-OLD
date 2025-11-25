import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import *
import numpy as np
import torch

def mask_to_adj(doc_sect_mask, sect_sent_mask):
    """
    Create adjacency matrix for the hierarchical graph
    doc_sect_mask: (doc_num, sect_num) - connection between documents and sections
    sect_sent_mask: (sect_num, sent_num) - connection between sections and sentences
    """
    doc_sect_mask = np.array(doc_sect_mask)
    sect_sent_mask = np.array(sect_sent_mask)

    sent_num = sect_sent_mask.shape[1]
    sect_num = sect_sent_mask.shape[0]
    doc_num = doc_sect_mask.shape[0]

    total_nodes = sent_num + sect_num + doc_num
    adj = np.zeros((total_nodes, total_nodes))
    
    # Connect sections to sentences
    adj[sent_num:sent_num+sect_num, 0:sent_num] = sect_sent_mask
    adj[0:sent_num, sent_num:sent_num+sect_num] = sect_sent_mask.T
    
    # Connect documents to sections
    adj[sent_num+sect_num:, sent_num:sent_num+sect_num] = doc_sect_mask
    adj[sent_num:sent_num+sect_num, sent_num+sect_num:] = doc_sect_mask.T
    
    # Self-connections within same level (sections within same document)
    for i in range(doc_num):
        doc_mask = doc_sect_mask[i]
        if doc_mask.ndim == 1:
            doc_mask = doc_mask.reshape((1, -1))
        elif doc_mask.ndim == 0:
            doc_mask = np.array([doc_mask])
        
        # Connect sections that belong to the same document
        adj[sent_num:sent_num+sect_num, sent_num:sent_num+sect_num] += doc_mask * doc_mask.T
    
    # Self-connections within same level (sentences within same section)
    for i in range(sect_num):
        sect_mask = sect_sent_mask[i]
        if sect_mask.ndim == 1:
            sect_mask = sect_mask.reshape((1, -1))
        elif sect_mask.ndim == 0:
            sect_mask = np.array([sect_mask])
        
        # Connect sentences that belong to the same section
        adj[0:sent_num, 0:sent_num] += sect_mask * sect_mask.T
    
    # Self-connections at document level
    adj[sent_num+sect_num:, sent_num+sect_num:] = 1

    return adj

class Graph:
    def __init__(self, sents, sentVecs, scores, doc_sec_mask, sec_sen_mask, golden, threds=0.5):
        assert len(sentVecs) == len(scores) == len(sents)
        self.docnum = len(doc_sec_mask)
        self.secnum = len(sec_sen_mask)
        self.adj = torch.from_numpy(mask_to_adj(doc_sec_mask, sec_sen_mask)).float()
        self.feature = np.concatenate((np.array(sentVecs), np.zeros((self.secnum+self.docnum+1, sentVecs[0].size))))
        self.score = torch.from_numpy(np.array(scores)).float()
        self.score_onehot = (self.score >= threds).float()
        self.sents = np.array(sents)
        self.golden = golden
        self.goldenVec = get_phoBert_vec(golden)
        self.init_node_vec()
        self.feature = torch.from_numpy(self.feature).float()

    def init_node_vec(self):
        docnum, secnum = self.docnum, self.secnum
        for i in range(-secnum-docnum-1, -docnum-1):
            mask = self.adj[i].clone()
            mask[-secnum-docnum-1:] = 0
            self.feature[i] = np.mean(self.feature[mask.bool()], axis=0)
        for i in range(-docnum-1, -1):
            mask = self.adj[i].clone()
            mask[-docnum-1:] = 0
            self.feature[i] = np.mean(self.feature[mask.bool()], axis=0)
        self.feature[-1] = np.mean(self.feature[-docnum-1:-1], axis=0)