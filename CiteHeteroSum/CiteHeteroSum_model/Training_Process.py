import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from CiteHeteroSum_model.Loss import pairwise_distances
from utils import *
from torch.amp import autocast
from torch.amp import GradScaler

scaler = GradScaler('cuda')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def concat_with_overlap(str1, str2):
    tokens1 = str1.split()
    tokens2 = str2.split()

    overlap_length = 0
    for i in range(1, min(len(tokens1), len(tokens2)) + 1):
        if tokens1[-i:] == tokens2[:i]:
            overlap_length = i

    merged_tokens = tokens1 + tokens2[overlap_length:]
    return ' '.join(merged_tokens)

def train_e2e_moe(train_dataloader, model, optimizer, args=args):
    model[0].train()
    model[1].train()
    c_loss, s_loss, loss, batch_num = 0, 0, 0, 0
    print_epo = 20
    rouge2_score = []
    moe_loss = 0

    for i, data in enumerate(train_dataloader):
        batch_loss, bc_loss, bs_loss, b_moe_loss, scores = train_e2e_batch_moe(data, model, optimizer)
        loss += batch_loss
        c_loss += bc_loss
        s_loss += bs_loss
        moe_loss += b_moe_loss
        batch_num += 1

        abs_text = data.golden
        summary_text = get_summary(scores[0], data.sents, args["max_word_num"], kappa=args["kappa"])
        rouge2_score.append(getRouge2(data.golden, summary_text, 'f'))

        if i % print_epo == 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            print("Batch {}, C-Loss: {}".format(i, c_loss / batch_num))
            print("Batch {}, S-Loss: {}".format(i, s_loss / batch_num))
            print("Batch {}, MoE-Loss: {}".format(i, moe_loss / batch_num))

            print(f"Main contribution: {model[0].graph_encoder.get_main_contribution_ratio():.2%}")
            if model[0].graph_encoder.use_dynamic_topk:
                topk_stats = model[0].graph_encoder.get_topk_statistics()
                print(f"Dynamic Top-K: {topk_stats}")

    return loss / batch_num, np.mean(rouge2_score)

def train_e2e_batch_moe(data_batch, model, optimizer):
    c_model = model[0]
    s_model = model[1]

    optimizer.zero_grad()
    feature = data_batch.feature.unsqueeze(0)
    adj = data_batch.adj.unsqueeze(0)
    docnum = data_batch.docnum
    secnum = data_batch.secnum
    labels = data_batch.score_onehot.unsqueeze(0)
    scores = data_batch.score.unsqueeze(0).float()
    goldenVec = data_batch.goldenVec
    
    # Lấy device từ feature
    device = feature.device

    pg = c_model(feature.cuda(), adj.cuda(), docnum, secnum)
    x = s_model(pg.cuda(), adj.cuda(), docnum, secnum)

    s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(10).cuda())
    pg = pg.squeeze(0)
    infonce = InfoNCE(tau=0.2)

    # Create positive mask
    pos_mask = torch.zeros(1, feature.shape[1])
    pos_mask[:, :-docnum-secnum-1] = labels
    pos_mask[:, -1] = 1
    
    # Create negative mask (inverse of positive mask)
    neg_mask = 1 - pos_mask
    # c_loss = infonce(goldenVec.cuda(), pg, mask.cuda(), neg_mask.cuda())

    # Use triplet loss instead of InfoNCE
    if torch.any(pos_mask):
        pos_distances, neg_distances = pairwise_distances(goldenVec.cuda(), pg, pos_mask.cuda(), neg_mask.cuda())
        c_loss = loss_method(pos_distances, neg_distances)
    else:
        c_loss = torch.tensor(0.0, device='cuda')

    # Lấy MoE loss từ cả 2 model c và e (s)
    # Model C
    moe_loss_c = c_model.graph_encoder.get_contribution_loss() if hasattr(c_model.graph_encoder, 'get_contribution_loss') else torch.tensor(0.0, device=device)
    if not isinstance(moe_loss_c, torch.Tensor):
        moe_loss_c = torch.tensor(moe_loss_c, device=device)
    
    # Model E (S)
    # moe_loss_e = s_model.graph_encoder.get_load_balance_loss() if hasattr(s_model.graph_encoder, 'get_load_balance_loss') else torch.tensor(0.0, device=device)
    # if not isinstance(moe_loss_e, torch.Tensor):
    #     moe_loss_e = torch.tensor(moe_loss_e, device=device)
    
    moe_loss = moe_loss_c

    loss = s_loss + 1. * c_loss + moe_loss
    loss.backward()
    optimizer.step()

    return loss.data, c_loss.data, s_loss.data, moe_loss.data, torch.sigmoid(x.squeeze(-1))

def val_e2e_moe(val_dataloader, model, args=args, mode='val', sent_num=0):
    model[0].eval()
    model[1].eval()
    loss, c_loss, s_loss = 0, 0, 0
    batch_num = 0
    rouge2_score = []
    moe_loss = 0

    all_summaries = []
    all_gt = []
    for i, data in enumerate(val_dataloader):
        cur_loss, c_loss_b, s_loss_b, moe_loss_b, scores = val_e2e_batch_moe(data, model)
        loss += cur_loss
        c_loss += c_loss_b
        s_loss += s_loss_b
        moe_loss += moe_loss_b

        abs_text = data.golden
        summary_text = get_summary(scores[0], data.sents, args["max_word_num"], kappa=args["kappa"])

        all_gt.append(data.golden)
        all_summaries.append(summary_text)
        rouge2_score.append(getRouge2(data.golden, summary_text, 'f'))
        batch_num += 1

    rouge2_score_mean = np.mean(rouge2_score)
    loss = loss / batch_num
    c_loss /= batch_num
    s_loss /= batch_num
    moe_loss /= batch_num

    if mode != 'val':
        return rouge2_score_mean, all_summaries, all_gt, rouge2_score
    return rouge2_score_mean, loss, c_loss, s_loss, moe_loss

def val_e2e_batch_moe(data_batch, model):
    c_model = model[0]
    s_model = model[1]
    feature = data_batch.feature.unsqueeze(0)
    adj = data_batch.adj.unsqueeze(0)
    docnum = data_batch.docnum
    secnum = data_batch.secnum
    labels = data_batch.score_onehot.unsqueeze(0)
    scores = data_batch.score.unsqueeze(0).float()
    goldenVec = data_batch.goldenVec
    
    # Lấy device từ feature
    device = feature.device

    with torch.no_grad():
        pg = c_model(feature.cuda(), adj.cuda(), docnum, secnum)
        x = s_model(pg.cuda(), adj.cuda(), docnum, secnum)

        pg = pg.squeeze(0)
        infonce = InfoNCE(tau=0.2)

        # Create positive mask
        pos_mask = torch.zeros(1, feature.shape[1])
        pos_mask[:, :-docnum-secnum-1] = labels
        pos_mask[:, -1] = 1
        
        # Create negative mask (inverse of positive mask)
        neg_mask = 1 - pos_mask
        # c_loss = infonce(goldenVec.cuda(), pg, mask.cuda(), neg_mask.cuda())

        # Use triplet loss instead of InfoNCE
        if torch.any(pos_mask):
            pos_distances, neg_distances = pairwise_distances(goldenVec.cuda(), pg, pos_mask.cuda(), neg_mask.cuda())
            c_loss = loss_method(pos_distances, neg_distances)
        else:
            c_loss = torch.tensor(0.0, device='cuda')
        
        s_loss = F.binary_cross_entropy_with_logits(x.squeeze(-1), labels.cuda(), pos_weight=torch.tensor(10).cuda())

        # Lấy MoE loss từ cả 2 model c và e (s)
        # Model C
        moe_loss_c = c_model.graph_encoder.get_contribution_loss() if hasattr(c_model.graph_encoder, 'get_contribution_loss') else torch.tensor(0.0, device=device)
        if not isinstance(moe_loss_c, torch.Tensor):
            moe_loss_c = torch.tensor(moe_loss_c, device=device)
        
        # Model E (S)
        # moe_loss_e = s_model.graph_encoder.get_load_balance_loss() if hasattr(s_model.graph_encoder, 'get_load_balance_loss') else torch.tensor(0.0, device=device)
        # if not isinstance(moe_loss_e, torch.Tensor):
        #     moe_loss_e = torch.tensor(moe_loss_e, device=device)
        
        moe_loss = moe_loss_c

        loss = c_loss * 1. + s_loss + moe_loss
        scores = torch.sigmoid(x.squeeze(-1))
    
    return loss.data, c_loss.data, s_loss.data, moe_loss.data, scores

def get_summary(scores, sents, max_word_num=160, sent_num=0, kappa=0.45):
    ranked_score_idxs = torch.argsort(scores, dim=0, descending=True)
    wordCnt = 0
    summSentIDList = []
    for i in ranked_score_idxs:
        if wordCnt >= max_word_num and sent_num == 0: break
        elif sent_num > 0 and len(summSentIDList) == sent_num: break
        s = sents[i]

        replicated = False
        for chosedID in summSentIDList:
            if getRouge2(sents[chosedID], s, 'p') >= kappa:
                replicated = True
                break
        if replicated: continue

        wordCnt += len(s.split(' '))
        summSentIDList.append(i)
    summSentIDList = sorted(summSentIDList)
    return ' '.join([s for i, s in enumerate(sents) if i in summSentIDList])