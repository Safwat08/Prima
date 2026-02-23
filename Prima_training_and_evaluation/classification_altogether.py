import torch
import sys, os, csv
import heapq

sys.path.append(os.getcwd())
from tqdm import tqdm
from sklearn import metrics
from dataset import collate_visual_hash, MrDataset

import numpy as np
try:
    import wandb
except ImportError:
    wandb = None

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def collate_emb_hash(datas):
    tensors, labels, hashes = zip(*datas)
    return torch.stack(tensors), torch.LongTensor(labels), list(hashes)


class ClassificationTask:

    def __init__(self,
                 dataset,
                 visualmodel,
                 trainlabels,
                 vallabels,
                 testlabels,
                 patchify,
                 protobatchsize=12,
                 classnum=1,
                 retpool=True):

        self.allembeds = []
        self.classnum = classnum
        self.trainlabels = trainlabels
        self.vallabels = vallabels
        self.testlabels = testlabels
        self.alllabels = {}
        self.alllabels.update(trainlabels)
        self.alllabels.update(vallabels)
        self.alllabels.update(testlabels)

        # obtain visual embeddings from pre-trained clip visual model
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=protobatchsize,
                                             shuffle=False,
                                             num_workers=0,
                                             collate_fn=collate_visual_hash(
                                                 patchify,
                                                 device,
                                                 put_to_device=True))
        with torch.no_grad():
            for d in tqdm(loader):
                h = d['hash']
                with torch.amp.autocast(device_type='cuda',
                                        dtype=torch.float16):
                    embeds = visualmodel(xdict=d, retpool=retpool)
                for i, (tensor, hashname) in enumerate(zip(embeds, h)):
                    if hashname in self.alllabels:
                        self.allembeds.append(
                            (tensor.float(), self.alllabels[hashname], hashname))

        # split into train/val/test sets
        self.trainembeds, self.valembeds, self.testembeds = self.split(
            self.allembeds)

        print(len(self.valembeds))
        print(len(self.testembeds))

        # compute class weights for multiclass CE
        ttotals = len(self.trainembeds)
        counts = torch.zeros(self.classnum).long()
        for _, l, _ in self.trainembeds:
            counts[l] += 1
        safe_counts = counts.clamp(min=1)
        weights = ttotals / (self.classnum * safe_counts.float())
        print('train totals: ' + str(ttotals))
        print('train class counts: ' + str(counts))
        print('class weights: ' + str(weights))

        self.trainembedsbalanced = self.trainembeds
        self.criterionweighted = torch.nn.CrossEntropyLoss(weight=weights.to(
            device))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax(dim=1)
        self.visembedlen = len(self.allembeds[0][0])

    def trainandval(self, model, optimizer, batch_size=200):
        trainloader = torch.utils.data.DataLoader(self.trainembedsbalanced,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_emb_hash)
        totalloss = 0.0
        totals = 0

        # train for 1 epoch
        for t, l, h in tqdm(trainloader):
            t = t.to(device)
            t *= (torch.randn_like(t) * 0.01 + 1)
            l = l.to(device).long()
            optimizer.zero_grad()
            out = model(t)
            loss = self.criterionweighted(out, l)
            loss.backward()
            optimizer.step()
            totalloss += loss.item() * len(l)
            totals += len(l)

        # run validation set
        trainloss = totalloss / totals
        valloss, valacc, valauc, correctposlist, correctneglist, fullprob = self.evalsplit(
            model, self.valembeds)
        testloss, testacc, testauc, testpos, testneg, testprob = self.evalsplit(
            model, self.testembeds)

        return trainloss, valloss, valacc, valauc, correctposlist, correctneglist, fullprob, [
            None
        ] * self.classnum, testloss, testacc, testauc, testpos, testneg

    def evalsplit(self, model, embeds, batch_size=200):
        if embeds is None or len(embeds) == 0:
            return 0.0, [0.0] * self.classnum, [float('nan')] * self.classnum, [
                0
            ] * self.classnum, [0] * self.classnum, np.zeros(
                (0, self.classnum), dtype=np.float32)

        loader = torch.utils.data.DataLoader(embeds,
                                             batch_size=batch_size,
                                             collate_fn=collate_emb_hash)
        classnum = self.classnum
        totalloss = 0.0
        totals = 0
        fullprob = []
        fulllabels = []
        fullpreds = []
        with torch.no_grad():
            for t, l, h in loader:
                t = t.to(device)
                ll = l.to(device).long()
                out = model(t)
                prob = self.softmax(out)
                pred = prob.argmax(dim=1).cpu()
                loss = self.criterion(out, ll)
                totalloss += loss.item() * len(l)
                totals += len(l)
                for i in range(len(l)):
                    fullpreds.append(pred[i].item())
                    fullprob.append(prob[i].cpu().numpy())
                    fulllabels.append(l[i])

        # calculate metrics
        valloss = totalloss / totals if totals > 0 else 0.0
        fullprob = np.array(fullprob)
        fullpreds = np.array(fullpreds)
        fulllabels = torch.stack(fulllabels).detach().cpu().numpy()
        valacc = []
        valauc = []
        correctposlist = []
        correctneglist = []
        for j in range(classnum):
            if len(fulllabels) == 0:
                valacc.append(0.0)
                valauc.append(float('nan'))
                correctposlist.append(0)
                correctneglist.append(0)
                continue
            gts_np = (fulllabels == j).astype(np.int32)
            pred_bin_np = (fullpreds == j).astype(np.int32)
            values = fullprob[:, j]
            valacc.append(float((pred_bin_np == gts_np).mean()))
            if len(np.unique(gts_np)) < 2:
                valauc.append(float('nan'))
            else:
                valauc.append(metrics.roc_auc_score(gts_np, values))
            correctposlist.append(int(np.logical_and(pred_bin_np == 1,
                                                     gts_np == 1).sum()))
            correctneglist.append(int(np.logical_and(pred_bin_np == 0,
                                                     gts_np == 0).sum()))

        return valloss, valacc, valauc, correctposlist, correctneglist, fullprob

    def split(self, pairs):
        # split embeddings into train/val/test sets
        vals = []
        tests = []
        trains = []
        trainhashset = set(self.trainlabels.keys())
        valhashset = set(self.vallabels.keys())
        testhashset = set(self.testlabels.keys())
        print(len(pairs))
        for t, l, h in pairs:
            if h in trainhashset:
                trains.append((t, l, h))
            elif h in valhashset:
                vals.append((t, l, h))
            elif h in testhashset:
                tests.append((t, l, h))
        return trains, vals, tests


# create empty list
def emptylist(num):
    ret = []
    for i in range(num):
        ret.append(0)
    return ret


# load clip-trained vision model
# load clip-trained vision model
def loadvismodel(path, devices):
    m = torch.load(path, map_location='cpu', weights_only=False)
    base = m.module if hasattr(m, "module") else m

    # visual encoder can live under different names depending on checkpoint type
    if hasattr(base, "clipvisualmodel"):          # FullMRIModel
        visual = base.clipvisualmodel
    elif hasattr(base, "visual_model"):           # CLIP model
        visual = base.visual_model
    else:
        raise AttributeError("Could not find visual model on loaded checkpoint")

    model = torch.nn.DataParallel(visual, device_ids=devices).to(device)
    model.module.patdis = False

    # patchifier location differs too
    if hasattr(base, "patchifier"):
        patchifier = base.patchifier
    elif hasattr(base, "clipmodel") and hasattr(base.clipmodel, "patchifier"):
        patchifier = base.clipmodel.patchifier
    else:
        raise AttributeError("Could not find patchifier on loaded checkpoint")

    return model, patchifier.to("cpu")



# main training loop for a specific checkpoint
def train(protodataset,
          vismodelpath,
          trainlabels,
          vallabels,
          testlabels,
          epochs=45,
          classnum=1,
          devices=[0],
          savesite='.',
          cnames=[],
          lr=0.00001,
          momentum=0.0,
          weight_decay=0.0,
          top_k=5,
          wandb_run=None):
    vismodel, patchify = loadvismodel(vismodelpath, devices)
    cls = ClassificationTask(protodataset,
                             vismodel,
                             trainlabels,
                             vallabels,
                             testlabels,
                             patchify,
                             classnum=classnum)
    bestvauc = emptylist(classnum)
    bestvacc = emptylist(classnum)
    bestaucfpred = [[] for i in range(classnum)]
    bestaccfpred = [[] for i in range(classnum)]
    newmodel = torch.nn.Sequential(torch.nn.Linear(cls.visembedlen,
                                                   4000), torch.nn.ReLU(),
                                   torch.nn.Linear(4000, 1000),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(1000, classnum))
    newmodel = torch.nn.DataParallel(
        newmodel, device_ids=config['train']['devices']).to(device)
    optim = torch.optim.RMSprop(newmodel.parameters(),
                                lr=lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    topk_heap = []
    for e in range(epochs):
        tloss, vloss, vacc, vauc, cpos, cneg, fpred, threshs, tloss_test, testacc, testauc, testpos, testneg = cls.trainandval(
            newmodel, optim)
        finite_vauc = [x for x in vauc if not np.isnan(x)]
        mean_vauc = float(np.mean(finite_vauc)) if len(finite_vauc) > 0 else float(
            'nan')
        finite_vacc = [x for x in vacc if not np.isnan(x)]
        mean_vacc = float(np.mean(finite_vacc)) if len(finite_vacc) > 0 else float(
            'nan')
        finite_testauc = [x for x in testauc if not np.isnan(x)]
        mean_testauc = float(np.mean(
            finite_testauc)) if len(finite_testauc) > 0 else float('nan')
        finite_testacc = [x for x in testacc if not np.isnan(x)]
        mean_testacc = float(np.mean(
            finite_testacc)) if len(finite_testacc) > 0 else float('nan')
        print('epoch ' + str(e) + ' train loss: ' + str(tloss))
        print('epoch ' + str(e) + ' val loss: ' + str(vloss))
        print('epoch ' + str(e) + ' mean val acc: ' + str(mean_vacc))
        print('epoch ' + str(e) + ' mean val auc: ' + str(mean_vauc))
        print('epoch ' + str(e) + ' val acc: ' + str(vacc))
        print('epoch ' + str(e) + ' val auc: ' + str(vauc))
        print('epoch ' + str(e) + ' val correct positive: ' + str(cpos))
        print('epoch ' + str(e) + ' val correct negative: ' + str(cneg))
        print('epoch ' + str(e) + ' test loss: ' + str(tloss_test))
        print('epoch ' + str(e) + ' mean test acc: ' + str(mean_testacc))
        print('epoch ' + str(e) + ' mean test auc: ' + str(mean_testauc))
        print('epoch ' + str(e) + ' test acc: ' + str(testacc))
        print('epoch ' + str(e) + ' test auc: ' + str(testauc))
        print('epoch ' + str(e) + ' test correct positive: ' + str(testpos))
        print('epoch ' + str(e) + ' test correct negative: ' + str(testneg))
        if wandb_run is not None:
            wandb_run.log({
                'epoch': e,
                'train/loss': tloss,
                'val/loss': vloss,
                'val/mean_acc': mean_vacc,
                'val/mean_auc': mean_vauc,
                'test/loss': tloss_test,
                'test/mean_acc': mean_testacc,
                'test/mean_auc': mean_testauc,
            })

        # save top-k checkpoints based on mean validation AUC
        if not np.isnan(mean_vauc):
            ckpt_name = f'top_valauc_epoch{e:03d}_{mean_vauc:.6f}.pt'
            ckpt_path = os.path.join(savesite, ckpt_name)
            torch.save(newmodel.module, ckpt_path)
            heapq.heappush(topk_heap, (mean_vauc, ckpt_path))
            if len(topk_heap) > top_k:
                _, remove_path = heapq.heappop(topk_heap)
                if os.path.exists(remove_path):
                    os.remove(remove_path)

        # save best checkpoints for each task
        for i in range(classnum):
            if np.isnan(vauc[i]):
                continue
            if vauc[i] > bestvauc[i]:
                bestvauc[i] = vauc[i]
                bestvacc[i] = vacc[i]
                bestaucfpred[i] = fpred[:, i]
                torch.save(newmodel.module,
                           savesite + '/bestauc_' + cnames[i] + '.pt')
    return bestvauc, bestvacc, bestaucfpred, bestaccfpred, cls


import yaml, argparse


def load_label_csv(path, classnum):
    mapping = {}
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            study_id = row[0].strip()
            label_raw = row[1].strip()
            if study_id == '':
                continue
            try:
                label = int(label_raw)
            except ValueError:
                # allow header like study_id,class
                continue
            if label < 0 or label >= classnum:
                raise ValueError(
                    f'label out of range in {path}: {study_id},{label}')
            mapping[study_id] = label
    return mapping


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        type=argparse.FileType('r'),
                        required=True,
                        help='config file for training')
    args = parser.parse_args()
    return args.config


if __name__ == '__main__':
    cf_fd = parse_args()
    config = yaml.load(cf_fd, Loader=yaml.FullLoader)
    thedataset = MrDataset
    protodataset = thedataset(data_json=config['data']['datajson'],
                              data_root_dir=config['data']['datarootdir'],
                              tokenizer='biomed',
                              text_max_len=200,
                              is_train=True,
                              vqvae_name=config['data']['vqvaename'],
                              visual_hash_only=True,
                              percentage=config['data']['percentage'],
                              no_visual_aug=True,
                              no_split=config['data']['nosplit']
                              if 'nosplit' in config['data'] else False)
    if 'vismodelpath' not in config['train']:
        raise ValueError('Missing required config key: train.vismodelpath')
    vismodelpath = config['train']['vismodelpath']
    classnum = config['cset']['classnum']
    trainlabels = load_label_csv(config['cset']['train_csv'], classnum)
    vallabels = load_label_csv(config['cset']['val_csv'], classnum)
    testlabels = load_label_csv(config['cset']['test_csv'], classnum)
    mysavesite = config['cset']['savesite']
    os.system('mkdir ' + mysavesite + '/' + config['cset']['markdate'])
    run = None
    if config['train'].get('use_wandb', False):
        if wandb is None:
            raise ImportError(
                'wandb is not installed. Install it or set train.use_wandb to false.'
            )
        run = wandb.init(project=config['train'].get('wandb_project',
                                                     'prima-classification'),
                         entity=config['train'].get('wandb_entity', None),
                         name=config['train'].get('wandb_run_name', None),
                         config=config)

    bestvauc, bestvacc, bestaucfpred, bestaccfpred, cls = train(
        protodataset,
        vismodelpath,
        trainlabels,
        vallabels,
        testlabels,
        epochs=config['train']['epochs'],
        classnum=classnum,
        savesite=mysavesite + '/' + config['cset']['markdate'],
        cnames=config['cset']['names'],
        devices=config['train']['devices'],
        lr=config['train'].get('lr', 0.00001),
        momentum=config['train'].get('momentum', 0.0),
        weight_decay=config['train'].get('weight_decay', 0.0),
        top_k=config['train'].get('save_top_k', 5),
        wandb_run=run)
    if run is not None:
        run.finish()
    print(vismodelpath + ' best val auc: ' + str(bestvauc))
    print('best auc and acc:')
    for i, (auc, acc) in enumerate(zip(bestvauc, bestvacc)):
        print(config['cset']['names'][i] + ' auc and acc:')
        print(str(auc) + ' ' + str(acc))
