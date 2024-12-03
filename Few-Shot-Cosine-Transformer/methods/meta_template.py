import backbone
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import tqdm
from abc import abstractmethod
import pdb
import wandb
global device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import warnings
warnings.filterwarnings("ignore")
class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, k_shot, n_query, change_way = True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.k_shot     = k_shot
        self.n_query    = n_query
        self.feature    = model_func()
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self, x, is_feature):
        # 디버깅용: 입력 데이터 크기 출력
        #print(f"Original x shape: {x.size()}")

        # GPU로 데이터 전송
        x = Variable(x.to(device))

        if is_feature:
            z_all = x
        else:
            # 배치 차원 추가 확인 및 제거
            if len(x.size()) == 5:
                x = x.view(-1, *x.size()[2:])  # [batch_size, 3, 224, 224]로 변환
                #print(f"Adjusted x shape after view: {x.size()}")

            # n_way * (k_shot + n_query)의 예상 크기 확인
            expected_size = self.n_way * (self.k_shot + self.n_query)
            actual_size = x.size(0)

            if actual_size != expected_size:
                raise ValueError(
                    f"Input size mismatch: expected {expected_size}, got {actual_size}"
                )

            # Feature 추출
            z_all = self.feature.forward(x)
            z_all = z_all.view(self.n_way, self.k_shot + self.n_query, *z_all.size()[1:])
            #print(f"Feature shape after forward pass: {z_all.size()}")

        # Support set과 Query set 분리
        z_support = z_all[:, :self.k_shot]  # [n_way, k_shot, ...]
        z_query = z_all[:, self.k_shot:]   # [n_way, n_query, ...]

        #print(f"z_support shape: {z_support.size()}, z_query shape: {z_query.size()}")
        return z_support, z_query


    def correct(self, x):       
        scores = self.set_forward(x)
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.to(device))

        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels
        top1_correct = (topk_ind[:,0] == y_query).sum().item()
        return float(top1_correct), len(y_query)

    def train_loop(self, epoch, num_epoch, train_loader, wandb_flag, optimizer):
        avg_loss = 0
        avg_acc = []
        with tqdm.tqdm(total = len(train_loader)) as train_pbar:
            for i, (x, _) in enumerate(train_loader):        
                if self.change_way:
                    self.n_way  = x.size(0)
                
                optimizer.zero_grad()
                acc, loss = self.set_forward_loss(x = x.to(device))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                avg_acc.append(acc)
                train_pbar.set_description('Epoch {:03d}/{:03d} | Acc {:.6f}  | Loss {:.6f}'.format(
                    epoch + 1, num_epoch, np.mean(avg_acc) * 100, avg_loss/float(i+1)))
                train_pbar.update(1)
        if wandb_flag:
            wandb.log({"Loss": avg_loss/float(i + 1),'Train Acc': np.mean(avg_acc) * 100},  step=epoch + 1)

    def val_loop(self, val_loader, epoch, wandb_flag, record = None):
        correct =0
        count = 0
        acc_all = []
        
        iter_num = len(val_loader)
        with tqdm.tqdm(total=len(val_loader)) as val_pbar:
            for i, (x,_) in enumerate(val_loader):
                if self.change_way:
                    self.n_way  = x.size(0)
                correct_this, count_this = self.correct(x)
                acc_all.append(correct_this / count_this * 100)
                val_pbar.set_description('Validation    | Acc {:.6f}'.format(np.mean(acc_all)))
                val_pbar.update(1)

        acc_all  = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        if wandb_flag:
            wandb.log({'Val Acc': acc_mean},  step = epoch + 1)
        print('Val Acc = %4.2f%% +- %4.2f%%' %(  acc_mean, 1.96* acc_std/np.sqrt(iter_num)))

        return acc_mean