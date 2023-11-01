import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_
from .BertConfigConv import BertConfig

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * F.sigmoid(x)


def get_activation(activation_string):
    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "leakyrelu":
        return nn.LeakyReLU()
    elif act == "swish":
        return Swish()
    elif act == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Unsupported activation: %s" % act)

class BertEmbedding(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embedding = nn.Linear(config.seq_len,config.hidden_len)
        self.acti = get_activation(config.act_fn)
        # self.dropout = nn.Dropout(config.dropout)
        self.batchnorm = nn.LayerNorm(config.hidden_len)
        self.lin = nn.Linear(config.vocab_size,config.hidden_size)

    def forward(self,x):
        # x [bsz,seq_len,feature]
        x = x.permute(0,2,1)
        x = self.acti(x)
        x = self.embedding(x)
        x = self.batchnorm(x)
        x = x.permute(0,2,1)
        x = self.lin(x)
        return x
    
class BertSelfAtten(nn.Module):
    def __init__(self, config) -> None:
        super(BertSelfAtten,self).__init__()
        self.multi_heat_attn = nn.MultiheadAttention(embed_dim=config.intermediate_size,
                                                     num_heads=config.num_heads,
                                                     dropout=config.attn_dropout,
                                                     batch_first=True)
        #except input size [bsz,src_len,vocabsize]
        # self.layer_norm = nn.LayerNorm(config.intermediate_size)

    def forward(self,query,key,value):
        atten_out, _ = self.multi_heat_attn(query,key,value)
        # atten_out = self.layer_norm(atten_out)
        return atten_out
        
class BertInter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size,config.intermediate_size)
        self.act_fn = get_activation(config.act_fn)

    def forward(self,x):
        x = self.act_fn(x)
        x = self.dense(x)
        return x
    
class BertOuter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size,config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self,x):
        x = self.dense(x)
        x = self.dropout(x)
        x = self.layernorm(x)
        return x
    
class BertLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.inter = BertInter(config)
        self.attn = BertSelfAtten(config)
        self.output = BertOuter(config)

    def forward(self,x):
        x = self.inter(x)
        attn_out = self.attn(x,x,x)
        lay_out = self.output(attn_out)
        return lay_out

class BertEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.emb = BertEmbedding(config)
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.out = nn.Linear(config.hidden_size,1)
        self._reset_parameters(config.initializer_range)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)

        # return [bsz,seq_len,hidden_size]

    def forward(self,x):
        x = self.emb(x)
        for layer in self.bert_layers:
            x = x + layer(x)
            # x = layer(x,x,x)
        x = x.permute(0,2,1)
        x = torch.sum(x,dim=1)
        x = x.unsqueeze(1)
        #return size [bsz,1,seq_len]
        return x
    

class BertForPretrain(nn.Module):
    '''
    In this model 
    input: [bsz,seq_len,vocab_size]
    out: pretrain loss, [bsz,fea_dim,fea_size],[bsz,fea_dim,fea_size]
    
    '''
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = BertEncoder(config)
        self.eads_pred = nn.Linear(config.hidden_len,6)
        self.cls_pred = nn.Linear(config.hidden_len,7)
        self.loss_func1 = nn.L1Loss(reduction='none')
        self.loss_func2 = nn.BCELoss()
        self.batchnorm = nn.LayerNorm(config.hidden_len)
        # self.loss_func2 = nn.CrossEntropyLoss()
        weight = torch.tensor([[1.25,97.6,0.28,0.41,0.353,0.018]]) 
        self.weight = weight.unsqueeze(0).cuda()

    def forward(self,input,eads,sys_cls):
        encoder_out = self.encoder(input)
        encoder_out = self.batchnorm(encoder_out)
        eads_pred = self.eads_pred(encoder_out)
        cls_pred = self.cls_pred(encoder_out).squeeze(1)

        loss1 = self.loss_func1(eads,eads_pred)
        bias = self.weight.expand(loss1.shape[0],-1,-1)
        loss1 = loss1 * bias
        loss1 = loss1.sum() / loss1.numel()
        # atom_label = F.softmax(cls_pred[:,:2],dim=1)
        # metal_label = F.softmax(cls_pred[:,2:4],dim=1)
        # pos_label = F.softmax(cls_pred[:,4:],dim=1)
        sys_cls = sys_cls.squeeze(1).float()
        loss2 = self.loss_func2(F.sigmoid(cls_pred),sys_cls)
        # loss2 = self.loss_func2(F.sigmoid(atom_label),sys_cls[:,:2]) + self.loss_func2(F.sigmoid(metal_label),sys_cls[:,2:4]) + self.loss_func2(F.sigmoid(pos_label),sys_cls[:,4:])

        loss = loss1 + loss2 

        return loss, eads_pred, cls_pred

    
if __name__ == '__main__':
    json_file = 'config/config_conv.json'
    config = BertConfig.from_json_file(json_file)
    aaa = torch.randn(32,1000,1)
    net = BertForPretrain(config)
    # print(net)
    # print(net.parameters)
    bbb = torch.randn(32,1,6)
    ccc = torch.ones(32,1,7)

    xxx = net(aaa,bbb,ccc)
    print(xxx[0])
    print(xxx[1].shape)
    print(xxx[2].shape)
