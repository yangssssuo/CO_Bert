import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_
from .BertConfig import BertConfig

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
    else:
        raise ValueError("Unsupported activation: %s" % act)

class InputEmb(nn.Module):
    def __init__(self, vocab_size,hidden_size,initializer_range=0.02) -> None:
        super(InputEmb,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Linear(vocab_size,hidden_size)
        self._reset_parameters(initializer_range)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)

    def forward(self,x):
        return self.embedding(x)
    

class BertEmbedding(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.embedding = InputEmb(config.vocab_size,
                                  config.hidden_size,
                                  config.initializer_range)
        self.batchnorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        x = self.embedding(x)
        x = self.batchnorm(x)
        x = self.dropout(x)
        return x
    
class BertSelfAtten(nn.Module):
    def __init__(self, config) -> None:
        super(BertSelfAtten,self).__init__()
        self.multi_heat_attn = nn.MultiheadAttention(embed_dim=config.hidden_size,
                                                     num_heads=config.num_heads,
                                                     dropout=config.attn_dropout,
                                                     batch_first=True)
        #except input size [bsz,src_len,vocabsize]
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.drop_out = nn.Dropout(config.attn_dropout)

    def forward(self,query,key,value):
        atten_out, _ = self.multi_heat_attn(query,key,value)
        atten_out = self.drop_out(atten_out)
        atten_out = self.layer_norm(atten_out)
        return atten_out
    
class BertInter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size,config.intermediate_size)
        self.act_fn = get_activation(config.act_fn)

    def forward(self,x):
        x = self.linear(x)
        x = self.act_fn(x)
        return x
    
class BertOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.linear = nn.Linear(config.intermediate_size,config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size,eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self,x):
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x
    
class BertLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn = BertSelfAtten(config)
        self.inter = BertInter(config)
        self.output = BertOutput(config)

    def forward(self,q,k,v):
        attn_out = self.attn(q,k,v)
        inter_out = self.inter(attn_out)
        lay_out = self.output(inter_out)
        return lay_out
    
class BertEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.emb = BertEmbedding(config)
        self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.out = nn.Linear(config.hidden_size,1)
        # return [bsz,seq_len,hidden_size]

    def forward(self,x):
        x = self.emb(x)
        for layer in self.bert_layers:
            x = x + layer(x,x,x)
            # x = layer(x,x,x)
        x = self.out(x)
        x = x.permute(0,2,1)
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
        self.eads_pred = nn.Linear(config.seq_len,6)
        self.cls_pred = nn.Linear(config.seq_len,7)
        self.loss_func1 = nn.L1Loss(reduction='mean')
        self.loss_func2 = nn.CrossEntropyLoss()

    def forward(self,input,eads,sys_cls):
        encoder_out = self.encoder(input)
        eads_pred = self.eads_pred(encoder_out)
        cls_pred = self.cls_pred(F.sigmoid(encoder_out))
        cls_vec = torch.concat([F.softmax(cls_pred[:,:,:2],dim=2), 
                                F.softmax(cls_pred[:,:,2:4],dim=2), 
                                F.softmax(cls_pred[:,:,4:],dim=2)],
                                dim=2)
        
        loss1 = self.loss_func1(eads,eads_pred)
        loss2 = self.loss_func2(cls_vec,sys_cls.float())

        loss = loss1 + loss2

        return loss, eads_pred, cls_pred
    

if __name__ == '__main__':
    json_file = 'config.json'
    config = BertConfig.from_json_file(json_file)
    aaa = torch.randn(32,400,1)
    net = BertForPretrain(config)
    bbb = torch.randn(32,1,6)
    ccc = torch.ones(32,1,7)

    xxx = net(aaa,bbb,ccc)
    print(xxx[0])
    print(xxx[1].shape)
    print(xxx[2].shape)





