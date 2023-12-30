import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from .meta_model import MetaModel

"""
way_num shot_num query_num eval_query eval_way eval_shot
"""

class INSTA(MetaModel):
    def __init__(self, **kwargs):
        super(INSTA,self).__init__(**kwargs)
        self.kwargs = kwargs
        backbone_ = 'resnet12'

        if backbone_ == 'resnet12':
            hdim = 640
            from core.model.backbone.resnet_12 import ResNet
            self.encoder = ResNet()
        elif backbone_ == 'resnet18':
            hdim = 512
            from core.model.backbone.resnet_18 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('Backbone not supported!')
        

    def split_instances(self, data):
        kwargs = self.kwargs
        if self.training:
            return (torch.Tensor(np.arange(kwargs.way_num*kwargs.shot_num)).long().view(1,kwargs.shot_num,kwargs.way_num),
                    torch.Tensor(np.arange(kwargs.way_num*kwargs.shot_num,kwargs.way_num*(kwargs.shot_num+kwargs.query_num))).long().view(1,kwargs.eval_query,kwargs.way_num))
        else:
            return (torch.Tensor(np.arange(kwargs.eval_way*kwargs.eval_shot)).long().view(1,kwargs.eval_shot,kwargs.eval_way),
                    torch.Tensor(np.arange(kwargs.eval_way*kwargs.eval_shot,kwargs.eval_way*(kwargs.eval_shot+kwargs.eval_query))).long().view(1,kwargs.eval_query,kwargs.eval_way))
        

    def forward(self, x,get_feature=False):
        if get_feature:
            return self.encoder(x)
        else:
            x = x.squeeze(0)
            instance_embs = self.encoder(x)

            support_idx,query_idx = self.split_instances(x)
            if self.training:
                logits,logits_reg = self._forwad(instance_embs,support_idx,query_idx)
                return logits,logits_reg
            else:
                logits = self._forwad(instance_embs,support_idx,query_idx)
                return logits
        
    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')
    

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 1, 5, 0, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 4, 0, 5, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, sigma, k, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.sigma = sigma
        self.k = k
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 5) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 5) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 5x5 frequency space
        # eg, (2,2) in 10x10 is identical to (1,1) in5x5

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel*self.sigma), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel*self.sigma), channel*self.k**2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, self.k, self.k)
        # return x * y.expand_as(x)
        return y

class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter

class INSTA_layer(nn.Module):
    def __init__(self, c, spatial_size, sigma, k, kwargs):
        super().__init__()
        self.channel = c
        self.h1 = sigma
        self.h2 = k **2
        self.k = k
        self.conv = nn.Conv2d(self.channel, self.h2, 1)
        self.fn_spatial = nn.BatchNorm1d(spatial_size**2)
        self.fn_channel = nn.BatchNorm1d(self.channel)
        self.unfold = nn.Unfold(kernel_size=self.k,padding=(self.k+1)/2-1)
        self.spatial_size = spatial_size
        c2wh = dict([(512,11),(640,self.spatial_size)])
        self.channel_att = MultiSpectralAttentionLayer(c, c2wh[c], c2wh[c], sigma=self.h1, k=self.k, freq_sel_method='low16')
        self.kwargs = kwargs
        self.CLM_upper = nn.Sequential(
            nn.Conv2d(c, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU()
        )

        self.CLM_lower = nn.Sequential(
            nn.Conv2d(c*2, c*2, 1),
            nn.BatchNorm2d(c*2),
            nn.ReLU(),
            nn.Conv2d(c*2, c, 1),
            nn.BatchNorm2d(c),
            nn.Sigmoid()
        )

    def CLM(self, featuremap):   #NxK,C,H,W
        featuremap = featuremap
        adap = self.CLM_upper(featuremap)
        intermediate = adap.sum(dim=0)
        adap_1 = self.CLM_lower(intermediate.unsqueeze(0))
        return adap_1
    
    def spatial_kernel_network(self, feature_map, conv):
        spatial_kernel = conv(feature_map)
        spatial_kernel = spatial_kernel.flatten(-2).transpose(-1,-2)
        size = spatial_kernel.size()
        spatial_kernel = spatial_kernel.view(size[0],-1,self.k,self.k)
        spatial_kernel = self.fn_spatial(spatial_kernel)

        spatial_kernel = spatial_kernel.flatten(-2)
        return spatial_kernel
    
    def channel_kernel_network(self, feature_map):
        channel_kernel = self.channel_att(feature_map)
        channel_kernel = self.fn_channel(channel_kernel)
        channel_kernel = channel_kernel.flatten(-2)
        channel_kernel = channel_kernel.squeeze().view(channel_kernel.shape[0],self.channel,-1)
        return channel_kernel
    
    def unfold(self,x,padding,k):
        x_padded = torch.cuda.FloatTensor(x.shape[0],x.shape[1],x.shape[2]+2*padding,x.shape[3]+2*padding).fill_(0)
        x_padded[:,:,padding:-padding,padding:-padding] = x
        x_unfolded = torch.cuda.FloatTensor(*x.shape,k,k).fill_(0)
        for i in range(int((self.k+1)/2-1), x.shape[2] + int((self.k+1)/2-1)):              ## if the spatial size of the input is 5,5, the sampled index starts from 1 ends with 7,
            for j in range(int((self.k+1)/2-1), x.shape[3] + int((self.k+1)/2-1)):
                x_unfolded[:, :, i - int(((self.k+1)/2-1)), j - int(((self.k+1)/2-1)), :, :] = x_padded[:, :, i-int(((self.k+1)/2-1)):i + int((self.k+1)/2), j - int(((self.k+1)/2-1)):j + int(((self.k+1)/2))]
        return x_unfolded
    
    def forward(self, x):
        spatial_kernel = self.spatial_kernel_network(x,self.conv).unsqueeze(-3)

        channel_kernel = self.channel_kernel_network(x).unsqueeze(-2)
        kernel = spatial_kernel * channel_kernel

        kernel_shape = kernel.size()
        feature_shape = x.size()
        instance_kernel = kernel.view(kernel_shape[0],kernel_shape[1],feature_shape[-2],feature_shape[-1],self.k,self.k)

        task_s = self.CLM(x)
        spatial_kernel_task = self.spatial_kernel_network(task_s,self.conv).unsqueeze(-3)
        channel_kernel_task = self.channel_kernel_network(task_s).unsqueeze(-2)
        task_kernel = spatial_kernel_task * channel_kernel_task
        task_kernel_shape = task_kernel.size()
        task_kernel = task_kernel.view(task_kernel_shape[0],task_kernel_shape[1],feature_shape[-2],feature_shape[-1],self.k,self.k)
        kernel = task_kernel*instance_kernel
        unfold_feature = self.unfold(x,int((self.k+1)/2-1),self.k)
        adapt_feature = (unfold_feature * kernel).mean(dim = (-1,-2)).squeeze(-1).squeeze(-1)
        return adapt_feature+x, task_kernel

class INSTA_ProtoNet(INSTA):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.INSTA = INSTA_layer(640,5,0.2,3,args = kwargs)

    def inner_loop(self, proto, support):
        SFC = proto.clone().detach()
        SFC = nn.Parameter(SFC.detach(),requires_grad=True)
        optimizer = torch.optim.SGD([SFC], lr=0.6,momentum=0.9,dampening=0.9,weight_decay=0)
        label_shot = torch.arange(self.kwargs.way_num).repeat(self.kwargs.shot_num)
        label_shot = label_shot.type(torch.cuda.LongTensor)
        with torch.enable_grad():
            for k in range(0,50):
                rand_id = torch.randperm(self.kwargs.way_num*self.kwargs.shot_num).cuda()
                for j in range(0,self.kwargs.way_num*self.kwargs.shot_num,4):
                    selected_id = rand_id[j:min(j+4,self.kwargs.way_num*self.kwargs.shot_num)]
                    batch_shot = support[selected_id,:]
                    batch_label = label_shot[selected_id]
                    optimizer.zero_grad()
                    logits = self.classifier(batch_shot.detach(),SFC)
                    if logits.dim()==1: logits = logits.unsqueeze(0)
                    loss = F.cross_entropy(logits,batch_label)
                    loss.backward()
                    optimizer.step()
        return SFC
    
    def classifier(self, query, proto):
        logits = - torch.sum((proto.unsqueeze(0) - query.unsqueeze(1))**2,2)/self.kwargs.temperature
        return logits.squeeze()
    
    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size()[-3:]
        channel_dim = emb_dim[0]

        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + emb_dim))
        query   = instance_embs[query_idx.flatten()].view(*(query_idx.shape + emb_dim))
        num_samples = support.shape[1]
        num_proto = support.shape[2]
        support = support.squeeze()

        adapted_s, task_kernel = self.INSTA(support.view(-1,*emb_dim))
        query = query.view(-1,*emb_dim)
        adapted_proto = adapted_s.view(num_samples, -1,adapted_s.shape[1:]).mean(0)
        adapted_proto = nn.AdaptiveAvgPool2d(1)(adapted_proto).squeeze(-1).squeeze(-1)

        query_ = nn.AdaptiveAvgPool2d(1)((self.INSTA.unfold(query, int((task_kernel.shape[-1]+1)/2-1),task_kernel.shape[-1]) * task_kernel)).squeeze()
        query = query_ + query
        adapted_q = nn.AdaptiveAvgPool2d(1)(query).squeeze(-1).squeeze(-1)
        if self.kwargs.testing:
            adapted_proto = self.inner_loop(adapted_proto, nn.AdaptiveAvgPool2d(1)(support).squeeze().view(num_proto*num_samples,channel_dim))
        logits = self.classifier(adapted_q, adapted_proto)

        if self.training:
            reg_logits = None
            return logits,reg_logits
        else:
            return logits