import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from rightTransfomer import rightTransformerBlock
from Embedding import Embedding
from Multihead_Attention import MultiHeadedAttention
from postionEmbedding import PositionalEmbedding
from LayerNorm import LayerNorm
from SubLayerConnection import *
from DenseLayer import DenseLayer
import numpy as np

class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.modelName=args.modelName
        self.layers=args.layers
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 1, self.feed_forward_hidden, 0.1, self.modelName) for _ in range(self.layers)])
        self.token_embedding = nn.Embedding(100, self.embedding_size-1)
        self.token_embedding1 = nn.Embedding(100, self.embedding_size)
        print(args)
        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.transformerBlocksTree = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.pos = PositionalEmbedding(self.embedding_size)
        self.loss = nn.CrossEntropyLoss()
        self.norm = LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size // 2, int(self.embedding_size / 4), batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.resLinear2 = nn.Linear(self.embedding_size, 1)
    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus,lineRank, tokenTypes, tokenRes,tokenNodes):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(tokenTypes, 1)
        # print("resmask = ",resmask)
        inputad = inputad.float()
        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        x = nodeem
        #lineem = self.token_embedding1(linenode)
        # print(tokenNodes.shape)
        lineem = self.token_embedding1(tokenNodes)
        # print(lineem.shape)
        x = torch.cat([x, lineem], dim=1)
        # print(x.shape)

        lineem = self.token_embedding(linenode)
        # print(lineem.shape)
        tempvis = lineRank.unsqueeze(-1).float()
        # print(lineRank.shape, tempvis.shape)
        lineem = torch.cat([lineem, tempvis ], dim=-1)
        # print(lineem.shape)
        x = torch.cat([x, lineem], dim=1)
        # print("x.shape",x.shape,inputad.shape)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, inputad)
        x = x[:,input_node.size(1):input_node.size(1)+tokenNodes.size(1)]
        # print(x.shape)
        resSoftmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(resmask==0, -1e9), dim=-1)
        # print("resSoftMax = ",resSoftmax)
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * tokenRes
        loss = loss.sum(dim=-1)
        return loss, resSoftmax, x

# git rm -r zhangzhanwen/AstCfgPdg_JapaserSoot-master/
# git rm -r zhangzhanwen/Automated-CFG-Generation-master/
# git rm -r zhangzhanwen/ConDecfects-main/
# git rm -r zhangzhanwen/control-flow-graph-master/
# git rm -r zhangzhanwen/FFL-production/
# git rm -r zhangzhanwen/FLITSR/
# git rm -r zhangzhanwen/for-wala/
# git rm -r zhangzhanwen/Grace/
# git rm -r zhangzhanwen/java_flow_analyser-main/
# git rm -r zhangzhanwen/scripts/
# git rm -r zhangzhanwen/GccovFL/GraceVsusResult/
# git rm -r zhangzhanwen/GccovFL/ITSP-data/
# git rm -r zhangzhanwen/GccovFL/pre_result/
# git rm -r zhangzhanwen/GccovFL/pre_VsusResult/
# git rm -r zhangzhanwen/GccovFL/result/
# git rm -r zhangzhanwen/GccovFL/resultVsusResultDown/
# git rm -r zhangzhanwen/GccovFL/test/
# git rm -r zhangzhanwen/GccovFL/transToGraceInput/Codeflaws/codeflaws
# git rm -r zhangzhanwen/GccovFL/transToGraceInput/codeflawsDatasymbol
# git rm -r zhangzhanwen/GccovFL/transToGraceInput/ITSP-data
# git rm -r zhangzhanwen/GccovFL/transToGraceInput/result
# git rm -r zhangzhanwen/GccovFL/transToGraceInput/result2023年11月7日
#
#
#
# git rm -r  zhangzhanwen/GccovFL/transToGraceInput/Codeflaws/codeflaws_1/99-A-bug-3759651-3759654/Tag_c/1.txt
# git rm zhangzhanwen/GccovFL/transToGraceInput/Codeflaws/codeflaws_1/99-A-bug-6985349-6985355/Tag_c/1.txt
# git rm zhangzhanwen/GccovFL/transToGraceInput/Codeflaws/codeflaws_1/99-B-bug-5811742-5811752/Tag_c/1.txt
# git rm zhangzhanwen/GccovFL/transToGraceInput/Codeflaws/temp.py
# git rm zhangzhanwen/GccovFL/transToGraceInput/test/17-A-13897450.c