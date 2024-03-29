import sys
import torch
import torch.utils.data as data
import random
import pickle
import os
from nltk import word_tokenize
from vocab import VocabEntry
import numpy as np
import re
from tqdm import tqdm
from scipy import sparse
import math
import json

dmap = {
    'grace2811': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
                  15: 15, 16: 16, 17: 17, 18: 18, 19: 19},

    'Math': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16,
             15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28, 27: 29,
             28: 30, 29: 31, 30: 32, 31: 33, 32: 34, 33: 35, 34: 36, 35: 37, 36: 38, 37: 39, 38: 40, 39: 41, 40: 42,
             41: 43, 42: 44, 43: 45, 44: 46, 45: 47, 46: 48, 47: 49, 48: 50, 49: 51, 50: 52, 51: 53, 52: 54, 53: 55,
             54: 56, 55: 57, 56: 58, 57: 59, 58: 60, 59: 61, 60: 62, 61: 63, 62: 64, 63: 65, 64: 66, 65: 67, 66: 68,
             67: 69, 68: 70, 69: 71, 70: 72, 71: 73, 72: 74, 73: 75, 74: 76, 75: 77, 76: 78, 77: 79, 78: 80, 79: 81,
             80: 82, 81: 83, 82: 84, 83: 85, 84: 86, 85: 87, 86: 88, 87: 89, 88: 90, 89: 91, 90: 92, 91: 93, 92: 94,
             93: 95, 94: 96, 95: 97, 96: 98, 97: 99, 98: 100, 99: 101, 100: 102, 101: 103, 102: 105, 103: 106},
    'Lang': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
             15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 26, 24: 27, 25: 28, 26: 29, 27: 30,
             28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38, 36: 39, 37: 40, 38: 41, 39: 42, 40: 43,
             41: 44, 42: 45, 43: 46, 44: 47, 45: 48, 46: 49, 47: 50, 48: 51, 49: 52, 50: 53, 51: 54, 52: 55, 53: 57,
             54: 58, 55: 59, 56: 60, 57: 61, 58: 62, 59: 63, 60: 64, 61: 65},
    'Chart': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 12, 12: 13, 13: 14, 14: 15,
              15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 24, 23: 25, 24: 26},
    'Time': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
             15: 17, 16: 18, 17: 19, 18: 20, 19: 22, 20: 23, 21: 24, 22: 25, 23: 26, 24: 27},
    'Mockito': {0: 1, 1: 2, 2: 3, 3: 4, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 12, 11: 13, 12: 14, 13: 15, 14: 16,
                15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 29, 27: 30,
                28: 31, 29: 32, 30: 33, 31: 34, 32: 35, 33: 36, 34: 37, 35: 38}
}


class SumDataset(data.Dataset):
    def __init__(self, config, dataName="train", proj="Math", testid=0, lst=[]):
        # print(config)
        self.train_path = proj + ".pkl"
        self.val_path = "ndev.txt"  # "validD.txt"
        self.test_path = "ntest.txt"
        self.proj = proj
        self.SentenceLen = config.SentenceLen
        self.Nl_Voc = {"pad": 0, "Unknown": 1}
        self.Code_Voc = {"pad": 0, "Unknown": 1}
        self.Char_Voc = {"pad": 0, "Unknown": 1}
        self.Nl_Voc['Method'] = len(self.Nl_Voc)  # 方法
        self.Nl_Voc['Test'] = len(self.Nl_Voc)  # 错误用例节点
        self.Nl_Voc['Line'] = len(self.Nl_Voc)  # 语句级节点
        self.Nl_Voc['RTest'] = len(self.Nl_Voc)  # 正确测试用例
        self.Nl_Voc['Token'] = len(self.Nl_Voc)  # token级节点
        self.Nl_Len = config['NlLen']  # 方法、测试用例三种节点的个数
        self.Code_Len = config.CodeLen  # 语句节点的个数
        self.Token_Len = config['TokenLen']
        self.Char_Len = config.WoLen
        self.batch_size = config.batch_size
        self.PAD_token = 0
        self.data = None
        self.dataName = dataName
        self.Codes = []
        self.ids = []
        self.Nls = []
        if os.path.exists("nl_voc.pkl"):
            self.init_dic()
            # self.Load_Voc()
        else:
            self.init_dic()
        # print(self.Nl_Voc)
        if not os.path.exists(self.proj + 'data.pkl'):
            data = self.preProcessData(open(self.train_path, "rb"))
        else:
            data = pickle.load(open(self.proj + 'data.pkl', 'rb'))
        self.data = []
        if dataName == "train":  # 训练集
            for i in range(len(data)):
                tmp = []
                for j in range(len(data[i])):
                    if j in lst:
                        continue
                    tmp.append(data[i][j])
                self.data.append(tmp)
        elif dataName == 'test':
            testnum = 0
            ids = []
            while len(ids) < testnum:
                rid = random.randint(0, len(data[0]) - 1)
                if rid == testid or rid in ids or rid == 51:  # if rid >= testid * testnum and rid < testid * testnum + testnum or rid in ids:
                    continue
                ids.append(rid)
            self.ids = ids
            for i in range(len(data)):
                tmp = []
                for x in self.ids:
                    tmp.append(data[i][x])
                self.data.append(tmp)
        else:
            testnum = int(len(data[0]) / 10) + 1  # 1 分成十份。
            print(testnum)
            ids = []
            for i in range(len(data)):
                tmp = []
                for x in range(testnum * testid, testnum * testid + testnum):
                    if x < len(data[i]):
                        if i == 0:
                            ids.append(x)
                        tmp.append(data[i][x])
                self.data.append(tmp)
            self.ids = ids

    def Load_Voc(self):
        if os.path.exists("nl_voc.pkl"):
            self.Nl_Voc = pickle.load(open("nl_voc.pkl", "rb"))
        if os.path.exists("code_voc.pkl"):
            self.Code_Voc = pickle.load(open("code_voc.pkl", "rb"))
        if os.path.exists("char_voc.pkl"):
            self.Char_Voc = pickle.load(open("char_voc.pkl", "rb"))

    def splitCamel(self, token):
        ans = []
        tmp = ""
        for i, x in enumerate(token):
            if i != 0 and x.isupper() and token[i - 1].islower() or x in '$.' or token[i - 1] in '.$':
                ans.append(tmp)
                tmp = x.lower()
            else:
                tmp += x.lower()
        ans.append(tmp)
        return ans

    def init_dic(self):  # 读取数据文件
        # print("initVoc")
        # print(self.proj)
        f = open(self.proj + '.pkl', 'rb')
        data = pickle.load(f)
        # print(data)
        maxNlLen = 0
        maxCodeLen = 0
        maxCharLen = 0
        Nls = []
        Codes = []
        for x in data:
            for s in x['methods']:
                s = s[:s.index('(')]
                # print('s = ',s)
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))
                # print(Codes[-1])
            for s in x['ftest']:
                if len(s.split(":")) > 1:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:] + [s.split(":")[1]])
                else:
                    tokens = ".".join(s.split(":")[0].split('.')[-2:])
                Codes.append(self.splitCamel(tokens))

        code_voc = VocabEntry.from_corpus(Codes, size=50000, freq_cutoff=0)
        self.Code_Voc = code_voc.word2id
        open("code_voc.pkl", "wb").write(pickle.dumps(self.Code_Voc))

    def Get_Em(self, WordList, voc):
        ans = []
        for x in WordList:
            if x not in voc:
                ans.append(1)
            else:
                ans.append(voc[x])
        return ans

    def Get_Char_Em(self, WordList):
        ans = []
        for x in WordList:
            tmp = []
            for c in x:
                c_id = self.Char_Voc[c] if c in self.Char_Voc else 1
                tmp.append(c_id)
            ans.append(tmp)
        return ans

    def pad_seq(self, seq, maxlen):
        act_len = len(seq)  # 用于
        if len(seq) < maxlen:
            seq = seq + [self.PAD_token] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_str_seq(self, seq, maxlen):
        act_len = len(seq)
        if len(seq) < maxlen:
            seq = seq + ["<pad>"] * maxlen
            seq = seq[:maxlen]
        else:
            seq = seq[:maxlen]
            act_len = maxlen
        return seq

    def pad_list(self, seq, maxlen1, maxlen2):
        if len(seq) < maxlen1:
            seq = seq + [[self.PAD_token] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def pad_multilist(self, seq, maxlen1, maxlen2, maxlen3):
        if len(seq) < maxlen1:
            seq = seq + [[[self.PAD_token] * maxlen3] * maxlen2] * maxlen1
            seq = seq[:maxlen1]
        else:
            seq = seq[:maxlen1]
        return seq

    def tokenize_for_bleu_eval(self, code):
        code = re.sub(r'([^A-Za-z0-9])', r' \1 ', code)
        # code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
        code = re.sub(r'\s+', ' ', code)
        code = code.replace('"', '`')
        code = code.replace('\'', '`')
        tokens = [t for t in code.split(' ') if t]
        return tokens

    def getoverlap(self, a, b):
        ans = []
        for x in a:
            maxl = 0
            for y in b:
                tmp = 0
                for xm in x:
                    if xm in y:
                        tmp += 1
                maxl = max(maxl, tmp)
            ans.append(int(100 * maxl / len(x)) + 1)
        return ans

    def getRes(self, codetoken, nltoken):
        ans = []
        for x in nltoken:
            if x == "<pad>":
                continue
            if x in codetoken and codetoken.index(x) < self.Code_Len and x != "(" and x != ")":
                ans.append(len(self.Nl_Voc) + codetoken.index(x))
            else:
                if x in self.Nl_Voc:
                    ans.append(self.Nl_Voc[x])
                else:
                    ans.append(1)
        for x in ans:
            if x >= len(self.Nl_Voc) + self.Code_Len:
                # print(codetoken, nltoken)
                exit(0)
        return ans



    def preProcessData(self, dataFile):  # 好恶心的代码
        path_stacktrace = os.path.join('../FLocalization/stacktrace', self.proj)
        lines = pickle.load(dataFile)  # dataFile.readlines()
        Nodes = []  # 存储节点
        Types = []  # 存储语句和token节点类型
        LineRank = []  # 存储语句的rank
        LineNodes = []  #
        LineTypes = []
        LineMus = []
        Res = []
        tokenRes = []
        inputText = []
        inputNlad = []
        overlap = []
        VsusFLRanks = []
        tokenNodes = []
        tokenTypes = []
        maxl = 0
        maxl2 = 0
        maxl3 = 0
        error = 0
        error1 = 0
        error2 = 0
        correct = 0
        for k in range(len(lines)):
            x = lines[k]
            if os.path.exists(path_stacktrace + '/%d.json' % k):  # dmap[self.proj][k]):
                stack_info = json.load(open(path_stacktrace + '/%d.json' % k))  # dmap[self.proj][k]))
                if x['ftest'].keys() != stack_info.keys():
                    with open("problem_stack", 'a') as f:
                        f.write("{} {} no!\n".format(k, k))  # dmap[self.proj][k]))
                        f.write(str(x['ftest'].keys()) + '\n')
                        f.write(str(stack_info.keys()) + '\n')
                    for error_trace in x['ftest'].keys():
                        if error_trace not in stack_info.keys():
                            error += 1
                        else:
                            correct += 1
                        # assert error_trace in stack_info.keys()
                    # error += 1
                # else:
                # correct += 1

            nodes = []  # 方法、测试用例节点
            types = []  # 语句和token节点类型分别是2和3
            res = []  # line的res
            tokenres = []  # token的res
            nladrow = []  # = np.zeros([3200, 3200])
            nladcol = []  #
            nladval = []
            linenodes = []  # 语句节点
            linetypes = []  # 语句类型
            tokennodes = []  # token节点
            tokentypes = []  # token类型

            """应该是提前按顺序保存节点类型，如methods、ftest、rtest"""
            methodnum = len(x['methods'])
            for i in range(methodnum):
                nodes.append('Method')
                overlap.append(0)

            for i in range(len(x['ftest'])):
                nodes.append('Test')

            for i in range(len(x['rtest'])):
                nodes.append('RTest')

            mus = []
            ltype = {}
            ts = {}


            temptokenType = {}
            tokentype = {}
            for i in range(len(x['tokenKindList'])):
                if x['tokenKindList'][i] not in self.Nl_Voc:
                    self.Nl_Voc[x['tokenKindList'][i]] = len(self.Nl_Voc)
            """上面这段代码，将token的类型添加到Nl_Voc中"""
            # for i in range(len(x['tokenKindList'])):
            #     temptokenType[x['tokenKindList'][i]] = i
            """上面这段代码，将token的类型映射为数字"""
            for e1,e2 in x['tokenKindEdge']:
                tokentype[e1] = x['tokenKindList'][e2]
            """上面这段代码，将tokenid和token类型的关系映射"""
            x['tokentype'] = tokentype
            for i in range(len(x['tokenId'])):
                if i not in x['tokentype']:
                    x['tokentype'][i] = 'Empty'
                if x['tokentype'][i] not in self.Nl_Voc:  # 类型
                    self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                tokennodes.append(x['tokentype'][i])
                tokentypes.append(1)
                types.append(3)
                if i in x['errorTokensId']:
                    tokenres.append(1)
                else:
                    tokenres.append(0)
            """
            
            上面这段代码，将token的类型映射为数字，将token的类型和token的关系映射为数字
            
            """
            """"""
            # 映射，将语句类型映射为数字
            for t in x['ltype']:
                ts[x['ltype'][t]] = t
            # 构建语句和语句类型的边
            for e in x['edge3']:  # 语句，类型
                ltype[e[0]] = ts[e[1]]  # 第i个语句的类型是第e1
            # print(ltype)
            x['ltype'] = ltype
            """遍历每个语句，将语句类型映射为数字，将语句类型和语句的关系映射为数字，将语句和方法的关系映射为数字，将语句和语句的控制流关系映射为数字，将语句和语句的数据流关系映射为数字"""
            for i in range(len(x['lines'])):  # 语句
                if i not in x['ltype']:
                    x['ltype'][i] = 'Empty'
                if x['ltype'][i] not in self.Nl_Voc:  # 类型
                    self.Nl_Voc[x['ltype'][i]] = len(self.Nl_Voc)
                linenodes.append(x['ltype'][i])
                linetypes.append(1)
                types.append(2)
                if i + 1 in x['ans']:  # 纠正 左边为id，右侧为行号，所以i+1
                    res.append(1)
                else:
                    res.append(0)
            """ 
            在上段代码，最终有
            res[i]表示第i个语句是否是错误的语句
            x['lytpe'][i] 表示id为i的语句的类型为x['ltype'][i]
            linenodes[i] 表示第i个语句的类型为linenodes[i]
            linetypes[i] 表示第i个语句的类型为1，表示这个节点是语句
            types[i] 表示第i个节点的类型为2，表示这个节点是语句
            """
            maxl = max(maxl, len(nodes))  # 测试用例、方法的节点的个数
            maxl2 = max(maxl2, len(linenodes))  # 语句节点的个数
            maxl3 = max(maxl3, len(tokennodes))  # token节点的个数
            VsusFLRanks = []
            for vsusrank in range(len(x['VsusRank'])):
                VsusFLRanks.append(x['VsusRank'][vsusrank] / len(x['VsusRank']))
            ed = {}

            line2method = {}
            for e in x['edge2']:  # 方法和语句的所属关系
                line2method[e[1]] = e[0]
                a = e[0]  # a是方法
                b = e[1] + self.Nl_Len + self.Token_Len # len(x['ftest']) + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(a, b)
                    assert (0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    assert (0)
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['CFGEdge']:  # 语句和语句的控制流关系
                if e[0] not in line2method:
                    error1 += 1
                if e[1] not in line2method:
                    error1 += 1
                a = e[0] + self.Nl_Len + self.Token_Len  # len(x['ftest']) + methodnum
                b = e[1] + self.Nl_Len + self.Token_Len # len(x['ftest']) + methodnum
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(a, b)
                    # assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    # assert(0)
                nladrow.append(a)
                nladcol.append(b)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge10']:  # 语句和正确测试用例
                if e[0] not in line2method:
                    error1 += 1
                a = e[0] + self.Nl_Len+self.Token_Len
                b = e[1] + methodnum + len(x['ftest'])
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    pass
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    pass
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            for e in x['edge']:  # 语句和错误测试用例
                if e[0] not in line2method:
                    error2 += 1
                a = e[0] + self.Nl_Len + self.Token_Len  # + len(x['ftest']) + methodnum
                b = e[1] + methodnum
                nladrow.append(a)
                nladcol.append(b)
                # print(a,b,ed,"DataSet 370")
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    print(e[0], e)
                    # print(a, b,x['methods'],len(x['methods']),self.Nl_Len,len(x['ftest']))
                    # assert(0)
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    print(a, b)
                    # assert(0)
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)
            # self.preProcessToken(x)
            for e in x['tokenToLineSetEdge']:  # token --> line
                a = e[0] + self.Nl_Len
                b = e[1] + self.Nl_Len + self.Token_Len
                nladrow.append(a)
                nladcol.append(b)
                if (a, b) not in ed:
                    ed[(a, b)] = 1
                else:
                    pass
                if (b, a) not in ed:
                    ed[(b, a)] = 1
                else:
                    pass
                nladval.append(1)
                nladrow.append(b)
                nladcol.append(a)
                nladval.append(1)

            Nodes.append(self.pad_seq(self.Get_Em(nodes, self.Nl_Voc), self.Nl_Len))
            Types.append(self.pad_seq(types, self.Code_Len))

            Res.append(self.pad_seq(res, self.Code_Len))
            tokenRes.append(self.pad_seq(tokenres,self.Token_Len))

            LineMus.append(self.pad_list(mus, self.Code_Len, 3))
            inputText.append(self.pad_seq(overlap, self.Nl_Len))
            # inputText.append(self.pad_list(text, self.Nl_Len, 10))
            LineNodes.append(self.pad_seq(self.Get_Em(linenodes, self.Nl_Voc), self.Code_Len))
            tokenNodes.append(self.pad_seq(self.Get_Em(tokennodes, self.Nl_Voc), self.Token_Len))  # token类型
            LineTypes.append(self.pad_seq(linetypes, self.Code_Len))
            tokenTypes.append(self.pad_seq(tokentypes, self.Token_Len))
            LineRank.append(self.pad_seq(VsusFLRanks, self.Code_Len))
            # toke
            """
            Nodes，第一个特征，保存了方法、测试用例节点的特征/标识
            Types, 第二个特征，标识所有语句的类型和token类型，分别是2和3
            Res， 错误语句
            LineMus： 不知道
            inputTest： 方法特征/标识
            LineNodes：  语句的类型
            tokenTypes： token类型
            LineTypes： 对于语句节点，存储1，否则填充0
            LineRank： 排名
            """
            row = {}
            col = {}
            for i in range(len(nladrow)):
                if nladrow[i] not in row:
                    row[nladrow[i]] = 0
                row[nladrow[i]] += 1
                if nladcol[i] not in col:
                    col[nladcol[i]] = 0
                col[nladcol[i]] += 1
            for i in range(len(nladrow)):
                nladval[i] = 1 / math.sqrt(row[nladrow[i]]) * 1 / math.sqrt(col[nladcol[i]])
            # print(self.Nl_Len)
            nlad = sparse.coo_matrix((nladval, (nladrow, nladcol)), shape=(
                self.Nl_Len + self.Code_Len + self.Token_Len, self.Nl_Len + self.Code_Len + self.Token_Len))
            inputNlad.append(nlad)
        print("max1: %d max2: %d" % (maxl, maxl2))
        print("correct: %d error: %d" % (correct, error))
        print("error1: %d error2: %d" % (error1, error2))
        f = open('nl_voc.pkl', 'wb')
        print(self.Nl_Voc)
        pickle.dump(self.Nl_Voc, f)
        f.close()
        # assert(0)#assert(0)
        batchs = [Nodes, Types, inputNlad, Res, inputText, LineNodes, LineTypes, LineMus, LineRank, tokenTypes, tokenRes,tokenNodes]

        self.data = batchs
        open(self.proj + "data.pkl", "wb").write(pickle.dumps(batchs, protocol=4))
        # open('nl_voc.pkl', 'wb').write(pickle.dumps(self.Nl_Voc))
        return batchs

    def __getitem__(self, offset):
        ans = []
        if True:
            for i in range(len(self.data)):
                if i == 2:
                    ans.append(self.data[i][offset].toarray())
                else:
                    ans.append(np.array(self.data[i][offset]))
        else:
            for i in range(len(self.data)):
                if i == 4:
                    continue
                ans.append(np.array(self.data[i][offset]))
            negoffset = random.randint(0, len(self.data[0]) - 1)
            while negoffset == offset:
                negoffset = random.randint(0, len(self.data[0]) - 1)
            if self.dataName == "train":
                ans.append(np.array(self.data[2][negoffset]))
                ans.append(np.array(self.data[3][negoffset]))
        return ans

    def __len__(self):
        return len(self.data[0])

    def Get_Train(self, batch_size):
        data = self.data
        loaddata = data
        batch_nums = int(len(data[0]) / batch_size)
        if True:
            if self.dataName == 'train':
                shuffle = np.random.permutation(range(len(loaddata[0])))
            else:
                shuffle = np.arange(len(loaddata[0]))
            for i in range(batch_nums):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_size * i: batch_size * (i + 1)]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * i, batch_size * (i + 1)):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append(
                                    [idx - batch_size * i, data[j][shuffle[idx]].row[p], data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size(
                            [batch_size, self.Nl_Len + self.Code_Len, self.Nl_Len + self.Code_Len])))
                yield ans
            if batch_nums * batch_size < len(data[0]):
                ans = []
                for j in range(len(data)):
                    if j != 2:
                        tmpd = np.array(data[j])[shuffle[batch_nums * batch_size:]]
                        ans.append(torch.from_numpy(np.array(tmpd)))
                    else:
                        ids = []
                        v = []
                        for idx in range(batch_size * batch_nums, len(data[0])):
                            for p in range(len(data[j][shuffle[idx]].row)):
                                ids.append([idx - batch_size * batch_nums, data[j][shuffle[idx]].row[p],
                                            data[j][shuffle[idx]].col[p]])
                                v.append(data[j][shuffle[idx]].data[p])
                        ans.append(torch.sparse.FloatTensor(torch.LongTensor(ids).t(), torch.FloatTensor(v), torch.Size(
                            [len(data[0]) - batch_size * batch_nums, self.Nl_Len + self.Code_Len,
                             self.Nl_Len + self.Code_Len])))
                yield ans


class node:
    def __init__(self, name):
        self.name = name
        self.father = None
        self.child = []
        self.id = -1
