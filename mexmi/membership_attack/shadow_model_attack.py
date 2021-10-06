import random

import numpy as np

from base_sss import SubsetSelectionStrategy
from classifier import iterate_minibatches
from utils.utils import clipDataTopX

class ShadowModelMemberAttack(SubsetSelectionStrategy):
        def __init__(self, size, shadow_attack_model, Y_vec, batch_size=10, previous_s=None):
            super(ShadowModelMemberAttack, self).__init__(size, Y_vec)
            self.shadow_attack_model = shadow_attack_model
            self.batch_size = batch_size
            self.previous_s = previous_s

        def get_subset(self):
            assert self.shadow_attack_model is not None, 'shadow_attack_model is none!'
            idx_chosen = [] #np.zeros([self.size])()
            training_confidence=[]
            if self.previous_s is not None:
                y_u = np.asarray([self.Y_vec[i] for i in self.previous_s])
            else:
                y_u = self.Y_vec
            targets = np.zeros(y_u.shape[0])
            # print("self.Y_vec,length", y_u.shape)
            # print("self.Y_vec:", self.Y_vec[:10])

            for input_batch, _ in iterate_minibatches(inputs=y_u, targets=targets, batch_size=self.batch_size, shuffle=False):
                # print("shadow_model_attack: input_batch:", input_batch.shape)
                input = clipDataTopX(input_batch, top=3)
                # top= [i[0]>0.5 for i in input]
                # print("top", top)
                pred = self.shadow_attack_model(input) #output
                training_confidence.append([p[1] for p in pred])

                # break
            if (training_confidence is not None) and (len(training_confidence) > 0):
                training_confidence = np.concatenate(training_confidence)
                sort_idx = np.argsort(training_confidence)  # 1,2,3,4
                # print("training_confidence", training_confidence.shape)
                # print("training_confidence,", training_confidence)
                for i in sort_idx:
                    if training_confidence[i] > 0.65:  # 0.67
                        idx_chosen.append(i)
                        # print("confidence is over 0.5!")
            # print("idx_chosen", len(idx_chosen))
            # if len(idx_chosen)<self.size:
            #     idx_chosen = random.sample([i for i in range(len(self.Y_vec))],  self.size)#sort_idx[:self.size]
            # print("idx_chosen,",len(idx_chosen))
            # return idx_chosen[:self.size] #most confident
            # if len(idx_chosen) < self.size:
            #     print("while decide idx_chosen, training data is not enough!")
            #     idx_chosen=sort_idx[-self.size:]
            # # print("idx_chosen,", len(idx_chosen))
            if self.previous_s is not None:
                idx_final = [self.previous_s[i] for i in idx_chosen]
            else:
                idx_final = idx_chosen
            return idx_final #most confident  [:self.size]

def adjust_thre(fragment):
    # thre=0.5
    # thre = 1/(1+np.exp((-fragment+0.0)*5))
    thre = 1 / (1 + np.exp((-fragment + 0.05) * 9))
    return thre

class AdjShadowModelMemberAttack(SubsetSelectionStrategy):
    def __init__(self, size, shadow_attack_model, Y_vec, batch_size=10, transfery=None, it=0):
        # thresholds=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]):
        super(AdjShadowModelMemberAttack, self).__init__(size, Y_vec)
        self.shadow_attack_model = shadow_attack_model
        self.batch_size = batch_size
        self.pre_transfery = transfery  # already got
        # self.threholds = thresholds
        self.it = it

    def get_subset(self):
        assert self.shadow_attack_model is not None, 'shadow_attack_model is none!'
        # np.zeros([self.size])()
        training_confidence = []
        maxs = []
        conf_de_num = [[], [], [], [], [], [], [], [], [], []]
        conf_de_idx = [[], [], [], [], [], [], [], [], [], []]
        targets = np.zeros(self.Y_vec.shape[0])
        print("self.Y_vec,length", self.Y_vec.shape)
        # print("self.Y_vec:", self.Y_vec[:10])
        # if self.it<6:
        #     th=0.55
        # else:
        #     th=0.6
        th=0.5 # in good presentation is 0.55

        thresholds = (th*np.ones([10])).tolist()
        labels = np.zeros(10)
        for pre_y in range(len(self.pre_transfery)):
            self.pre_transfery = np.asarray(self.pre_transfery)
            label = np.argmax(self.pre_transfery[pre_y])
            labels[label] = labels[label] + 1
        # for lb in range(len(labels)):  # 10
        #     fragment = 1.0 * labels[lb] / len(self.pre_transfery)
        #     print("class {} has framgent {}".format(lb, fragment))
        #     thresholds[lb] = adjust_thre(fragment)
        for input_batch, _ in iterate_minibatches(inputs=self.Y_vec, targets=targets, batch_size=self.batch_size,
                                                  shuffle=False):
            # print("shadow_model_attack: input_batch:", input_batch.shape)
            # inputs = np.zeros(10).tolist()
            maxs.append(np.argmax(input_batch, axis=1))
            input = clipDataTopX(input_batch, top=3)
            # top= [i[0]>0.5 for i in input]
            # print("top", top)
            pred = self.shadow_attack_model(input)  # output
            training_confidence.append([p[1] for p in pred])
        training_confidence = np.concatenate(training_confidence)
        maxs = np.concatenate(maxs)
        # not j, but j+batch_size
        # print(len(maxs))
        # conf_de_cls is corresponding to conf_de_idx
        for j, lb in enumerate(maxs):
            conf_de_num[lb].append(training_confidence[j])
            conf_de_idx[lb].append(j)
        # print("j:", j)
        # sort_idx_de_cls= [[], [], [], [], [], [], [], [], [], []]
        print("conf_de_cls0:", len(conf_de_num[0]))
        print("conf_de_cls1:", len(conf_de_num[1]))
        print("conf_de_cls2:", len(conf_de_num[2]))
        print("conf_de_cls3:", len(conf_de_num[3]))
        print("conf_de_cls4:", len(conf_de_num[4]))
        print("conf_de_cls5:", len(conf_de_num[5]))
        print("conf_de_cls6:", len(conf_de_num[6]))
        print("conf_de_cls7:", len(conf_de_num[7]))
        print("conf_de_cls8:", len(conf_de_num[8]))
        print("conf_de_cls9:", len(conf_de_num[9]))
        idx_chosen_de_cls = [[], [], [], [], [], [], [], [], [], []]
        idx_chosen_final = []
        # saved_idx = []
        saved_confi = []
        for cls in range(10):
            sort_idx_de = np.argsort(conf_de_num[cls])
            # print(conf_de_cls[cls])
            idx_chosen = []
            if conf_de_num[cls] is not []:
                # conf_de_cls[cls] = np.concatenate(conf_de_cls[cls])
                for i_o in range(len(conf_de_num[cls])):  # 0
                    if conf_de_num[cls][i_o] > thresholds[cls]:
                        idx_chosen.append(i_o)
                        # print("con_de_cls_[{}]:".format(cls), conf_de_cls[cls][i_o], +",threholds=",
                        #       thresholds[cls])
                idx_chosen_de_cls[cls] = [conf_de_idx[cls][i_t] for i_t in idx_chosen]
                if len(idx_chosen_de_cls[cls]) < self.size / 10:
                    idx_chosen = sort_idx_de[-int(self.size / 10):]
                    idx_chosen_de_cls[cls] = [conf_de_idx[cls][i_t] for i_t in
                                              idx_chosen]  # random.sample(conf_de_idx,int(self.size/10))
            else:
                if len(sort_idx_de) < self.size:
                    idx_chosen = sort_idx_de
                else:
                    idx_chosen = sort_idx_de[-int(self.size / 10):]
                idx_chosen_de_cls[cls] = [conf_de_idx[cls][i_t] for i_t in
                                          idx_chosen]  # random.sample(conf_de_idx,int(self.size/10))
            idx_chosen_final.append(idx_chosen_de_cls[cls])
            # saved_idx.append(idx_chosen_de_cls[cls])
            saved_confi.append([conf_de_num[cls][ifor1] for ifor1 in idx_chosen])
        idx_chosen_final = np.concatenate(idx_chosen_final)
        # sort_idx = np.argsort(training_confidence) #1,2,3,4
        # for i in sort_idx:
        #     if training_confidence[i] >0.5:
        #         idx_chosen.append(i)
        # print("confidence is over 0.5!")
        # print("idx_chosen", len(idx_chosen))
        # if len(idx_chosen)<self.size:
        #     idx_chosen = random.sample([i for i in range(len(self.Y_vec))],  self.size)#sort_idx[:self.size]
        # print("idx_chosen,",len(idx_chosen))
        # return idx_chosen[:self.size] #most confident
        # if len(idx_chosen) < self.size:
        #     print("while decide idx_chosen, training data is not enough!")
        #     idx_chosen=sort_idx[-self.size:]
        # print("idx_chosen,", len(idx_chosen))
        # adjust threholds.
        return idx_chosen_final, idx_chosen_de_cls, saved_confi  # most confident  [:self.size]

class ConfShadowModelMemberAttack(SubsetSelectionStrategy):
    def __init__(self, size, shadow_attack_model, Y_vec, batch_size=10):
        super(ConfShadowModelMemberAttack, self).__init__(size, Y_vec)
        self.shadow_attack_model = shadow_attack_model
        self.batch_size = batch_size

    def get_subset(self):
        assert self.shadow_attack_model is not None, 'shadow_attack_model is none!'
        idx_chosen = []  # np.zeros([self.size])()
        training_confidence = []
        targets = np.zeros(self.Y_vec.shape[0])
        print("self.Y_vec,length", self.Y_vec.shape)
        for input_batch, _ in iterate_minibatches(inputs=self.Y_vec, targets=targets, batch_size=self.batch_size,
                                                  shuffle=False):
            # print("shadow_model_attack: input_batch:", input_batch.shape)
            input = clipDataTopX(input_batch, top=2)
            # print("shadow_model_attack: input:", input)
            pred = self.shadow_attack_model(input)  # output
            # print("shadow_model_attack: pred:", pred)
            training_confidence.append([p[1] for p in pred])
            # break
        training_confidence = np.concatenate(training_confidence)
        sort_idx = np.argsort(-1*(training_confidence))
        print("training_confidence", len(training_confidence))
        for i in sort_idx:
            if training_confidence[i] > 0.5:
                idx_chosen.append(i)
        print("idx_chosen", len(idx_chosen))
        if len(idx_chosen) < self.size:
            idx_chosen = sort_idx[:self.size]
        print("idx_chosen,", len(idx_chosen))
        return idx_chosen[:self.size]