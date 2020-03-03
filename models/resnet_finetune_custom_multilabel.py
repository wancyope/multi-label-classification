import chainer
import chainer.functions as F
from chainer import links as L
from chainer.links.model.vision.resnet import ResNet50Layers


class Encoder(chainer.Chain):
    #def __init__(self,classlabels=123):
    def __init__(self,classlabels=66):
        super(Encoder, self).__init__(
            #model = L.ResNet50Layers(),
            model = ResNet50Layers(),
            fc = L.Linear(None,classlabels))
        
    def __call__(self, x,t):
        h = self.model(x,layers=['res5'])['res5']
        h = F.average_pooling_2d(h, ksize=(h.shape[2],h.shape[3]), stride=1)
        h = self.fc(h)
        
        loss =  F.sigmoid_cross_entropy(h, t)
        #summary = F.classification_summary(h,t,beta = 1.0)
        chainer.report({'loss': loss,
                        #'accuracy': F.accuracy(h, t),
                        #'precision': chainer.functions.mean(summary[0]),
                        #'recall': chainer.functions.mean(summary[1]),
                        #'f_value': chainer.functions.mean(summary[2])
                        },self)
        return loss
