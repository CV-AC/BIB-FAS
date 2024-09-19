import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
from timm.models.pvt_v2 import pvt_v2_b2

class fas_base_model(nn.Module):

    def __init__(self, pretrained=True):
        super(fas_base_model, self).__init__()
        final_dim = 512

        self.encoder = resnet18(pretrained=True)

        self.fc0 = nn.Sequential(nn.Linear(final_dim, 1),
                                 nn.Sigmoid())

        self.bnneck = nn.BatchNorm2d(final_dim)
        self.bnneck.bias.requires_grad_(False)  # no shift

        self.live_classifier = nn.Linear(final_dim, 2, bias=False)
        self.spoof_classifier = nn.Linear(final_dim, 2, bias=False)

        self.live_classifier_ = nn.Linear(final_dim, 2, bias=False)
        self.live_classifier_.weight.requires_grad_(False)
        self.live_classifier_.weight.data = self.live_classifier.weight.data

        self.spoof_classifier_ = nn.Linear(final_dim, 2, bias=False)
        self.spoof_classifier_.weight.requires_grad_(False)
        self.spoof_classifier_.weight.data = self.spoof_classifier.weight.data

        # reduction='batchmean'
        self.KLDivLoss = nn.KLDivLoss()
        self.update_rate = 0.2
        self.update_rate_ = self.update_rate
        self.id_loss = nn.CrossEntropyLoss(label_smoothing=0.01)

    def _encode(self, x):

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        
        return x

    def forward(self, x):

        x = self._encode(x)
        x_mertic = self.bnneck(x)

        x = x.mean([-2, -1])
        x_mertic = x_mertic.mean([-2, -1])

        b = x.shape[0]
        live = x[0:b // 2]
        spoof = x[b // 2:b]
        lable_live = torch.zeros(b // 2).cuda().long()
        label_spoof = torch.ones(b // 2).cuda().long()

        logits_live = self.live_classifier(live)
        live_cls_loss = self.id_loss(logits_live.float(), lable_live)

        logits_spoof = self.spoof_classifier(spoof)
        spoof_cls_loss = self.id_loss(logits_spoof.float(), label_spoof)

        logits_m = torch.cat([logits_live, logits_spoof], 0).float()

        with torch.no_grad():
            self.spoof_classifier_.weight.data = self.spoof_classifier_.weight.data * (1 - self.update_rate) \
                                                 + self.spoof_classifier.weight.data * self.update_rate
            self.live_classifier_.weight.data = self.live_classifier_.weight.data * (1 - self.update_rate) \
                                                + self.live_classifier.weight.data * self.update_rate

            logits_live_ = self.spoof_classifier_(live)
            logits_spoof_ = self.live_classifier_(spoof)
            logits_m_ = torch.cat([logits_live_, logits_spoof_], 0).float()

        logits_m = F.softmax(logits_m, 1)
        logits_m_ = F.log_softmax(logits_m_, 1)
        mod_loss = self.KLDivLoss(logits_m_, logits_m)

        if self.training:
            return [x, x_mertic, mod_loss, live_cls_loss, spoof_cls_loss]
        else:
            return self.fc0(x)