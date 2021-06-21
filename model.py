
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics
import torch
import torchvision.models as models

from argparse import ArgumentParser
from torch.nn import functional as F
from torchvision import transforms

from augmentations import basic_augmentations, color_augmentations, no_augmentations, mdmm_augmentations
from data_utils import plot_confusion_matrix
from mdmm.model import MD_multi


class Args:
    concat = 1
    crop_size = 216 # only used as an argument for training
    dis_norm = None
    dis_scale = 3
    dis_spectral_norm = False
    dataroot ='data'
    gpu = 1
    input_dim = 3
    isDcontent = False
    nThreads = 4
    num_domains = 5
    nz = 8
    resume = '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/mdmm_model.pth'

    
class Classifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, l2_reg=1e-6, weighted=False, dropout=0.0, mdmm_aug=False, mdmm_norm=None, transform=no_augmentations):
        super().__init__()
        self.save_hyperparameters()
        self.mdmm_aug = mdmm_aug
        self.mdmm_norm = mdmm_norm

        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
            
        for param in self.model.layer3.parameters():
            param.requires_grad = True
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        self.model.fc = torch.nn.Sequential(
                            torch.nn.Dropout(p=dropout),
                            torch.nn.Linear(512, 1),
        )

#         self.weight = torch.ones(2)
        self.weight = torch.ones(1)
        if weighted:
#             self.weight[0] = 0.08
            self.weight *= 12.5
        
        self.hp_metric = -1
        self.count = 0
        
        if self.mdmm_aug or self.mdmm_norm in range(5):
            opts = Args()
            aug_model = MD_multi(opts)
            aug_model.resume(opts.resume, train=False)
            aug_model.eval();
            self.enc = aug_model.enc_c
            self.gen = aug_model.gen
            
            self.shift = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.transforms_after = transform
            
            
            self.mean_domains = torch.Tensor([
                [ 0.3020, -2.6476, -0.9849, -0.7820, -0.2746,  0.3361,  0.1694, -1.2148],
                [ 0.1453, -1.2400, -0.9484,  0.9697, -2.0775,  0.7676, -0.5224, -0.2945],
                [ 2.1067, -1.8572,  0.0055,  1.2214, -2.9363,  2.0249, -0.4593, -0.9771],
                [ 0.8378, -2.1174, -0.6531,  0.2986, -1.3629, -0.1237, -0.3486, -1.0716],
                [ 1.6073,  1.9633, -0.3130, -1.9242, -0.9673,  2.4990, -2.2023, -1.4109],
            ])

            self.std_domains = torch.Tensor([
                [0.6550, 1.5427, 0.5444, 0.7254, 0.6701, 1.0214, 0.6245, 0.6886],
                [0.4143, 0.6543, 0.5891, 0.4592, 0.8944, 0.7046, 0.4441, 0.3668],
                [0.5576, 0.7634, 0.7875, 0.5220, 0.7943, 0.8918, 0.6000, 0.5018],
                [0.4157, 0.4104, 0.5158, 0.3498, 0.2365, 0.3612, 0.3375, 0.4214],
                [0.6154, 0.3440, 0.7032, 0.6220, 0.4496, 0.6488, 0.4886, 0.2989],
            ])
#             print(self.enc)
            print('mdmm initialized')

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.mdmm_aug:
            bs, _, img_h, img_w = x.shape
            indices = torch.randint(2, (bs, )) # mdmm augmentations are applied with probability 0.5
            num_aug = indices.sum()

#             indices = torch.randint(5, (bs, )).bool() # mdmm augmentations are applied with probability 0.2
#             num_aug = indices.sum()
        
            if num_aug > 0:
                new_domains = torch.randint(5, (num_aug, )).to(self.device)
#                 new_domains = (torch.ones((num_aug, ))*2).to(self.device).long()
                z_attr = (torch.randn((num_aug, 8, )) * self.std_domains[new_domains] + self.mean_domains[new_domains]).to(self.device)
                domain = torch.eye(5)[new_domains].to(self.device)

                x_aug = x[indices.bool()]
#                 z_content = self.enc(x_aug.sub(0.5).mul(2))
                z_content = self.enc(x_aug)
                x_aug = self.gen(z_content, z_attr, domain).detach() # in range [-1, 1]
#                 x_aug = x_aug.add(1.).div(2)
#                 mean, std = x_aug.mean(dim=(0, 2, 3)), x_aug.std(dim=(0, 2, 3))
#                 x_aug = x_aug.sub(mean[None, :, None, None]).div(std[None, :, None, None])
                x[indices.bool()] = x_aug

        if self.count < 6:
#                 mean = torch.Tensor([0.6710, 0.5327, 0.6448]).to(self.device)
#                 std = torch.Tensor([0.2083, 0.2294, 0.1771]).to(self.device)
#                 if indices[0].item() == 1:
#                     img = x[0].detach().mul(std[ :, None, None]).add(mean[ :, None, None])
#                 else:
            img = x[0].detach().add(1.).div(2)
            self.logger.experiment.add_image(f'train_images/{self.count}', img, global_step=self.global_step)
            self.count += 1
        
#             x[~indices.bool()] = self.transforms_after(x[~indices.bool()])
            
            # apply random erasing
#             indices = torch.randint(2, (bs, )) # mdmm augmentations are applied with probability 0.5
#             n = indices.sum()
            
#             scale, ratio = (0.02, 0.33), (0.3, 3.3)
#             area = img_h * img_w
#             erase_area = area * torch.empty((n, )).uniform_(scale[0], scale[1]).item()
#             aspect_ratio = torch.empty((n, )).uniform_(ratio[0], ratio[1]).item()
                
        y_hat = self.forward(x)
        
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('train_loss', loss, on_epoch=True)
        
        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # accuracy 
        self.log('train_metrics/acc', metrics.classification.accuracy(preds, y, num_classes=2), on_epoch=True)
        
        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)
        
        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}
    
    def training_epoch_end(self, train_outputs):
        logits = torch.cat([batch['outputs'] for batch in train_outputs]).squeeze(-1)     
        targets = torch.cat([batch['targets'] for batch in train_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1) # for fast_dev_run
        if targets.sum() > 0.:
            precision, recall, _ = metrics.precision_recall_curve(logits, targets)
            pr_auc = metrics.classification.auc(recall, precision)
        self.log('train_metrics/PR_AUC', pr_auc, on_epoch=True)
        
        cm = torch.stack([batch['cm'] for batch in train_outputs]).sum(dim=0)

        if (cm[0, 0] + cm[1, 0]) > 0. and (cm[0, 0] + cm[0, 1]) > 0. and (cm[1, 1] + cm[0, 1]) > 0. and (cm[1, 1] + cm[1, 0]) > 0.:
            # log precision and recall
            prec_n = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            recall_n = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            prec_t = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            recall_t = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            self.log('train_metrics/precision_normal', prec_n, on_epoch=True)
            self.log('train_metrics/recall_normal', recall_n, on_epoch=True)
            self.log('train_metrics/precision_tumor', prec_t, on_epoch=True)
            self.log('train_metrics/recall_tumor', recall_t, on_epoch=True)

            cm_figure = plot_confusion_matrix(cm.cpu().numpy(), ['normal', 'tumor'])
            self.logger.experiment.add_figure('confusion_matrix/train', cm_figure, global_step=self.global_step)

            # log F1 score
            self.log('train_metrics/F1_normal', 2 * prec_n * recall_n / (prec_n + recall_n))            
            self.log('train_metrics/F1_tumor', 2 * prec_t * recall_t / (prec_t + recall_t))
            
    def validation_step(self, batch, batch_idx):
        x, y = batch  # (bs, 3, 512, 512), (bs, 1)

        y_hat = self.forward(x)
#         loss = F.cross_entropy(y_hat[0], y.long().squeeze(-1), weight=self.weight.to(self.device))
#         loss = F.cross_entropy(y_hat, y.long().squeeze(-1), weight=self.weight.to(self.device))
        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('val_loss', loss, on_epoch=True)
        
        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # accuracy 
        self.log('val_metrics/acc', metrics.classification.accuracy(preds, y, num_classes=2), on_epoch=True)
        
        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)
                
        # log specific images
#         pos_list = torch.Tensor([2061, 2071, 2246, 2486, 2522, 3262])
#         neg_list = torch.Tensor([3396, 2, 4264, 1588, 3756, 2699])
        
#         bs, _ = y.shape
#         batch_idx = torch.Tensor([batch_idx])
        
#         pos_mask = batch_idx.le(pos_list // bs) * (pos_list // bs).le(batch_idx)
#         neg_mask = batch_idx.le(neg_list // bs) * (neg_list // bs).le(batch_idx)
        
#         pos_ind = pos_list[pos_mask] % bs
#         neg_ind = neg_list[neg_mask] % bs
        
#         mean = torch.Tensor([0.6710, 0.5327, 0.6448]).to(self.device)
#         std = torch.Tensor([0.2083, 0.2294, 0.1771]).to(self.device)
        
#         for i in pos_ind.tolist():
#             img = x[int(i)].unsqueeze(0) * std[None, :, None, None] + mean[None, :, None, None]
#             if preds[int(i)].item() == 0:
#                 img = F.pad(img, (10, 10, 10, 10)) # pad if image was wrongly classified
#             img = F.interpolate(img, size=(128, 128))
#             self.logger.experiment.add_image(f'tumor/{i}', img.squeeze(0), self.global_step)
#         for i in neg_ind.tolist():
#             img = x[int(i)].unsqueeze(0) * std[None, :, None, None] + mean[None, :, None, None]
#             if preds[int(i)].item() == 1:
#                 img = F.pad(img, (10, 10, 10, 10))
#             img = F.interpolate(img, size=(128, 128))
#             self.logger.experiment.add_image(f'normal/{i}', img.squeeze(0), self.current_epoch)

        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}

    def validation_epoch_end(self, val_outputs):
        logits = torch.cat([batch['outputs'] for batch in val_outputs]).squeeze(-1)     
        targets = torch.cat([batch['targets'] for batch in val_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1) # for fast_dev_run
        if targets.sum() > 0.:
            precision, recall, _ = metrics.precision_recall_curve(logits, targets)
            pr_auc = metrics.classification.auc(recall, precision)
            
            # plot the PR curve
            fig = plt.figure()
            tumor_ratio = len(targets[targets==1.]) / len(targets)
            plt.plot([0, 1], [tumor_ratio, tumor_ratio], linestyle='--', label='random')
            plt.plot(recall.cpu(), precision.cpu(), marker='.', label='our model')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            self.logger.experiment.add_figure('prec-recall-curve', fig, global_step=self.global_step)
        self.log('val_metrics/PR_AUC', pr_auc, on_epoch=True)
        self.log('PR_AUC', pr_auc, on_epoch=True) # log this separately for model checkpoint callback
        
        # compute metrics based on confusion matrix
        cm = torch.stack([batch['cm'] for batch in val_outputs]).sum(dim=0)

        if (cm[0, 0] + cm[1, 0]) > 0. and (cm[0, 0] + cm[0, 1]) and (cm[1, 1] + cm[0, 1]) and (cm[1, 1] + cm[1, 0]):
            # log precision and recall
            prec_n = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            recall_n = cm[0, 0] / (cm[0, 0] + cm[0, 1])
            prec_t = cm[1, 1] / (cm[1, 1] + cm[0, 1])
            recall_t = cm[1, 1] / (cm[1, 1] + cm[1, 0])
            self.log('val_metrics/precision_normal', prec_n, on_epoch=True)
            self.log('val_metrics/recall_normal', recall_n, on_epoch=True)
            self.log('val_metrics/precision_tumor', prec_t, on_epoch=True)
            self.log('val_metrics/recall_tumor', recall_t, on_epoch=True)

            cm_figure = plot_confusion_matrix(cm.cpu().numpy(), ['normal', 'tumor'])
            self.logger.experiment.add_figure('confusion_matrix/val', cm_figure, global_step=self.global_step)

            # log F1 score
            self.log('val_metrics/F1_normal', 2 * prec_n * recall_n / (prec_n + recall_n))            
            self.log('val_metrics/F1_tumor', 2 * prec_t * recall_t / (prec_t + recall_t))
            self.log('F1_tumor', 2 * prec_t * recall_t / (prec_t + recall_t))

 
        # compute area under the ROC curve
        if pr_auc.item() > self.hp_metric:
            self.logger.experiment.add_scalar('hp_metric', pr_auc, global_step=0)
            self.hp_metric = pr_auc.item()
                
    def test_step(self, batch, batch_idx):
        x, y = batch  # (bs, 3, 512, 512), (bs, 1)
        
        if self.mdmm_norm in range(5):
            bs, _, _, _ = x.shape
            z_attr = (torch.randn((bs, 8, )) * self.std_domains[self.mdmm_norm] + self.mean_domains[self.mdmm_norm]).to(self.device)
            domain = torch.eye(5)[self.mdmm_norm].to(self.device).unsqueeze(0).repeat(bs, 1)
            z_content = self.enc(x)
            x = self.gen(z_content, z_attr, domain).detach() # in range [-1, 1]

        y_hat = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('test_loss', loss, on_epoch=True)
        
        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # accuracy 
        self.log('test_metrics/acc', metrics.classification.accuracy(preds, y, num_classes=2), on_epoch=True)
        
        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)
        
        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}

    def test_epoch_end(self, test_outputs):
        logits = torch.cat([batch['outputs'] for batch in test_outputs]).squeeze(-1)      
        targets = torch.cat([batch['targets'] for batch in test_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1) # for fast_dev_run
        if targets.sum() > 0.:
            precision, recall, _ = metrics.precision_recall_curve(logits, targets)
            pr_auc = metrics.classification.auc(recall, precision)
        self.log('test_metrics/PR_AUC', pr_auc, on_epoch=True)
        
        # compute metrics based on confusion matrix
        cm = torch.stack([batch['cm'] for batch in test_outputs]).sum(dim=0)

        # log precision and recall
        prec_n = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        recall_n = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        prec_t = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        recall_t = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        self.log('test_metrics/precision_normal', prec_n, on_epoch=True)
        self.log('test_metrics/recall_normal', recall_n, on_epoch=True)
        self.log('test_metrics/precision_tumor', prec_t, on_epoch=True)
        self.log('test_metrics/recall_tumor', recall_t, on_epoch=True)

        cm_figure = plot_confusion_matrix(cm.cpu().numpy(), ['normal', 'tumor'])
        self.logger.experiment.add_figure('confusion_matrix/test', cm_figure, global_step=self.global_step)

        # log F1 score
        self.log('test_metrics/F1_normal', 2 * prec_n * recall_n / (prec_n + recall_n))            
        self.log('test_metrics/F1_tumor', 2 * prec_t * recall_t / (prec_t + recall_t))
        
        return {
            'precision_normal': prec_n,
            'recall_normal': recall_n,
            'precision_tumor': prec_t,
            'recall_tumor': recall_t,
            'confusion_matrix_00': cm[0, 0],
            'confusion_matrix_01': cm[0, 1],
            'confusion_matrix_10': cm[1, 0],
            'confusion_matrix_11': cm[1, 1],
            'F1_normal': 2 * prec_n * recall_n / (prec_n + recall_n),
            'F1_tumor': 2 * prec_t * recall_t / (prec_t + recall_t),
            'PR_AUC': pr_auc,
        }
        
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.l2_reg
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--l2_reg', type=float, default=1e-06)
        return parser


class Backbone(torch.nn.Module):
    """
    Backbone of the Classification modelin the paper "Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology "
    """
    def __init__(self, in_channels=3):
        global ch_in, ch_out
        super().__init__()
        self.conv_layers = torch.nn.Sequential()

        # initial convolutional layer
        self.conv_layers.add_module(torch.nn.Conv2d(in_channels, 32))
        self.conv_layers.add_module(torch.nn.ReLU())

        # append 9 convolutional layers
        for i in range(9):
            if i % 2 == 0:
                if i == 0:
                    ch_in = in_channels
                else:
                    ch_in = 32 * i
                ch_out = 32 * (i + 1)
                s = 1
            else:
                ch_in = 32 * i
                ch_out = 32 * i
                s = 2

            self.conv_layers.add_module(f'conv_{i+1}', torch.nn.Conv2d(ch_in, ch_out, stride=s))
            self.conv_layers.add_module(f'bn_{i+1}', torch.nn.BatchNorm2d(ch_out))
            self.conv_layers.add_module(f'act_fct_{i+1}', torch.nn.LeakyReLU())

        self.fc_layers = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.mean(x, dim=(2, 3), keepdim=True)
        x = self.fc_layers(x)
        return x
