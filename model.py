
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as metrics
import torch
import torchvision.models as models

from argparse import ArgumentParser
from torch.nn import functional as F
from torchvision import transforms

from augmentations import basic_augmentations, color_augmentations, no_augmentations, gan_augmentations, mean_domains, std_domains
from utils import plot_confusion_matrix
from histaugan.model import MD_multi


class Args:
    concat = 1
    crop_size = 216  # only used as an argument for training
    dis_norm = None
    dis_scale = 3
    dis_spectral_norm = False
    dataroot = 'data'
    gpu = 1
    input_dim = 3
    isDcontent = False
    nThreads = 4
    num_domains = 5
    nz = 8
    resume = '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/gan_weights.pth'


class Classifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, l2_reg=1e-6, weighted=False, dropout=0.0, gan_aug=False, transform=no_augmentations):
        super().__init__()
        self.save_hyperparameters()
        self.gan_aug = gan_aug

        self.model = models.resnet18(pretrained=True)
        # freeze the first resnet blocks
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

        if weighted:
            self.weight = 12.5

        self.hp_metric = -1
        self.count = 0

        # initialize GAN for augmentations
        if self.gan_aug:
            opts = Args()
            aug_model = MD_multi(opts)
            aug_model.resume(opts.resume, train=False)
            aug_model.eval()
            self.enc = aug_model.enc_c
            self.gen = aug_model.gen

            self.shift = transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.transforms_after = transform

            self.mean_domains = mean_domains
            self.std_domains = std_domains

            print('histaugan initialized')

    def forward(self, x):
        out = self.model(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.gan_aug:
            # ----------------------
            # HistAuGAN augmentation
            # ----------------------
            bs, _, _, _ = x.shape
            
            indices = torch.randint(2, (bs, ))  # augmentations are applied with probability 0.5
            num_aug = indices.sum()

            if num_aug > 0:
                # sample new domain
                new_domains = torch.randint(5, (num_aug, )).to(self.device)
                domain = torch.eye(5)[new_domains].to(self.device)

                # sample attribute vector
                z_attr = (torch.randn(
                    (num_aug, 8, )) * self.std_domains[new_domains] + self.mean_domains[new_domains]).to(self.device)

                # compute content encoding
                z_content = self.enc(x[indices.bool()])

                # generate augmentations
                x_aug = self.gen(z_content, z_attr, domain).detach()  # in range [-1, 1]

                x[indices.bool()] = x_aug
            # ----------------------

        # for visualization, log the first image of the first 6 batches 
        if self.count < 6:
            img = x[0].detach().add(1.).div(2)
            self.logger.experiment.add_image(f'train_images/{self.count}', img, global_step=self.global_step)
            self.count += 1

        # forward
        y_hat = self.forward(x)

        # compute and log loss
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('train_loss', loss, on_epoch=True)

        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # log accuracy
        self.log('train_metrics/acc', metrics.classification.accuracy(preds,
                 y, num_classes=2), on_epoch=True)

        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)

        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}

    def training_epoch_end(self, train_outputs):
        logits = torch.cat([batch['outputs'] for batch in train_outputs]).squeeze(-1)
        targets = torch.cat([batch['targets'] for batch in train_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1)  # initialize (otherwise breaks in fast dev run) 
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

            cm_figure = plot_confusion_matrix(
                cm.cpu().numpy(), ['normal', 'tumor'])
            self.logger.experiment.add_figure(
                'confusion_matrix/train', cm_figure, global_step=self.global_step)

            # log F1 score
            self.log('train_metrics/F1_normal', 2 * prec_n * recall_n / (prec_n + recall_n))
            self.log('train_metrics/F1_tumor', 2 * prec_t * recall_t / (prec_t + recall_t))

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # forward
        y_hat = self.forward(x)

        # compute and log loss
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('val_loss', loss, on_epoch=True)

        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # log accuracy
        self.log('val_metrics/acc', metrics.classification.accuracy(preds,
                 y, num_classes=2), on_epoch=True)

        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)

        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}

    def validation_epoch_end(self, val_outputs):
        logits = torch.cat([batch['outputs'] for batch in val_outputs]).squeeze(-1)
        targets = torch.cat([batch['targets'] for batch in val_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1)  # initialize (otherwise breaks in fast dev run) 
        if targets.sum() > 0.:
            precision, recall, _ = metrics.precision_recall_curve(
                logits, targets)
            pr_auc = metrics.classification.auc(recall, precision)

            # plot the PR curve
            fig = plt.figure()
            tumor_ratio = len(targets[targets == 1.]) / len(targets)
            plt.plot([0, 1], [tumor_ratio, tumor_ratio],
                     linestyle='--', label='random')
            plt.plot(recall.cpu(), precision.cpu(),
                     marker='.', label='our model')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            self.logger.experiment.add_figure('prec-recall-curve', fig, global_step=self.global_step)
        self.log('val_metrics/PR_AUC', pr_auc, on_epoch=True)
        self.log('PR_AUC', pr_auc, on_epoch=True)  # logged separately for model checkpoint callback

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

        # compute area under the precision-recall curve
        if pr_auc.item() > self.hp_metric:
            self.logger.experiment.add_scalar(
                'hp_metric', pr_auc, global_step=0)
            self.hp_metric = pr_auc.item()

    def test_step(self, batch, batch_idx):
        x, y = batch 

        # forward pass
        y_hat = self.forward(x)

        # compute and log loss
        loss = F.binary_cross_entropy_with_logits(
            y_hat, y, pos_weight=self.weight.to(self.device))
        self.log('test_loss', loss, on_epoch=True)

        # log metrics
        logits = torch.sigmoid(y_hat.detach())
        preds = torch.round(logits)

        # log accuracy
        self.log('test_metrics/acc', metrics.classification.accuracy(preds, y, num_classes=2), on_epoch=True)

        # log tp, fp, tn, fn
        cm = metrics.confusion_matrix(preds, y, num_classes=2)

        return {'loss': loss, 'outputs': logits, 'targets': y, 'cm': cm}

    def test_epoch_end(self, test_outputs):
        logits = torch.cat([batch['outputs']for batch in test_outputs]).squeeze(-1)
        targets = torch.cat([batch['targets']or batch in test_outputs]).squeeze(-1)

        # compute AUC of precision-recalll-curve
        pr_auc = - torch.ones(1)  # initialize (otherwise breaks in fast dev run) ß
        if targets.sum() > 0.:
            precision, recall, _ = metrics.precision_recall_curve(
                logits, targets)
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

        cm_figure = plot_confusion_matrix(
            cm.cpu().numpy(), ['normal', 'tumor'])
        self.logger.experiment.add_figure(
            'confusion_matrix/test', cm_figure, global_step=self.global_step)

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
