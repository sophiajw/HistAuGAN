from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import Stain_Normalization.stainNorm_Reinhard as stainNorm_Reinhard
import Stain_Normalization.stainNorm_Vahadane as stainNorm_Vahadane

from data_utils import Center0Dataset, TestCenterDataset, calculate_stats, ImbalancedDatasetSampler, MultipleCentersSeq, BalancedBatchSampler, OneCenterLoad
from augmentations import basic_augmentations, color_augmentations, no_augmentations, mdmm_augmentations, geom_augmentations, normalization
from model import Classifier


def main():
#     pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--weighted', action='store_true', help='model trains with weighted loss when flag is set')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)

    args = parser.parse_args()

    # ––––––––––––––––––––––––– configure
    center = 0
    mdmm_aug = False
#     bs = args.batch_size
#     models_to_test = [
#         ['no_augmentations', '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/no_augmentations/version_3/checkpoints/Classifier-Center0-epoch=05-PR_AUC=0.93.ckpt'],
#         ['geom_augmentations', '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/geom_augmentations/version_1/checkpoints/Classifier-Center0-epoch=08-PR_AUC=0.96-v0.ckpt'],
#         ['basic_augmentations', '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/basic_augmentations/version_3/checkpoints/Classifier-Center0-epoch=06-PR_AUC=0.96-v0.ckpt'],
#         ['color_augmentations', '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/color_augmentations/version_1/checkpoints/Classifier-Center0-epoch=26-PR_AUC=0.95.ckpt'],
#         ['mdmm_augmentations', '/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/mdmm_augmentations/version_17/checkpoints/Classifier-Center0-epoch=08-PR_AUC=0.96.ckpt']
#     ]
    data_dir = '/home/haicu/sophia.wagner/datasets/2101_camelyon17/'
    dataset = OneCenterLoad(data_dir, 0, 'train', transform=None)
    template = dataset.data['patches'][17455]
    
    reinhard = stainNorm_Reinhard.Normalizer()
    reinhard.fit(template.permute(1, 2, 0).numpy())
    
#     vahadane = stainNorm_Vahadane.Normalizer()
#     vahadane.fit(template.permute(1, 2, 0).numpy())

    
    models_to_test = [
        ['/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/geom_augmentations/version_14/checkpoints/Classifier-Center0-epoch=10-PR_AUC=0.95.ckpt', reinhard],
#         ['/home/haicu/sophia.wagner/projects/stain_color/stain_aug/lightning_logs/basic_augmentations/version_3/checkpoints/Classifier-Center0-epoch=06-PR_AUC=0.96-v0.ckpt', vahadane]
    ]
    
    for checkpoint, norm_method in models_to_test:
        logger = TensorBoardLogger('lightning_logs', name=args.name)
        print(logger.log_dir)

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.logger = logger        
        # ------------
        # testing
        # ------------
        model = Classifier.load_from_checkpoint(checkpoint_path=checkpoint)

        test_centers = [[i,] for i in range(5)]
#         test_centers.remove([center,])
        test_all = list(range(5))
        test_all.remove(center)
        test_centers.append(test_all)

        results = []
        for c in test_centers:
            print(f'results for dataset {c}')
            if c == [center, ]:
                test_dataset = OneCenterLoad('/home/haicu/sophia.wagner/datasets/2101_camelyon17/', center, 'val', norm=norm_method)
            else:
                test_dataset = MultipleCentersSeq('/home/haicu/sophia.wagner/datasets/2101_camelyon17/', c, norm=norm_method)
            test_loader = DataLoader(test_dataset, batch_size=128, num_workers=1)
            result = trainer.test(model, test_dataloaders=test_loader)
            results.append(result)

        print(test_centers)
        print('PR_AUC')
        pr_auc = [round(res[0]['PR_AUC'], 4) for res in results]
        print(pr_auc)
        print('F1_tumor')
        f1 = [round(res[0]['F1_tumor'], 4) for res in results]
        print(f1)

    # tune over multiple parameter settings
#     count = 0
# #     for reg in [1e-4, 1e-5, 1e-6]:
#     for d in [0.2, 0.5, 0.]:
#         for w in [True, False]:
# #         for lr in [1e-3, 1e-4, 1e-5]:
#             model = Classifier(args.learning_rate, args.l2_reg, weighted=w, dropout=d)

#             logger = TensorBoardLogger('lightning_logs', name=args.name)
#             print(logger.log_dir)

#             trainer = pl.Trainer.from_argparse_args(args)
#             trainer.logger = logger
#             trainer.log_every_n_steps = 10
#             trainer.val_check_interval = 0.5
#             trainer.fit(model, train_loader, val_loader)

#             count +=1
#             print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')
#             print(f'starting run {count}/6')
#             print('–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––')


if __name__ == '__main__':
    main()