from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_utils import Center0Dataset, TestCenterDataset, calculate_stats, ImbalancedDatasetSampler, MultipleCentersSeq
from augmentations import basic_augmentations, color_augmentations, no_augmentations, mdmm_augmentations
from model import Classifier


def main():
#     pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
#     parser.add_argument('--max_epochs', default=10, type=int)

    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--weighted', action='store_true', help='model trains with weighted loss when flag is set')
    
#     parser.add_argument('--learning_rate', type=float, default=0.0001)
#     parser.add_argument('--l2_reg', type=float, default=1e-06)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)
#     parser.add_argument('--max_epochs', default=10, type=int)

    args = parser.parse_args()

    # ------------
    # data
    # ------------
    data_file = '/home/haicu/sophia.wagner/datasets/2101_camelyon17/center0_level2.hdf5'
    print('load data')
    train_dataset = Center0Dataset(data_file, 'train', transform=None)
    val_dataset = Center0Dataset(data_file, 'val', transform=no_augmentations)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=6)

    # one basic run:
    # ------------
    # model
    # ------------
    model = Classifier(args.learning_rate, args.l2_reg, args.weighted, mdmm_aug=True)

    # ------------
    # training
    # ------------
    logger = TensorBoardLogger('lightning_logs', name=args.name)
    print(logger.log_dir)

    early_stop_callback = EarlyStopping(
       monitor='val_metrics/PR_AUC',
       min_delta=0.,
       patience=5,
       verbose=False,
       mode='max'
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='PR_AUC',
        dirpath=logger.log_dir + '/checkpoints/',
        filename='Classifier-Center0-{epoch:02d}-{PR_AUC:.2f}',
        save_top_k=3,
        mode='max'
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.logger = logger
    trainer.callbacks = [checkpoint_callback, early_stop_callback]
    trainer.val_check_interval = 0.5
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    model = Classifier.load_from_checkpoint(
        checkpoint_path=checkpoint_callback.best_model_path,
    )
    
    del train_dataset, val_dataset
    test_dataset = MultipleCentersSeq('/home/haicu/sophia.wagner/datasets/2101_camelyon17/', [2, ], transform=no_augmentations)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=1)
    result = trainer.test(model, test_dataloaders=test_loader)
    
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