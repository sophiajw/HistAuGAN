from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_utils import Center0Dataset, TestCenterDataset, MultipleCentersSeq, OneCenterLoad
from augmentations import basic_augmentations, color_augmentations, no_augmentations, gan_augmentations, geom_augmentations, normalization, color_augmentations_light
from model import Classifier


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--weighted', action='store_true',
                        help='model trains with weighted loss when flag is set')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Classifier.add_model_specific_args(parser)

    args = parser.parse_args()

    # ------------
    # configure type of augmentations
    # ------------
    augmentations = {
        None: no_augmentations,
        'no_augmentations': no_augmentations,
        'geom_augmentations': geom_augmentations,
        'basic_augmentations': basic_augmentations,
        'color_augmentations': color_augmentations,
        'color_augmentations_light': color_augmentations_light,
        'gan_augmentations': gan_augmentations,
    }

    name = args.name
    if name in augmentations.keys():
        aug = augmentations[name]
    else:
        aug = no_augmentations

    print(aug)

    gan_aug = False
    if args.name == 'gan_augmentations':
        args.batch_size = 8
        gan_aug = True

    print('gan_aug=', gan_aug)

    # evaluation over all five centers
    for center in [1, 2, 3, 4]:

        # ------------
        # data
        # ------------
        # adapt this part to your data loading strategy and classes
        print('load data')
        data_dir = '/storage/groups/haicu/datasets/2101_camelyon17/patches/'
        train_dataset = OneCenterLoad(data_dir, center, 'train', transform=aug)
        val_dataset = OneCenterLoad(
            data_dir, center, 'val', transform=no_augmentations)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, num_workers=6)

        # ------------
        # model
        # ------------
        model = Classifier(args.learning_rate, args.l2_reg, args.weighted,
                           gan_aug=gan_aug, transform=no_augmentations)

        # ------------
        # training
        # ------------
        logger = TensorBoardLogger('lightning_logs', name=name)
        print(logger.log_dir)

        early_stop_callback = EarlyStopping(
            monitor='val_metrics/PR_AUC',
            min_delta=0.,
            patience=20,
            verbose=False,
            mode='max'
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='PR_AUC',
            dirpath=logger.log_dir + '/checkpoints/',
            filename='Classifier-Center0-{epoch:02d}-{PR_AUC:.4f}',
            save_top_k=1,
            mode='max'
        )

        trainer = pl.Trainer.from_argparse_args(args)
        trainer.logger = logger
        trainer.callbacks = [checkpoint_callback, early_stop_callback]
        trainer.val_check_interval = 0.5

        trainer.fit(model, train_loader, val_loader)
        del train_dataset, val_dataset

        # ------------
        # testing
        # ------------
        model = Classifier.load_from_checkpoint(
            checkpoint_path=checkpoint_callback.best_model_path)
        print(checkpoint_callback.best_model_path)

        # test on all centers except for the training center
        test_centers = [[i, ] for i in range(5)]
        test_all = list(range(5))
        test_all.remove(center)
        test_centers.append(test_all)

        results = []
        for c in test_centers:
            print(f'results for dataset {c}')
            if c == [center, ]:
                test_dataset = OneCenterLoad(data_dir, center, 'val')
            else:
                test_dataset = MultipleCentersSeq(data_dir, c)
            test_loader = DataLoader(
                test_dataset, batch_size=128, num_workers=1)
            result = trainer.test(test_dataloaders=test_loader)
            results.append(result)

        # print final test results for each center except the training center
        print('center', center)
        print(test_centers)
        print('PR_AUC')
        pr_auc = [round(res[0]['PR_AUC'], 4) for res in results]
        print(pr_auc)
        print('F1_tumor')
        f1 = [round(res[0]['F1_tumor'], 4) for res in results]
        print(f1)


if __name__ == '__main__':
    main()
