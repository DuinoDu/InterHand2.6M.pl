import os
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
import interhand
from interhand.data import DataModule
from interhand.model import Model


def main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_data_specific_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = DataModule(
        batch_size=args.batch_size,
        annot_subset=args.annot_subset)

    # ------------
    # model
    # ------------
    model = Model(args.lr)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
