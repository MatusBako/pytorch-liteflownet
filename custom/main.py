#!/usr/bin/env python3


import argparse
import torch

from configparser import ConfigParser
from torch.utils.data import DataLoader
from sys import exc_info

from custom.dataset import Dataset
from custom.train import Solver


def get_config() -> ConfigParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file of net.')
    args = parser.parse_args()

    config = ConfigParser()
    config.optionxform = str
    config.read(args.config)
    return config


def main():
    config = get_config()
    nn_config = config['CNN'] if 'CNN' in config else config['GAN']
    data_config = config['Dataset']

    # dynamically import Solver
    solver = Solver(config)

    # TODO: dynamically choose dataset loader based on config

    # torch.multiprocessing.set_start_method('spawn')
    train_set = Dataset(data_config['TrainData'], length=data_config.getint('TrainLength'))
    # test_set = Dataset(data_config['TestData'], nn_config.getint('UpscaleFactor'), length=data_config.getint('TestLength'))

    training_data_loader = DataLoader(dataset=train_set, batch_size=nn_config.getint('BatchSize'),
                                      shuffle=True, num_workers=3)
    # testing_data_loader = DataLoader(dataset=test_set, batch_size=nn_config.getint('BatchSize'), shuffle=True)

    # if 'Snapshot' in nn_config:
    #     solver.load_model(nn_config['Snapshot'])

    try:
        solver.train(training_data_loader)
    except KeyboardInterrupt as e:
        # solver.save_model()
        if solver.logger:
            solver.logger.finish()

        raise e
    except Exception:
        if solver.logger:
            solver.logger.finish()

        ex_type, ex_inst, ex_tb = exc_info()
        raise ex_type.with_traceback(ex_inst, ex_tb)
    finally:
        print("Iterations:", solver.iteration)


if __name__ == "__main__":
    main()
