#!/usr/bin/env python3

import os
print(os.getcwd())

import argparse

from configparser import ConfigParser
from torch.utils.data import DataLoader
from sys import exc_info

from custom.dataset import DatasetBasler, DatasetDtd


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
    nn_config = config['CNN']
    data_config = config['Dataset']

    # TODO: choose solver dynamically based on config
    Solver = __import__("custom.models." + nn_config['ModelName'], fromlist=['models']).Solver
    solver = Solver(config)

    # account for 
    levels = nn_config.getint('LevelCount') + (nn_config.getint('AddLevels') if 'AddLevels' in nn_config else 0)


    # TODO: dynamically choose dataset loader based on config
    train_set = DatasetDtd(data_config['TrainData'], length=data_config.getint('TrainLength'),
                           levels=levels, input_size=data_config.getint('InputSize'))
    # test_set = Dataset(data_config['TestData'], nn_config.getint('UpscaleFactor'), length=data_config.getint('TestLength'))

    training_data_loader = DataLoader(dataset=train_set, batch_size=nn_config.getint('BatchSize'),
                                      shuffle=True, num_workers=0)
    # testing_data_loader = DataLoader(dataset=test_set, batch_size=nn_config.getint('BatchSize'), shuffle=True)

    # if 'Snapshot' in nn_config:
    #     solver.load_model(nn_config['Snapshot'])

    if 'Snapshot' in nn_config:
        solver.load_model(nn_config['Snapshot'])

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
