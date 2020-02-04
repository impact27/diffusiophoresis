# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 15:29:14 2018

@author: Quentin Peter

Parse comsol txt file and save numpy file.

This script is distributed under CC BY-NC-SA version 4.0. You should have
recieved a copy of the licence along with it. If not, see
https://creativecommons.org/licenses/by-nc-sa/4.0/.
"""
import numpy as np
import os


class ComsolParser():
    """Parse comsol txt file and save numpy file."""

    def __init__(self):
        super().__init__()
        self._ndim = 0
        self.shape = None
        self.grid = []
        self.datasets = {}
        self.functions = {
            "Model:": self.ignore,
            "Version:": self.ignore,
            "Date:": self.ignore,
            "Dimension:": self.load_dim,
            "Nodes:": self.ignore,
            "Expressions:": self.ignore,
            "Description:": self.ignore,
            "Length": self.ignore,
            "Grid": self.load_grid,
            "Data": self.load_data,
        }

    def parse(self, fn):
        with open(fn) as f:
            self.file = f
            self.parse_next()
        return self.datasets

    def parse_next(self):
        line = self.load_macro()
        if line is None:
            return
        self.functions[line[0]](line[1:])
        self.parse_next()

    def ignore(self, args):
        return

    def load_dim(self, args):
        self._ndim = int(args[0])

    def load_grid(self, args):
        for i in range(self._ndim):
            self.grid.append(self.load_1darray())
        self.grid = self.grid
        self.shape = tuple(len(axis) for axis in self.grid)

    def load_data(self, args):

        line = self.load_macro()
        assert len(line) == 4
        var_name = line[0]

        if var_name not in self.datasets:
            self.datasets[var_name] = {
                "times": [],
                "data": np.zeros((0, *self.shape))}

        dataset = self.datasets[var_name]

        var_units = line[1]
        assert var_units[0] == '(' and var_units[-1] == ')'
        dataset["units"] = var_units[1:-1]

        var_time = line[3]
        assert var_time[:2] == 't='
        var_time = float(var_time[2:])
        dataset["times"].append(var_time)
        shape = self.shape[::-1]
        if len(shape) > 2:
            shape = (np.product(shape[:-1]), shape[-1])
        if len(shape) > 1:
            data = np.zeros(shape)
            for idx in range(shape[0]):
                data[idx] = self.load_1darray()
            data.shape = self.shape[::-1]
            data = np.moveaxis(data,
                               np.arange(self._ndim),
                               np.arange(self._ndim)[::-1])
        else:
            data = self.load_1darray()
        dataset["data"] = np.append(dataset["data"], data[np.newaxis], axis=0)

    def load_1darray(self):
        line = self.file.readline()
        return np.fromstring(line, sep=' ')

    def load_macro(self):
        line = self.file.readline().split()
        if line == []:
            return
        assert line[0] == "%"
        return line[1:]


def comsol_parse(fn):
    parser = ComsolParser()
    parser.parse(fn)
    return parser.grid, parser.datasets


def comsol_to_python(fn, name, out_folder, mean_z=True):
    grid, dataset = comsol_parse(fn)
    times = None
    for data_name in dataset:
        # unit = dataset[data_name]['units']
        data = dataset[data_name]['data']
        times_tmp = dataset[data_name]['times']
        assert times is None or np.all(times == times_tmp)
        times = times_tmp
        fn = f"{name}_{data_name}"
        # If remove z
        if mean_z and data.ndim == 4:
            data = np.mean(data, axis=-1)
        np.save(os.path.join(out_folder, fn), data)

    axes = {'t': times}
    if mean_z:
        spatial_axes = ['x', 'y']
    else:
        spatial_axes = ['x', 'y', 'z']
    for axis_name, axis in zip(spatial_axes, grid):
        axes[axis_name] = axis
    np.savez(os.path.join(out_folder, f"{name}_axes"), **axes)


if __name__ == '__main__':
    out_folder = '../Data/Simulations_Processed'

    # fn = "../Data/Simulations/1d-2.txt"
    # comsol_to_python(fn, 'k-device_gamma-1p5e-10_1d', out_folder)

    # fn = "../Data/Simulations/2d-2.txt"
    # comsol_to_python(fn, 'k-device_gamma-1p5e-10_2d', out_folder)

    # fn = "../Data/Simulations/3d-2.txt"
    # comsol_to_python(fn, 'k-device_gamma-1p5e-10_3d', out_folder)

    # for name in ['-1e-9', '-1e-10', '-1e-11']:
    #     fn = f"../Data/Simulations/gamma_{name}.txt"
    #     comsol_to_python(fn, f'k-device_gamma{name}', out_folder)

    # for name in ['C', 'logC']:
    #     fn = f"../Data/Simulations/{name}.txt"
    #     comsol_to_python(fn, f'k-device_gamma-1.5e-10_{name}', out_folder)

    # for name in ['10um', '20um', '50um', '100um']:
    #     fn = f"../Data/Simulations/ramp_{name}.txt"
    #     comsol_to_python(fn, f'k-device_gamma-1.5e-10_ramp_{name}', out_folder)

    # for i, name in enumerate(
    #         ['0p1uM', '1uM', '10uM', '100uM', '1mM', '10mM']):
    #     fn = f"../Data/Simulations/Cs_{i+1}.txt"
    #     comsol_to_python(fn, f'k-device_gamma-1p5e-10_CsOut_{name}', out_folder)

    # for i, name in enumerate(
    #         np.fromstring('-2e-11 -5e-11 -1e-10 -2e-10 -5e-10', sep=' ')):
    #     fn = f"../Data/Simulations/gamma_{i+1}.txt"
    #     comsol_to_python(fn, f'k-device_gamma{name}', out_folder)

    # for i, name in enumerate(
    #         np.fromstring('1e-10 2e-10 5e-10 1e-9 2e-9 5e-9 1e-8', sep=' ')):
    #     fn = f"../Data/Simulations/Cs_{i+1}.txt"
    #     comsol_to_python(fn, f'k-device_Ds_{name}', out_folder)

    for i, name in enumerate(
            np.fromstring('5e-12 1e-11 2e-11 5e-11 1e-10 2e-10 5e-10',
                          sep=' ')):
        fn = f"../Data/Simulations/Dp_{i+1}.txt"
        comsol_to_python(fn, f'k-device_Dp_{name}', out_folder)
