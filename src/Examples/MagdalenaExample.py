import os
from functools import lru_cache

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import figure
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

from src.ActivationFunctions.ActivationFunction import Relu
from src.BatchFunctions.BatchFunction import MiniBatch
from src.Callbacks.Callback import GraphicCallback
from src.CostFunctions.CostFunction import *
from src.DataTools.LazyLoader import LazyLoader
from src.Examples.ExampleTemplate import ExampleTemplate
from src.Metrics.Metrics import MseMetric
from src.Networks.Layer.ClassicLayer import ClassicLayer, Xavier
from src.Optimizers.Optimizers import Adam


class MagdalenaExample(ExampleTemplate):

    def _get_excel(self, file):
        print(f"Getting '{file}'")
        return pd.read_excel(file, sheet_name=None, parse_dates=True)

    def _get_excel_pandas(self, files_folder, ROOT_FOLDER):
        pandas_files = dict()
        summary = ""

        i = 0
        for folder in sorted(files_folder):
            if not os.path.isdir(folder):
                continue

            summary += f"[{i}] {folder.replace(ROOT_FOLDER, '').strip('/')}\n"
            d = pandas_files.get(i, dict())
            pandas_files[i] = d
            i += 1

            j = 0
            for file in sorted(os.listdir(folder)):
                abs_file = folder + '/' + file
                if file.endswith('.xlsx') and not file.startswith('~$'):
                    dd = LazyLoader((lambda name: lambda: self._get_excel(name))(abs_file))
                    # dd = pd.read_excel(abs_file, sheet_name=None, parse_dates=True)
                    d[j] = dd

                    summary += f'\t[{j}] {file}\n'
                    j += 1
                else:
                    summary += f'\t[-] {file}\n'

        return pandas_files, summary

    def create_sequence(self, X, Y, sequence_length):
        return timeseries_dataset_from_array(
            X,
            Y,
            sequence_length=sequence_length,
            sequence_stride=1,
            sampling_rate=1,
            batch_size=1
        )

    def stack_all_sequences(self, sequence):
        all_inputs = []
        for batch in sequence:
            inputs = batch
            all_inputs.append(inputs.numpy())

        return np.vstack(all_inputs)

    def _get_split_data(self, all_flows):
        SEQUENCE_LENGTH = 120
        FUTURE_DAYS = 15

        train = all_flows.loc['1990-01-01':'2010-12-31']
        X_train, Y_train = train, train[train.columns[-2]]

        X_train = X_train.drop(all_flows.columns[-2], axis=1)

        val = all_flows.loc['2011-01-01':'2013-12-31']
        X_val, Y_val = val, val[val.columns[-2]]
        X_val = X_val.drop(X_val.columns[-2], axis=1)

        X_train, Y_train = X_train.to_numpy(), Y_train.to_numpy()

        X_val, Y_val = X_val.to_numpy(), Y_val.to_numpy()

        train_sequence = self.create_sequence(X_train, None, SEQUENCE_LENGTH)
        validation_sequence = self.create_sequence(X_val, None, SEQUENCE_LENGTH)

        train_sequence = self.stack_all_sequences(train_sequence)[:-FUTURE_DAYS]
        validation_sequence = self.stack_all_sequences(validation_sequence)[:-FUTURE_DAYS]

        train_sequence = train_sequence.reshape(train_sequence.shape[0], -1)
        validation_sequence = validation_sequence.reshape(validation_sequence.shape[0], -1)

        y_train_sequence = Y_train[SEQUENCE_LENGTH - 1:][FUTURE_DAYS:]
        y_validation_sequence = Y_val[SEQUENCE_LENGTH - 1:][FUTURE_DAYS:]

        return train_sequence, y_train_sequence, validation_sequence, y_validation_sequence

    @lru_cache()
    def get_data(self) -> np.array:
        BASE_DIR = '../../datasets/magdalena'
        INPUTS_FOLDER = BASE_DIR + '/Inputs'
        OUTPUTS_FOLDER = BASE_DIR + '/Outputs/Calibration dataset'
        TEST_OUTPUTS_FOLDER = BASE_DIR + '/Outputs'

        input_folders, output_folders = self.setup_folders(INPUTS_FOLDER, OUTPUTS_FOLDER)

        pandas_outputs, osummary = self._get_excel_pandas(output_folders, OUTPUTS_FOLDER)

        caudales_test = self.read_test_data(TEST_OUTPUTS_FOLDER)

        all_flows = self.join_flow_data(caudales_test, pandas_outputs)

        return self._get_split_data(all_flows)

    def join_flow_data(self, caudales_test, pandas_outputs):
        caudales = pandas_outputs[0][0].copy()
        caudales_sheets = list(caudales.keys())
        dataset = caudales[caudales_sheets[0]]
        dataset = dataset.set_index('Fecha')
        caudales_test.index.names = ['Fecha']
        dataset.columns = caudales_test.columns
        all_flows = pd.concat([dataset, caudales_test])
        all_flows = all_flows.fillna(0)
        all_flows = all_flows.rolling(window=7, min_periods=0).mean()
        return all_flows

    def read_test_data(self, TEST_OUTPUTS_FOLDER):
        with open(TEST_OUTPUTS_FOLDER + '/answers.csv', 'r') as csvfile:
            caudales_test_tmp = pd.read_csv(csvfile, index_col='Date', parse_dates=True)
            caudales_cols = caudales_test_tmp['Id'].unique()

            caudales_test = pd.DataFrame()
            for col in caudales_cols:
                caudales_test[col] = caudales_test_tmp[caudales_test_tmp['Id'] == col]['Value']

            for i in range(len(caudales_cols)):
                # i = -3
                print(caudales_cols[i], caudales_test[caudales_cols[i]][~caudales_test[caudales_cols[i]].isna()])
                # # print(caudales_test)
                # print(caudales_cols)

            del caudales_test_tmp
            del caudales_cols
        return caudales_test

    def setup_folders(self, INPUTS_FOLDER, OUTPUTS_FOLDER):
        input_folders = sorted(os.listdir(INPUTS_FOLDER))
        input_folders = [INPUTS_FOLDER + '/' + folder for folder in input_folders]
        print(">>> Input Folders <<<")
        print("\n".join([f"[{i}] {folder}" for i, folder in enumerate(input_folders)]))
        output_folders = sorted(os.listdir(OUTPUTS_FOLDER))
        output_folders = [OUTPUTS_FOLDER + '/' + folder for folder in output_folders]
        print(">>> Output Folders <<<")
        print("\n".join([f"[{i}] {folder}" for i, folder in enumerate(output_folders)]))
        return input_folders, output_folders

    def define_data(self):
        x_train, y_train, x_test, y_test = self.get_data()
        self.training_data = x_train
        self.expected_output = y_train

    def define_architecture(self):
        self.architecture = [
            ClassicLayer(8, Relu(), Xavier()),
            # TODO add BatchNormalization
            ClassicLayer(5, Relu(), Xavier()),
            ClassicLayer(4, Relu(), Xavier()),
            ClassicLayer(7, Relu(), Xavier()),
            ClassicLayer(6, Relu(), Xavier()),
            ClassicLayer(5, Relu(), Xavier()),
            ClassicLayer(10, Relu(), Xavier()),
            ClassicLayer(5, Relu(), Xavier()),
            ClassicLayer(1, Relu(), Xavier()),
        ]

        self.cost_function = MeanSquaredError()

    def define_training_hyperparameters(self):
        self.metrics = [MseMetric()]
        self.callbacks = [GraphicCallback()]
        self.learning_rate = 0.01
        self.iterations = 50
        self.batch_function = MiniBatch(self.training_data, self.expected_output, 256)
        self.optimizer = Adam()

    def run_tests(self):
        train_sequence, y_train_sequence, validation_sequence, y_validation_sequence = self.get_data()

        self.plot_serie(train_sequence, y_train_sequence, "training")

        self.plot_serie(validation_sequence, y_validation_sequence, "validation")

    def plot_serie(self, validation_sequence, y_validation_sequence, types):
        metric = MseMetric()
        val_predicted = self.neural_net.predict(validation_sequence)
        val_predicted = self.smooth_output(val_predicted)
        f = figure(num=None, figsize=(30, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(val_predicted)
        plt.plot(y_validation_sequence)
        plt.show()
        plt.close(f)
        print(f"{types} - MSE: {metric.calculate(val_predicted, y_validation_sequence):,}")

    def smooth_output(self, train_predicted):
        train_predicted = pd.DataFrame(train_predicted)
        train_predicted = train_predicted.rolling(window=10, min_periods=0).mean()
        train_predicted = train_predicted.to_numpy()
        return train_predicted


if __name__ == '__main__':
    # np.random.seed(0)
    xor_example = MagdalenaExample("magdalena_example")
    xor_example.run()
