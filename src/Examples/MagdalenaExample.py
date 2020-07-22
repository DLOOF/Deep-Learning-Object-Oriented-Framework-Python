import os
import pandas as pd
from functools import lru_cache

from src.ActivationFunctions.ActivationFunction import Sigmoid, TanH, Relu
from src.BatchFunctions.BatchFunction import MiniBatch
from src.Callbacks.Callback import GraphicCallback
from src.CostFunctions.CostFunction import *
from src.DataTools.LazyLoader import LazyLoader
from src.Examples.ExampleTemplate import ExampleTemplate
from src.Metrics.Metrics import MseMetric
from src.Networks.Layer.ClassicLayer import ClassicLayer, Xavier
from src.Networks.Layer.Convolution.Convolution2DLayer import Convolution2DLayer
from src.Networks.Layer.Convolution.FlattenLayer import FlattenLayer
from src.Networks.Layer.Convolution.MaxPoolingLayer import MaxPoolingLayer
from src.Networks.Layer.Convolution.UnFlattenLayer import UnFlattenLayer
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

    def _get_split_data(self, all_flows):
        train = all_flows.loc['1990-01-01':'2010-12-31']
        X_train, Y_train = train, train[train.columns[-2]]

        X_train = X_train.drop(all_flows.columns[-2], axis=1)

        val = all_flows.loc['2011-01-01':'2013-12-31']
        X_val, Y_val = val, val[val.columns[-2]]
        X_val = X_val.drop(X_val.columns[-2], axis=1)

        X_train, Y_train = X_train.to_numpy(), Y_train.to_numpy()

        X_val, Y_val = X_val.to_numpy(), Y_val.to_numpy()

        return X_train, Y_train, X_val, Y_val

    @lru_cache()
    def get_data(self) -> np.array:
        BASE_DIR = '../../datasets/magdalena'
        INPUTS_FOLDER = BASE_DIR + '/Inputs'
        OUTPUTS_FOLDER = BASE_DIR + '/Outputs/Calibration dataset'
        TEST_OUTPUTS_FOLDER = BASE_DIR + '/Outputs'

        INITIAL_DATE = '1990-01-01'

        input_folders = sorted(os.listdir(INPUTS_FOLDER))
        input_folders = [INPUTS_FOLDER + '/' + folder for folder in input_folders]
        print(">>> Input Folders <<<")
        print("\n".join([f"[{i}] {folder}" for i, folder in enumerate(input_folders)]))

        output_folders = sorted(os.listdir(OUTPUTS_FOLDER))
        output_folders = [OUTPUTS_FOLDER + '/' + folder for folder in output_folders]
        print(">>> Output Folders <<<")
        print("\n".join([f"[{i}] {folder}" for i, folder in enumerate(output_folders)]))

        pandas_inputs, isummary = self._get_excel_pandas(input_folders, INPUTS_FOLDER)

        print(">>> Inputs <<<")
        print(isummary)

        pandas_outputs, osummary = self._get_excel_pandas(output_folders, OUTPUTS_FOLDER)

        print(">>> OUTPUTS <<<")
        print(osummary)

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

        caudales_test.describe()

        caudales = pandas_outputs[0][0].copy()
        caudales_sheets = list(caudales.keys())
        dataset = caudales[caudales_sheets[0]]
        dataset = dataset.set_index('Fecha')

        caudales_test.index.names = ['Fecha']

        dataset.columns = caudales_test.columns

        all_flows = pd.concat([dataset, caudales_test])

        all_flows = all_flows.fillna(0)

        all_flows = all_flows.rolling(window=7, min_periods=0).mean()

        return self._get_split_data(all_flows)

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
        self.iterations = 100
        self.batch_function = MiniBatch(self.training_data, self.expected_output, 256)
        self.optimizer = Adam()

    def run_tests(self):
        _, _, x_test, y_test = self.get_data()
        import random
        errors = []
        for x, y in random.sample(list(zip(x_test, y_test)), k=2000):
            # print(x, y, sep="\t")
            y_predicted = self.neural_net.predict(x)
            d = y - y_predicted
            error = np.linalg.norm(d, 2)
            errors.append(error)

        print("Mean error %.3f" % np.mean(np.asarray(errors)))


if __name__ == '__main__':
    np.random.seed(0)
    xor_example = MagdalenaExample("magdalena_example")
    xor_example.run()
