from utils.data_utils import DataUtils
import numpy as np

class VRDataLoader(object):

    def __init__(self):
        self.data_df = None
        self.data_formated_df = None
        self.num_samples = -1

    def read_data(self, game_name, data_folder_path ='/data/ysun209/VR.net/parquet/'):
        self.data_df = DataUtils.load_data_by_name(game_name, data_folder_path)
        self.data_formated_df = DataUtils.format_dataset(self.data_df)
        self.num_samples = len(self.data_formated_df)

        pass # TODO: implement

    def get_random_batch(self, batch_size):
        """
        Randomly sample a batch of data from the dataset

        Args:
            batch_size: int, the number of time point in a batch
        """
        batch_start_idx = np.random.randint(0, self.num_samples - batch_size)

        batch_df = self.data_formated_df.iloc[batch_start_idx:batch_start_idx + batch_size]

        batch_np = batch_df.to_numpy()

        return batch_np

    def num_features(self):
        pass



