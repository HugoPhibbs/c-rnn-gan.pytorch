from utils.data_utils import DataUtils

data_df = DataUtils.load_data_by_name("3D_Play_House", r"C:\Users\hugop\Documents\Uni\Graphics\COMPSCI715\datasets\parquet")

data_formated_df = DataUtils.format_dataset(data_df)


print(data_formated_df.head())
print(data_formated_df.columns)
