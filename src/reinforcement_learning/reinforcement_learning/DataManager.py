import os
import uuid
import pandas as pd

class DataManager:

    def __init__(self, name='default-name', x_label='x_values', x_data=None, y_label='y_values', y_data=None, checkpoint_freq=100) -> None:
        
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []

        self.x_data = x_data
        self.y_data = y_data

        self.x_label = x_label
        self.y_label = y_label
        
        self.plot_num = 0
        self.checkpoint_freq = checkpoint_freq

        self.name = name

    def post(self, y_point):
        self.x_data.append(len(self.x_data))
        self.y_data.append(y_point)

        if self.plot_num % self.checkpoint_freq == 0:
            self.save_csv(f'{self.name}-checkpoint.csv')

    def save_csv(self, file_name=str(uuid.uuid4().hex)):
        dir_exists = os.path.exists("data")

        if not dir_exists:
            os.makedirs("data")

        data_dict = {self.x_label: self.x_data, self.y_label: self.y_data}
        df = pd.DataFrame(data=data_dict)

        df.to_csv(f"data/{file_name}", index=False)
        