import os
from sklearn.model_selection import train_test_split
import json

GeoExplain_path = '/scratch/user/xxxx/GuessWhere'

GeoExplain_samples_list = os.listdir(GeoExplain_path)
train, test = train_test_split(GeoExplain_samples_list, test_size=0.2, random_state=1)

GeoExplain_samples_name_list = {'Train': train, 'Test': test}

print("Train:", len(GeoExplain_samples_name_list['Train']))
print("Test:", len(GeoExplain_samples_name_list['Test']))

GeoExplain_samples_name_list_path = '/scratch/user/xxxx/GeoExplain_train_test_split.json'
with open(GeoExplain_samples_name_list_path, 'w') as fp:
    json.dump(GeoExplain_samples_name_list, fp)