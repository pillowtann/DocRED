import csv
import json
import pandas as pd
import re

# point to log path for processing summary table
log_path = "./log/CNN3"
# refer to Config or EviConfig "self.period"; Default is 50
period = 50

with open(log_path, 'r') as fp:
    data = fp.readlines()

status = 0
max_len = len(data)
train_df = [[]]
test_df = [[]]
non_decimal = re.compile(r'[^\d.]+')

for idx, row in enumerate(data):
    if 'train' in row:
        # if train info
        keep_info = [non_decimal.sub('', info.strip()) for info in row.split(' | ')]
        train_df.append(keep_info)
    elif '-----' in row:
        # skip row
        pass
    elif ('ALL' in row) or (status>0):
        # if start of test info
        status += 1
        row = row.lower().replace("f1", "")
        
        if status==1:
            # first row
            keep_info = [non_decimal.sub('', info.strip()) for info in row.split(' | ')]
        elif status==2:
            # second row
            splitted = row.split(' | ')
            # second item dirty
            middle_items = splitted[1].split('test_result')
            keep_info_2 = [splitted[0], middle_items[0], middle_items[1], splitted[2]]
            keep_info_2 = [non_decimal.sub('', info.strip()) for info in keep_info_2]
            keep_info.extend(keep_info_2)
        else:
            # third row
            keep_info_2 = [non_decimal.sub('', info.strip()) for info in row.split(' | ')]
            keep_info.extend(keep_info_2)
            # end of test info
            # reset status
            status = 0
            test_df.append(keep_info)

train_df = pd.DataFrame(train_df, columns=['epoch', 'step', 'ms/b', 'loss', 'NA acc', 'not NA acc', 'tot acc']).drop([0])
test_df = pd.DataFrame(test_df, columns=['theta', 'f1', 'AUC', 'ign_max_f1', 'ign_theta', 'ign_f1', 'ign_AUC', 'epoch', 'time']).drop([0])

summary_df = pd.DataFrame(
    [[sum([float(i) for i in train_df['ms/b']])*0.001*period, 
     float(test_df.iloc[-1]['ign_max_f1']), 
     float(test_df.iloc[-1]['ign_AUC']), 
     float(test_df.iloc[-1]['f1']),
     float(test_df.iloc[-1]['AUC'])]],
    columns=['Train Time', 'Ign F1', 'Ign AUC', 'F1', 'AUC'])
print(summary_df)