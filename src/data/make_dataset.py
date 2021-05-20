import pandas as pd
import os
def save_file(df):
    save_path='../../data/modified/'
    df.to_csv(os.path.join(save_path+'data_to_work_on.csv'),index=False)
    return os.path.join(save_path+'data_to_work_on.csv')


if __name__=='__main__':
    #import the raw data
    df=pd.read_csv('../../data/raw/train.csv')
    # Drop the unwanted columns
    df.drop(columns=['id'],inplace=True)

    #save the file
    path=save_file(df)
    print(f'Data prepared for further steps and and saved to {path}')
    