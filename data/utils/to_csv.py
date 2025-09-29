import pandas as pd
import numpy as np
import os
import re
from pathlib import Path


def generate_df(dir):
    rows = []
    video_names = [x.stem for x in Path(dir).glob("*.mp4")]
    for name in video_names:
        row = {"file_name":name, "Stopped":0, "Thinking":0}
        rows.append(row)
        
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate_df("/n/work1/muyun/Dataset/datastes_v3/videos")
    print(df)
    df.to_csv("./annotation.csv")