import pandas as pd

df = pd.read_csv('/home/chris/Desktop/VLM/datasets/flickr30k/results.csv', sep='|', skipinitialspace=True)

df_select = df[["image_name", "comment"]]
df_select.columns = ["image", "caption"]
df_select.to_csv('/home/chris/Desktop/VLM/datasets/flickr30k/captions.csv', index=False)