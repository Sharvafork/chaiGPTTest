# import pandas as pd

# mydataset = {
#   'cars': ["BMW", "Volvo", "Ford"],
#   'passings': [3, 7, 2]
# }

# myvar = pd.DataFrame(mydataset)

# print(myvar)

import pandas as pd
df=pd.read_csv("cluster_0.csv")
dc=pd.read_csv("cluster_2.csv")
print(df.head())
print(dc.head())
# print(dc.corr)
print(pd.options.display.max_rows)
