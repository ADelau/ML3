# ! /usr/bin/env python
# -*- coding: latin_1 -*-

import pandas as pd
import numpy as np

def clean(series):
	for i in range(len(series)):
		if (not np.isnan(series[i])) and (series[i] > 1000):
			series[i] /= 1000

	return series


data = pd.read_csv("data_user_2.csv", delimiter=",", encoding = "latin_1")
data.info()
data["longitude"] = clean(data["longitude"])
data["latitude"] = clean(data["latitude"])
data.to_csv("data_user_3.csv", sep=",", encoding = "latin_1", index = False)