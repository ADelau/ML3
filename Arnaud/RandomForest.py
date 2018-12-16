import dataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
	x, y = dataset.make_dataset_1()

	estimator = RandomForestRegressor(n_estimators=600, max_depth = None, max_features = "auto", min_samples_split = 50, bootstrap = True)
	error = 0

	error = cross_val_score(estimator, x, y, cv = 5, scoring = "neg_mean_squared_error")

	print("mean square error = {}".format(score))