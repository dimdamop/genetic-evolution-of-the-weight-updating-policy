import numpy as np

keep_last = 50
num_trials = 7

trials = {}

for key in "-prev", "":

    min_len = None
    curr_trials = []

    for i in range(num_trials):
        with open(f"{i + 1}{key}.out") as fd:
            trial = np.array([float(line.rsplit("=", maxsplit=1)[1].strip()) for line in fd])
        min_len = min(len(trial), min_len) if min_len else len(trial)
        curr_trials.append(trial)

    trials[key] = np.stack([trial[-min_len:] for trial in curr_trials])

    sample = trials[key][:, -keep_last:]

    print(key)
    for test in np.shape, np.mean, np.median, np.std, np.min, np.max:
        print(f"{test.__name__}: {test(sample)}")
