from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings

class GridSearcher:
    def __init__(self, cfg_list, debug=False, parallel=True):
        self.cfg_list = cfg_list
        self.debug = debug
        self.parallel = parallel
        self.scores = []

    @property
    def best_model(self):
        cls, cfg = self.scores[0][0]
        return cls(cfg)

    # score a model, return None on failure
    def score_model(self, cls, cfg, data, n_test):
        # show all warnings and fail on exception if debugging
        model = cls(cfg)
        if self.debug:
            result = model.walk_forward_validation(data, n_test)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    filterwarnings("ignore")
                    result = model.walk_forward_validation(data, n_test)
            except:
                result = None
        return (cls, cfg), result

    # grid search configs
    def search(self, data, n_test):
        if self.parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(self.score_model)(cls, cfg, data, n_test)
                    for cls, cfg in self.cfg_list)
            scores = executor(tasks)
        else:
            scores = [self.score_model(cls, cfg, data, n_test) for cls, cfg in self.cfg_list]
        # remove empty results
        self.scores = [r for r in scores if r[-1] != None]
        # sort configs by error, asc
        self.scores.sort(key=lambda tup: tup[-1])
        print('\nTop three:')
        for (cls, cfg), rmse in self.scores[:3]:
            print(f'{cls}({cfg}): {rmse}')
