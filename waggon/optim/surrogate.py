import gc
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib import  ticker
from joblib import Parallel, delayed

from .base import Optimiser
from .utils import create_dir


class SurrogateOptimiser(Optimiser):
    '''
    Surrogate based black-box optimiser.

    Parameter
    ----------
    func : Callable, waggon.functions.Function
        Black-box function to be optimised.
    
    surr : waggon.surrogates.Surrogate or waggon.surrogate.GenSurrogate
        Surrogate model for the black-box function.
    
    acqf : waggon.acquisitions.Acquisition
        Acquisition function defining the optimisation strategy.
    
    max_iter : int, default = 100
        Maximum number of optimisation loop iterations.
    
    eps : float, default = 1e-1
        Epsilon-solution criterion value.
    
    error_type : {'x', 'f'}, default = 'x'
        Optimisation error type - either in argument, 'x', or function, 'f'.
    
    num_opt : bool, default = False
        Whether the acquisition function is optimised numerically or not. If False,
        the search of the next best parameters is done via direct search.
    
    eq_cons : dict, default = None
        Equality-type constraints for numerical optimisation.
    
    ineq_cons : dict, default = None
        Ineqaulity-type constraints for numerical optimisation.

    fix_candidates : bool, default = False if num_opt else True
        Whether the candidate points should be fixed or not.
    
    n_candidates_ : int, default = 1 if num_opt else 101**2
        Number of candidates points.
    
    olhs : bool, default = True
        Whether orthogonal Latin hypercube sampling (LHS) is used for choosing candidate points.
        If False, simple LHS is used.

    lhs_seed : int, default = None
        Controls the randomness of candidates sampled via (orthogonal) LHS.
    
    verbose : int, default = 1
        Controls verbosity when fitting and predicting. By default only a progress bar over the
        optimisation loop is displayed.
    
    save_results : bool, default = True
        Whether results are saved or not.
    '''
    def __init__(self, func, surr, acqf, **kwargs):
        kwargs['func'] = func
        super(SurrogateOptimiser, self).__init__(**kwargs)
        
        self.func               = func
        self.surr               = surr
        self.acqf               = acqf
        
        self.n_candidates       = kwargs['n_candidates'] if 'n_candidates' in kwargs else 10201
        self.num_opt_start      = kwargs['num_opt_start'] if 'num_opt_start' in kwargs else 'grid'
        self.num_opt_disp       = kwargs['num_opt_disp'] if 'num_opt_disp' in kwargs else False
        self.num_opt_tol        = kwargs['num_opt_tol'] if 'num_opt_tol' in kwargs else 1e-8
        self.num_opt_candidates = kwargs['num_opt_candidates'] if 'num_opt_candidates' in kwargs else 128

        self.eq_cons            = kwargs['eq_cons'] if 'eq_cons' in kwargs else None
        self.ineq_cons          = kwargs['ineq_cons'] if 'ineq_cons' in kwargs else None

        self.jitter             = kwargs['jitter'] if 'jitter' in kwargs else 1e0
        self.parallel           = kwargs['parallel'] if 'parallel' in kwargs else 0
        
        if type(self.surr) == list:
            for s in self.surr:
                s.verbose = self.verbose
        else:
            self.surr.verbose   = self.verbose
    
    
    def _scalar_acqf(self, x):
        v = self.acqf(np.asarray(x, dtype=float))
        try:
            return float(v)
        except Exception:
            if hasattr(v, "item"):
                return float(v.item())
            return float(np.asarray(v).reshape(()))

    def _reflect_into_bounds(self, x):
        x = np.asarray(x, dtype=float).copy()
        bounds = np.asarray(self.func.domain, dtype=float)
        for i, (lo, hi) in enumerate(bounds):
            span = hi - lo
            if span <= 0:
                x[i] = lo
                continue
            t = (x[i] - lo) % (2*span)
            x[i] = lo + (t if t <= span else 2*span - t)
        return x

    def metropolis(self, x0, n_samples=10_000, prop_scale=0.1, burn=1_000, seed=None):
        """
        Random-walk Metropolis для МИНИМИЗАЦИИ self.acqf.
        Возвращает (samples, acc_rate, best_x, best_val)
        """
        if seed is not None:
            random.seed(seed)

        x = np.array(x0, dtype=float).reshape(-1)
        d = x.shape[0]
        # нормализуем prop_scale
        ps = np.full(d, float(prop_scale), dtype=float) if np.isscalar(prop_scale) \
            else np.asarray(prop_scale, dtype=float).reshape(-1)

        samples, accepts = [], 0
        total = int(n_samples) + int(burn)

        def energy(z):  # минимизируем acqf → энергия = acqf
            return self._scalar_acqf(z)

        e_cur = energy(x)
        best_x, best_val = x.copy(), e_cur

        for it in range(total):
            x_prop = x + np.array([random.gauss(0.0, ps[i]) for i in range(d)], dtype=float)
            x_prop = self._reflect_into_bounds(x_prop)

            lp_cur  = -e_cur
            e_prop  = energy(x_prop)
            lp_prop = -e_prop

            if math.log(random.random()) < (lp_prop - lp_cur):
                x, e_cur = x_prop, e_prop
                accepts += 1
                if e_cur < best_val:
                    best_x, best_val = x.copy(), e_cur

            if it >= burn:
                samples.append(x.copy())

        acc_rate = accepts / max(1, total)
        samples = np.array(samples) if samples else np.empty((0, d))
        return samples, acc_rate, best_x, best_val


    def numerical_search(self, x0=None):
        '''
        Numerical optimisation of the acquisition function.

        Returns
        -------
        best_x : np.array of shape (func.dim,)
            Predicted optimum of the acquisition function.
        '''
        
        if (x0 is None) and (self.num_opt_start != 'grid'):
            candidates = self.create_candidates()
        elif self.num_opt_start == 'random':
            candidates = np.array(self.num_opt_candidates * [x0])
            candidates += np.random.normal(0, self.eps, candidates.shape)
        elif self.num_opt_start == 'grid':
            inter_conds = self.create_candidates(N=self.n_candidates)
            if self.num_opt_candidates < self.n_candidates:
                ei = self.acqf(inter_conds)
                try:
                    ids = np.argsort(ei, axis=0)[:self.num_opt_candidates].reshape(-1, 1)
                    candidates = np.take_along_axis(inter_conds, ids, axis=0)
                except ValueError:
                    ei = ei.squeeze()
                    ids = np.argsort(ei, axis=0)[:self.num_opt_candidates].reshape(-1, 1)
                    candidates = np.take_along_axis(inter_conds, ids, axis=0)
            else:
                candidates = inter_conds
        
        n_samples  = getattr(self, "mcmc_n_samples", 10_000)
        burn       = getattr(self, "mcmc_burn",      1_000)
        prop_scale = getattr(self, "mcmc_prop_scale", 0.1)
        seed       = getattr(self, "mcmc_seed",      None)

        if self.parallel in [0, 1]:
            best_x, best_val = None, np.inf
            for i, start in enumerate(candidates):
                # Metropolis минимизирует self.acqf
                _, _, bx, bval = self.metropolis(
                    start,
                    n_samples=n_samples,
                    prop_scale=prop_scale,
                    burn=burn,
                    seed=None if seed is None else (seed + i),
                )
                if i == 0 or bval < best_val:
                    best_x, best_val = bx, bval
            return best_x

        else:
            def _one_run(idx, start):
                _, _, bx, bval = self.metropolis(
                    start,
                    n_samples=n_samples,
                    prop_scale=prop_scale,
                    burn=burn,
                    seed=None if seed is None else (seed + idx),
                )
                return (bval, bx)

            r = Parallel(n_jobs=self.parallel, prefer="threads")(
                delayed(_one_run)(i, s) for i, s in enumerate(candidates)
            )
            bvals, bxs = zip(*r)
            bvals, bxs = np.array(bvals), np.array(bxs, dtype=float)
            return bxs[np.argmin(bvals)]

            
         

    def predict(self, X, y):
        '''
        Predicts the next best set of parameter values by optimising the acquisition function.
        
        Parameters
        ----------
        X : np.array of shape (n_samples * func.n_obs, func.dim)
            Training input points.

        y : np.array of shape (n_samples * func.n_obs, func.dim)
            Target values of the black-box function.
        
        n_pred : int, defult = 1
            Number of predictions to make.
        
        Returns
        -------
        next_x : np.array of shape (func.dim, n_pred)
        '''
        
        self.surr.fit(X, y)
        
        self.acqf.y = y.reshape(y.shape[0]//self.func.n_obs, self.func.n_obs)
        self.acqf.conds = X[::self.func.n_obs]
        self.acqf.surr = self.surr
            
        next_x = self.numerical_search(x0=X[np.argmin(y)])
    
        return np.array([next_x])
    
    
    def _save(self, base_dir='test_results'):
        res_path = create_dir(self.func, self.acqf.name, self.surr.name, base_dir=base_dir)

        res = {'X': self.params,
               'y': self.res,
               'err': self.errors}

        with open(f'{res_path}/{time.strftime("%d_%m_%H_%M_%S")}.pkl', 'wb') as f:
            pickle.dump(res, f)
    

    def plot_iteration_results(self, X, next_x):
        '''
        For surrogate optimiser only.
        '''
        if self.func.dim == 1:
            inter_conds = np.linspace(self.func.domain[:, 0], self.func.domain[:, 1], 121)
        else:
            inter_conds = self.create_candidates(N=10201)
        
        # transform = lambda x: np.exp(x) if self.func.log_transform else lambda x: x

        mu, _ = self.surr.predict(inter_conds)
        mu = mu
        y_true = self.func(inter_conds)
        
        if self.func.dim == 1:
            plt.plot(inter_conds, mu, label=f'Pred: {np.mean((y_true - mu)**2):.2f}', c='cornflowerblue')
            plt.plot(inter_conds, y_true, label='True', c='orange')
            # plt.scatter(X, y, c='black')
            plt.legend()
        else:
            
            y_true, mu = y_true.reshape(101, 101), mu.reshape(101, 101)
            mse = (y_true - mu)**2

            ei = self.acqf(inter_conds).reshape(101, 101)
            x_pred = inter_conds[np.argmin(ei)]
            
            def single_2d_plot(axis, f, title):
                plt.subplot(axis)
                colormap = plt.contourf(f, locator=ticker.LinearLocator(), extent=self.func.domain.flatten(), vmin=np.min(f), vmax=np.max(f))
                plt.scatter(X[:, 0], X[:, 1], color='black')
                for gb in self.func.glob_min:
                    plt.scatter(gb[0], gb[1], color='red', marker='*')
                plt.scatter(x_pred[0], x_pred[1], color='cyan', marker='*')
                plt.scatter(next_x[0], next_x[1], color='magenta', marker='*')
                plt.title(title)
                plt.colorbar(colormap)
            
            plt.figure(figsize=(24, 6))

            single_2d_plot(141, y_true, 'True Function')
            single_2d_plot(142, mu, 'Estimated Function')
            single_2d_plot(143, mse, 'MSE')
            single_2d_plot(144, ei, 'EI')
            
            plt.tight_layout()
        
        plt.show()

