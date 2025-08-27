import numpy as np
from waggon.optim.surrogate import SurrogateOptimiser

# ---- Игрушечные заглушки ----
class DummyFunc:
    dim = 2
    n_obs = 1
    domain = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    glob_min = [(0.3, -0.2)]
    log_transform = False      # <-- добавили

    def __call__(self, X):
        X = np.atleast_2d(X)
        return np.sum(X**2, axis=1)


class DummySurr:
    name = "dummy-surr"
    verbose = 0
    def fit(self, X, y):  # никаких действий
        pass
    def predict(self, X):
        X = np.atleast_2d(X)
        mu = np.sum(X**2, axis=1)  # не важно, лишь бы форма совпала
        var = np.zeros_like(mu)
        return mu, var

class DummyAcqf:
    name = "dummy-acqf"
    # цель: минимум в (0.3, -0.2)
    def __call__(self, X):
        X = np.atleast_2d(X)
        target = np.array([0.3, -0.2])
        vals = np.sum((X - target)**2, axis=1)
        return vals

# ---- Инициализация оптимизатора ----
opt = SurrogateOptimiser(
    func=DummyFunc(),
    surr=DummySurr(),
    acqf=DummyAcqf(),
    num_opt_start="random",       # чтобы не зависеть от create_candidates()
    num_opt_candidates=8,         # 8 стартов для мультистарта
)

# Настроим шаги MCMC (можно не задавать; есть дефолты)
opt.mcmc_n_samples  = 4000
opt.mcmc_burn       = 800
opt.mcmc_prop_scale = 0.1
opt.mcmc_seed       = 7

# ---- Подготовим фиктивные данные для predict (хотя тестируем numerical_search) ----
X = np.array([[0.0, 0.0]])        # одна точка обучения
y = np.array([0.0])               # одно значение (форма не важна)

# Проверим напрямую numerical_search:
best_x = opt.numerical_search(x0=np.array([0.9, 0.9]))
print("best_x from numerical_search:", best_x)

# Можно также проверить predict() — он просто вызывает numerical_search под капотом:
nx = opt.predict(X=np.array([[0.0, 0.0]]), y=np.array([0.0]))
print("predict() next_x:", nx)
