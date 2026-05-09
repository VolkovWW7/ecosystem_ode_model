import numpy as np
from scipy.optimize import differential_evolution
import mathmodel
import json

# ==================== ЭКСПЕРИМЕНТАЛЬНЫЕ ДАННЫЕ ====================
# Углеродная серия: (C0, Хлорелла, Псевдомонады), N0=0.15
exp_carbon = [
    (0.10, 0.001815, 0.000185),
    (0.15, 0.002411, 0.000250),
    (0.25, 0.002976, 0.000372),
    (0.35, 0.002976, 0.000595),
    (0.40, 0.003095, 0.000774),
    (0.45, 0.003869, 0.000536),
    (0.50, 0.005208, 0.000982),
    (0.55, 0.004911, 0.000714),
    (0.60, 0.005804, 0.000789),
    (0.65, 0.008631, 0.000774),
    (0.85, 0.007589, 0.000878),
    (1.05, 0.007887, 0.000417),
    (1.25, 0.009673, 0.001012),
]

# Азотная серия: (N0, Хлорелла, Псевдомонады), C0=0.625
exp_nitrogen = [
    (0.019, 0.000216, 0.000728),
    (0.028, 0.001071, 0.001216),
    (0.045, 0.015476, 0.002857),
    (0.054, 0.015327, 0.001046),
    (0.063, 0.031994, 0.000890),
    (0.080, 0.004256, 0.000476),
    (0.098, 0.007946, 0.000402),
    (0.115, 0.006696, 0.001049),
    (0.133, 0.003423, 0.000536),
    (0.176, 0.006101, 0.000670),
    (0.194, 0.005119, 0.000564),
    (0.211, 0.002381, 0.000652),
]

C0_FIXED_N_SERIES = 0.625
N0_FIXED_C_SERIES = 0.15

# ==================== ЦЕЛЕВАЯ ФУНКЦИЯ ====================
# Глобальные переменные для отслеживания прогресса
best_error = float('inf')
eval_count = 0

def objective(params):
    global best_error, eval_count
    eval_count += 1
    Vx_c, Kc, Kn, Mxx, k_d1 = params

    # Базовые параметры модели
    p = mathmodel.get_default_params()
    p.update({
        'Vx_c': Vx_c,
        'Kc': Kc,
        'Kn': Kn,
        'Mxx': Mxx,
        'k_d1': k_d1,
        'total_time': 5000.0,    # для ускорения, можно увеличить до 20000
        'output_step': 100,
    })
    p = mathmodel.update_dependent_params(p)

    errors = []
    eps = 1e-9

    # ---- Углеродная серия ----
    p['N0'] = N0_FIXED_C_SERIES
    for C0, x_exp, y_exp in exp_carbon:
        p['C0'] = C0
        try:
            sol = mathmodel.run_simulation(p,"BDF")
            X_mod = sol.y[0, -1]
            Y_mod = sol.y[1, -1]
        except Exception:
            return 1e10
        # относительные квадраты ошибок
        err_x = ((X_mod - x_exp) / (x_exp + eps)) ** 2
        err_y = ((Y_mod - y_exp) / (y_exp + eps)) ** 2
        errors.append(err_x)
        errors.append(err_y)

    # ---- Азотная серия ----
    p['C0'] = C0_FIXED_N_SERIES
    for N0, x_exp, y_exp in exp_nitrogen:
        p['N0'] = N0
        try:
            sol = mathmodel.run_simulation(p,"BDF")
            X_mod = sol.y[0, -1]
            Y_mod = sol.y[1, -1]
        except Exception:
            return 1e10
        err_x = ((X_mod - x_exp) / (x_exp + eps)) ** 2
        err_y = ((Y_mod - y_exp) / (y_exp + eps)) ** 2
        errors.append(err_x)
        errors.append(err_y)

    rmsre = np.sqrt(np.mean(errors))

    # Вывод прогресса каждые 20 вычислений
    if eval_count % 20 == 0:
        print(f"Оценка {eval_count}: RMSRE = {rmsre:.6f}, текущие параметры: {params}")
    # Если нашли лучшее
    if rmsre < best_error - 1e-12:
        best_error = rmsre
        print(f">>> НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ (оценка {eval_count}): RMSRE={rmsre:.6f}, параметры={params}")
    return rmsre

# ==================== ЗАПУСК ОПТИМИЗАЦИИ ====================
if __name__ == "__main__":
    # Границы поиска для 5 параметров: Vx_c, Kc, Kn, Mxx, k_d1
    bounds = [
        (5.0, 20.0),   # Vx_c
        (0.1, 1.0),    # Kc
        (0.01, 0.1),   # Kn
        (0.001, 0.02), # Mxx
        (0.0001, 0.01) # k_d1
    ]

    print("Запуск калибровки параметров модели (дифференциальная эволюция)")
    print("Это может занять 20-60 минут в зависимости от числа поколений...")

    result = differential_evolution(
        objective,
        bounds,
        maxiter=20,      # количество поколений (можно увеличить для точности)
        popsize=10,      # размер популяции
        seed=42,
        workers=1,
        disp=True        # выводить прогресс от scipy
    )

    if result.success:
        print("\n" + "="*50)
        print("ОПТИМИЗАЦИЯ УСПЕШНО ЗАВЕРШЕНА")
        print(f"Лучшее значение функции (RMSRE): {result.fun:.8f}")
        print("Оптимальные параметры:")
        names = ['Vx_c', 'Kc', 'Kn', 'Mxx', 'k_d1']
        for name, val in zip(names, result.x):
            print(f"  {name} = {val:.8f}")

        # Сохраняем параметры в JSON для использования в GUI
        opt_params = mathmodel.get_default_params()
        opt_params.update({
            'Vx_c': result.x[0],
            'Kc': result.x[1],
            'Kn': result.x[2],
            'Mxx': result.x[3],
            'k_d1': result.x[4],
        })
        opt_params = mathmodel.update_dependent_params(opt_params)
        mathmodel.save_params_to_json(opt_params, "calibrated_params.json")
        print("\nПараметры сохранены в 'calibrated_params.json'")
    else:
        print("Ошибка оптимизации:", result.message)
