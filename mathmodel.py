#Математическая модель микробиоты автотрофов и гететрофов
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json

# --- МАТЕМАТИЧЕСКАЯ МОДЕЛЬ ---

def get_default_params():
    """Возвращает базовый словарь параметров из спецификации модели."""
    p = {
        # Автотрофы
        'Vx_c': 13.19696,    # было 10.0
        'Kc': 0.815067,      # было 0.3
        'Kn': 0.020917,      # было 0.03
        'Ki': 0.0,
        'Mxx': 0.01938,      # было 0.005
        'Ax': 1.0,
        # Гетеротрофы
        'Vy_c': 25.0, 'Kpp': 0.2, 'Kfc': 0.1, 'Ka': 0.2, 'Myy': 0.03, 'Ay': 1.0,
        # Потребление
        'Vm': 5.0, 'Kp': 1.0, 'Kf': 1.0, 'Kch': 0.5,
        # Детрит и АТФ
        'k_d1': 0.0007, 'k_d2': 0.0007, 'k_d': 0.5, 'eps2': 5.0,
        'aP': 0.3333, 'aF': 0.3333, 'aCh': 0.3333,
        # Стехиометрия биомассы
        'bXP': 0.58, 'bXF': 0.22, 'bYP': 0.60, 'bYF': 0.10,
        # Элементный состав
        'gPC': 0.77, 'gPN': 0.23, 'gFC': 1.0, 'gChC': 1.0,
        # Среда и начальные условия
        'C0': 0.11, 'N0': 0.15,
        'X0': 0.0024, 'Y0': 0.00025, 'Dcl0': 0.04, 'Dps0': 0.04, 'Ac0': 0.9,
        # Время
        'total_time': 20000.0, 'output_step': 10.0
    }
    
    # Производные коэффициенты
    p['bXCh'] = 1.0 - p['bXP'] - p['bXF']
    p['bYCh'] = 1.0 - p['bYP'] - p['bYF']
    p['gXC'] = p['bXP'] * p['gPC'] + p['bXF'] * p['gFC'] + p['bXCh'] * p['gChC']
    p['gXN'] = p['bXP'] * p['gPN']
    p['gYC'] = p['bYP'] * p['gPC'] + p['bYF'] * p['gFC'] + p['bYCh'] * p['gChC']
    p['gYN'] = p['bYP'] * p['gPN']
    
    return p

def update_dependent_params(p):
    """Пересчитывает зависимые коэффициенты на основе базовых."""
    p['bXCh'] = 1.0 - p['bXP'] - p['bXF']
    p['bYCh'] = 1.0 - p['bYP'] - p['bYF']
    p['gXC'] = p['bXP']*p['gPC'] + p['bXF']*p['gFC'] + p['bXCh']*p['gChC']
    p['gXN'] = p['bXP']*p['gPN']
    p['gYC'] = p['bYP']*p['gPC'] + p['bYF']*p['gFC'] + p['bYCh']*p['gChC']
    p['gYN'] = p['bYP']*p['gPN']
    return p

def sys(t, x, p):
    """
    Функция дифференциальных уравнений. t- время интегрирования, x =[ X,Y,Dcl,Dps,P,F,Ch,Ac], p - словарь параметров системы
    """
    #распаковка основных параметров
    X, Y, Dcl, Dps, P, F, Ch, Ac = x

    # Ограничение параметров. не позволяет использовать отрицательные значения
    X = max(0.0, X)
    Y = max(0.0, Y)
    Dcl = max(0.0, Dcl)
    Dps = max(0.0, Dps)
    P = max(0.0, P)
    F = max(0.0, F)
    Ch = max(0.0, Ch)
    Ac = np.clip(Ac, 0.0, 1.0)

    C = max(1e-9, p['C0'] - (p['gXC']*(X + Dcl) + p['gYC']*(Y + Dps) + p['gPC']*P + p['gFC']*F + p['gChC']*Ch))
    N = max(1e-9, p['N0'] - (p['gXN']*(X + Dcl) + p['gYN']*(Y + Dps) + p['gPN']*P))

    Ph_c = p['Vx_c'] * C * N / ((p['Kc'] + C) * (p['Kn'] + N + p['Ki'] * N**4))
    mX_c = p['Mxx'] * (1 + C*N + p['Ki'] * N**4) / (1 + p['Ax'] * C * N)

    v_atp = Ac * (2*(1 - Ac) - 0.1) / ((1 - Ac) + 0.1)
    atp_val = max(0.0, v_atp)

    f_c = p['Vy_c'] * P * F * Ch / ((p['Kpp'] + P) * (p['Kfc'] + F) * (p['Kfc'] + Ch)) * Ac / (p['Ka'] + Ac)

    gP_c = p['Vm'] * P / (p['Kp'] + P)
    gF_c = p['Vm'] * F / (p['Kf'] + F)
    gCh_c = p['Vm'] * Ch / (p['Kch'] + Ch)
    mY_c = p['Myy'] * (0.01 + Ac) / (0.01 + p['Ay'] * Ac)

    dx = [0.0]*8
    dx[0] = (Ph_c - mX_c) * X
    dx[1] = (f_c - mY_c) * Y
    dx[2] = mX_c * X - p['k_d1'] * Dcl
    dx[3] = mY_c * Y - p['k_d2'] * Dps
    dx[4] = (p['bXP']*p['k_d1']*Dcl + p['bYP']*p['k_d2']*Dps - (p['bYP']*f_c + p['aP']*gP_c*atp_val)*Y)
    dx[5] = (p['bXF']*p['k_d1']*Dcl + p['bYF']*p['k_d2']*Dps - (p['bYF']*f_c + p['aF']*gF_c*atp_val)*Y)
    dx[6] = (p['bXCh']*p['k_d1']*Dcl + p['bYCh']*p['k_d2']*Dps - (p['bYCh']*f_c + p['aCh']*gCh_c*atp_val)*Y)
    dx[7] = 10 * (5*atp_val*(p['aP']*gP_c + p['aF']*gF_c + p['aCh']*gCh_c)*Y/(0.0001+Y) - f_c - p['k_d']*Ac/(p['eps2']+Ac))
    return dx

def run_simulation(p,method):
    p0_val = (p['Dcl0'] + p['Dps0']) / 6
    x0 = [p['X0'], p['Y0'], p['Dcl0'], p['Dps0'], p0_val, p0_val, p0_val, p['Ac0']]
    t_span = (0, p['total_time'])
    t_eval = np.arange(0, p['total_time'] + p['output_step'], p['output_step'])
    return solve_ivp(sys, t_span, x0, args=(p,), method=method, t_eval=t_eval)

# --- ФУНКЦИИ ГРАФИКОВ ---
def _setup_ax(ax, title, xlabel='Время', ylabel='Концентрация'):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

def graph_Trofs(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[0], 'g-', lw=2, label='Автотрофы (X)')
    ax.plot(sol.t, sol.y[1], 'r-', lw=2, label='Гетеротрофы (Y)')
    _setup_ax(ax, 'Динамика автотрофов и гетеротрофов')
    return fig

def graph_Detrit(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[2], 'b-', label='Детрит хлореллы')
    ax.plot(sol.t, sol.y[3], 'm-', label='Детрит бактерий')
    _setup_ax(ax, 'Концентрация детрита')
    return fig

def graph_PFC(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[4], label='Белки (P)')
    ax.plot(sol.t, sol.y[5], label='Жиры (F)')
    ax.plot(sol.t, sol.y[6], label='Углеводы (Ch)')
    _setup_ax(ax, 'Концентрация БЖУ')
    return fig

def graph_Ac(sol):
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(111)
    ax.plot(sol.t, sol.y[7], 'k-', label='АТФ (Ac)')
    _setup_ax(ax, 'Внутриклеточная концентрация АТФ')
    return fig

# --- ФУНКЦИИ ДЛЯ РАБОТЫ С JSON И ОТЧЁТАМИ ---
def save_params_to_json(params, filepath):
    to_save = {k: v for k, v in params.items() 
               if k not in ['bXCh', 'bYCh', 'gXC', 'gXN', 'gYC', 'gYN']}
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(to_save, f, indent=2, ensure_ascii=False)

def load_params_from_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
    params = get_default_params()
    params.update(loaded)
    params = update_dependent_params(params)
    return params

def save_report_pdf(solution, pdf_path):
    with PdfPages(pdf_path) as pdf:
        fig1 = graph_Trofs(solution); pdf.savefig(fig1); plt.close(fig1)
        fig2 = graph_Detrit(solution); pdf.savefig(fig2); plt.close(fig2)
        fig3 = graph_PFC(solution); pdf.savefig(fig3); plt.close(fig3)
        fig4 = graph_Ac(solution); pdf.savefig(fig4); plt.close(fig4)
