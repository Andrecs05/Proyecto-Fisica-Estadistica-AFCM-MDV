"""
Avance semana 12 - Codigo preliminar.

Reproduccion de J. R. Ray y C. Frelechoz, "Microcanonical ensemble Monte Carlo
method for discrete systems", Phys. Rev. E 53, 3402 (1996).

Modelo de Ising 2D (J = 1, k_B = 1, B = 0, condiciones periodicas):
    U(s) = - sum_<ij> s_i s_j

Se implementan:
  1. Metropolis canonico (T fija), barrido secuencial de espines.
  2. Monte Carlo microcanonico de Ray (E fija) con densidad de configuraciones
         W_E(s) ∝ (E - U(s))^(n-1) Θ(E - U(s)),   n = d N / 2 = N  (d = 2).
  3. Prueba de verificacion interna: red 4x4 enumerada exactamente (2^16
     configuraciones) -> c(T) canonico y c(E) microcanonico exactos.
  4. Referencia exacta canonica para la red L x L finita (formula de
     Kaufman / Ferdinand-Fisher) evaluada con mpmath.

Uso:
    python ising_microcanonico.py            # corrida preliminar (unos minutos)
    python ising_microcanonico.py --rapido   # corrida corta de humo

Todas las corridas usan semillas fijas para que los resultados sean
reproducibles.
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
from numba import njit

AQUI = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(AQUI, "..", "figuras")
RES = os.path.join(AQUI, "..", "resultados")

# --------------------------------------------------------------------------
# Datos del articulo (Tablas I y II, red 30x30)
# --------------------------------------------------------------------------
PAPER_T_CAN = np.array([1.5, 2.0, 2.1, 2.2, 2.25, 2.3, 2.4, 2.5, 3.0])
PAPER_E_CAN = np.array([-406.02, 229.00, 394.09, 587.74, 703.09, 829.94,
                        1066.83, 1253.00, 1964.52])
PAPER_C_CAN = np.array([0.19723, 0.72530, 0.96278, 1.3950, 1.72195, 1.8712,
                        1.3026, 0.89491, 0.40124])
PAPER_T_MIC = np.array([1.5005, 2.0014, 2.1086, 2.2031, 2.2531, 2.2997,
                        2.3957, 2.4977, 2.9984])
PAPER_E_MIC = np.array([-406.00, 229.00, 394.08, 588.20, 703.09, 829.94,
                        1066.00, 1253.00, 1964.00])
PAPER_C_MIC = np.array([0.19576, 0.71799, 0.97132, 1.3632, 1.7727, 2.1707,
                        1.2687, 0.87424, 0.40004])


# --------------------------------------------------------------------------
# Utilidades de red
# --------------------------------------------------------------------------
@njit
def energia(s):
    """U = -sum_<ij> s_i s_j con fronteras periodicas (cada enlace una vez)."""
    L = s.shape[0]
    U = 0
    for i in range(L):
        for j in range(L):
            U -= s[i, j] * (s[(i + 1) % L, j] + s[i, (j + 1) % L])
    return U


@njit
def campo_local(s, i, j):
    L = s.shape[0]
    return (s[(i + 1) % L, j] + s[(i - 1) % L, j]
            + s[i, (j + 1) % L] + s[i, (j - 1) % L])


# --------------------------------------------------------------------------
# 1. Metropolis canonico
# --------------------------------------------------------------------------
@njit
def metropolis_canonico(L, T, n_eq, n_med, semilla):
    """Devuelve las series U(t) y |M|(t), una medida por barrido."""
    np.random.seed(semilla)
    s = np.ones((L, L), dtype=np.int64)
    U = energia(s)
    M = L * L
    # tabla de aceptacion: dU en {4, 8} son los unicos casos con dU > 0
    acc4 = np.exp(-4.0 / T)
    acc8 = np.exp(-8.0 / T)
    serie_U = np.empty(n_med)
    serie_M = np.empty(n_med)
    for t in range(n_eq + n_med):
        for i in range(L):
            for j in range(L):
                dU = 2 * s[i, j] * campo_local(s, i, j)
                if dU <= 0:
                    acepta = True
                elif dU == 4:
                    acepta = np.random.random() < acc4
                else:
                    acepta = np.random.random() < acc8
                if acepta:
                    s[i, j] = -s[i, j]
                    U += dU
                    M += 2 * s[i, j]
        if t >= n_eq:
            serie_U[t - n_eq] = U
            serie_M[t - n_eq] = abs(M)
    return serie_U, serie_M


# --------------------------------------------------------------------------
# 2. Monte Carlo microcanonico de Ray
# --------------------------------------------------------------------------
@njit
def metropolis_microcanonico(L, E, n_eq, n_med, semilla, d=2):
    """
    Cadena de Markov con distribucion estacionaria
        W_E(s) ∝ (E - U)^(n-1) Θ(E - U),  n = d N / 2.
    Probabilidad de aceptacion: min[1, ((E-U')/(E-U))^(n-1)] si E - U' > 0.
    Devuelve las series K(t) = E - U(t) y |M|(t).
    """
    np.random.seed(semilla)
    N = L * L
    n = d * N / 2.0
    s = np.ones((L, L), dtype=np.int64)  # U = -2N: estado de minima energia
    U = energia(s)
    M = N
    if E - U <= 0:
        raise ValueError("E debe ser mayor que la energia del estado base")
    serie_K = np.empty(n_med)
    serie_M = np.empty(n_med)
    for t in range(n_eq + n_med):
        for i in range(L):
            for j in range(L):
                dU = 2 * s[i, j] * campo_local(s, i, j)
                K_nueva = E - (U + dU)
                if K_nueva <= 0.0:
                    continue  # Θ(E - U') = 0: estado prohibido
                if dU <= 0:
                    acepta = True
                else:
                    # ((E-U')/(E-U))^(n-1) evaluado en forma logaritmica
                    razon = np.exp((n - 1.0) * np.log(K_nueva / (E - U)))
                    acepta = np.random.random() < razon
                if acepta:
                    s[i, j] = -s[i, j]
                    U += dU
                    M += 2 * s[i, j]
        if t >= n_eq:
            serie_K[t - n_eq] = E - U
            serie_M[t - n_eq] = abs(M)
    return serie_K, serie_M


# --------------------------------------------------------------------------
# Estimadores (formulas de fluctuacion)
# --------------------------------------------------------------------------
def estimadores_canonicos(serie_U, serie_M, T, N, d=2):
    """Ec. (4) y Ec. (6) del articulo, en unidades J = k_B = 1."""
    U = serie_U.mean()
    c = serie_U.var() / (N * T**2)
    E = d * N * T / 2 + U
    return dict(T=T, E=E, U=U, c=c, M=serie_M.mean() / N)


def estimadores_microcanonicos(serie_K, serie_M, E, N, d=2):
    """Ec. (11) y Ec. (12) del articulo, en unidades J = k_B = 1."""
    n = d * N / 2
    K = serie_K.mean()
    K_inv = (1.0 / serie_K).mean()
    T = K / n                                   # Ec. (11)
    C_B = n / (n - (n - 1) * K * K_inv)         # Ec. (12)
    c = (C_B - n) / N                           # C_B = d N k_B / 2 + C
    return dict(T=T, E=E, U=E - K, c=c, M=serie_M.mean() / N)


def una_corrida(args):
    """Una corrida independiente: ensamble ('can' o 'mic'), L, T o E, barridos, semilla."""
    ens, L, x, n_eq, n_med, semilla = args
    if ens == "can":
        a, b = metropolis_canonico(L, x, n_eq, n_med, semilla)
        return estimadores_canonicos(a, b, x, L * L)
    a, b = metropolis_microcanonico(L, x, n_eq, n_med, semilla)
    return estimadores_microcanonicos(a, b, x, L * L)


def resumir(vals):
    """Media y desviacion estandar de las corridas independientes (como el articulo)."""
    out = {}
    for key in vals[0]:
        arr = np.array([v[key] for v in vals])
        out[key] = arr.mean()
        out["d" + key] = arr.std(ddof=1) if len(vals) > 1 else np.nan
    return out


def corridas(ens, L, x, n_eq, n_med, n_corr, semilla0):
    return resumir([una_corrida((ens, L, x, n_eq, n_med, semilla0 + k))
                    for k in range(n_corr)])


# --------------------------------------------------------------------------
# 3. Enumeracion exacta (prueba de verificacion interna)
# --------------------------------------------------------------------------
def densidad_estados_exacta(L):
    """g(U) por enumeracion de las 2^(L^2) configuraciones (solo L <= 4)."""
    N = L * L
    idx = np.arange(2**N, dtype=np.int64)
    bits = ((idx[:, None] >> np.arange(N)) & 1).astype(np.int8)
    s = (2 * bits - 1).reshape(-1, L, L)
    U = -(s * np.roll(s, 1, axis=1)).sum(axis=(1, 2)) \
        - (s * np.roll(s, 1, axis=2)).sum(axis=(1, 2))
    niveles, g = np.unique(U, return_counts=True)
    return niveles.astype(float), g.astype(float)


def c_canonico_exacto(niveles, g, T, N):
    b = 1.0 / T
    w = g * np.exp(-b * (niveles - niveles.min()))
    Z = w.sum()
    U1 = (w * niveles).sum() / Z
    U2 = (w * niveles**2).sum() / Z
    return (U2 - U1**2) / (N * T**2), U1


def micro_exacto(niveles, g, E, N, d=2):
    """T(E) y c(E) exactos a partir de Z_m(E) = sum_U g(U) (E-U)^m Θ(E-U)."""
    n = d * N / 2
    K = E - niveles
    ok = K > 0
    K, gg = K[ok], g[ok]
    Z = lambda m: (gg * K**m).sum()
    K_med = Z(n) / Z(n - 1)
    K_inv = Z(n - 2) / Z(n - 1)
    T = K_med / n
    C_B = n / (n - (n - 1) * K_med * K_inv)
    return T, (C_B - n) / N


# --------------------------------------------------------------------------
# 4. Solucion exacta canonica para una red m x n finita (Kaufman 1949;
#    Ferdinand y Fisher 1969). J = k_B = 1.
# --------------------------------------------------------------------------
mp.mp.dps = 40


def lnZ_kaufman(beta, m, n):
    K = mp.mpf(beta)
    c2, s2 = mp.cosh(2 * K), mp.sinh(2 * K)
    def gamma(l):
        if l == 0:
            return 2 * K + mp.log(mp.tanh(K))  # con signo
        return mp.acosh(c2 * c2 / s2 - mp.cos(mp.pi * l / n))
    Z1 = Z2 = Z3 = Z4 = mp.mpf(1)
    for r in range(n):
        g_odd, g_even = gamma(2 * r + 1), gamma(2 * r)
        Z1 *= 2 * mp.cosh(m * g_odd / 2)
        Z2 *= 2 * mp.sinh(m * g_odd / 2)
        Z3 *= 2 * mp.cosh(m * g_even / 2)
        Z4 *= 2 * mp.sinh(m * g_even / 2)
    return mp.log(mp.mpf(1) / 2 * (2 * s2) ** (m * n / mp.mpf(2))
                  * (Z1 + Z2 + Z3 + Z4))


def c_kaufman(T, L):
    """c = beta^2 d^2 lnZ / d beta^2 / N ;  U = -d lnZ / d beta."""
    b = mp.mpf(1) / T
    f = lambda x: lnZ_kaufman(x, L, L)
    d1 = mp.diff(f, b, 1)
    d2 = mp.diff(f, b, 2)
    return float(b**2 * d2 / (L * L)), float(-d1)


# --------------------------------------------------------------------------
# Programa principal
# --------------------------------------------------------------------------
def verificar_4x4(n_eq, n_med, n_corr):
    print("== Prueba interna: red 4x4 exacta ==")
    L, N = 4, 16
    niv, g = densidad_estados_exacta(L)
    assert g.sum() == 2**N

    # (a) la formula de Kaufman reproduce la enumeracion exacta
    for T in (1.5, 2.269, 3.0):
        c_enum, U_enum = c_canonico_exacto(niv, g, T, N)
        c_k, U_k = c_kaufman(T, L)
        print(f"  T={T:5.3f}  c_enum={c_enum:.10f}  c_Kaufman={c_k:.10f}"
              f"  U_enum={U_enum:.8f}  U_Kaufman={U_k:.8f}")
        assert abs(c_enum - c_k) < 1e-8 and abs(U_enum - U_k) < 1e-8

    # (b) curvas exactas y puntos Monte Carlo
    T_fino = np.linspace(1.0, 4.0, 300)
    c_can = np.array([c_canonico_exacto(niv, g, T, N)[0] for T in T_fino])
    E_fino = np.array([N * T + c_canonico_exacto(niv, g, T, N)[1] for T in T_fino])
    mic = np.array([micro_exacto(niv, g, E, N) for E in E_fino])

    T_mc = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    can_mc, mic_mc = [], []
    for T in T_mc:
        rc = corridas("can", L, T, n_eq, n_med, n_corr, 1000)
        E = N * T + c_canonico_exacto(niv, g, T, N)[1]
        rm = corridas("mic", L, E, n_eq, n_med, n_corr, 2000)
        Tm_ex, cm_ex = micro_exacto(niv, g, E, N)
        print(f"  T={T:.2f}: c_can MC={rc['c']:.4f}±{rc['dc']:.4f} exacto="
              f"{c_canonico_exacto(niv, g, T, N)[0]:.4f} | E={E:.3f}: "
              f"T_mic MC={rm['T']:.4f} exacto={Tm_ex:.4f}, "
              f"c_mic MC={rm['c']:.4f}±{rm['dc']:.4f} exacto={cm_ex:.4f}")
        can_mc.append(rc)
        mic_mc.append(rm)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(T_fino, c_can, "-", color="C0", label="canónico exacto")
    ax.plot(mic[:, 0], mic[:, 1], "--", color="C3", label="microcanónico exacto")
    ax.errorbar(T_mc, [r["c"] for r in can_mc], yerr=[r["dc"] for r in can_mc],
                fmt="s", mfc="none", color="C0", label="Metropolis canónico")
    ax.errorbar([r["T"] for r in mic_mc], [r["c"] for r in mic_mc],
                yerr=[r["dc"] for r in mic_mc], fmt="D", mfc="none", color="C3",
                label="MC microcanónico")
    ax.set_xlabel(r"$k_BT/J$")
    ax.set_ylabel(r"$c = C/Nk_B$")
    ax.set_title(r"Verificación: red $4\times4$ (enumeración exacta)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "verificacion_4x4.pdf"))
    fig.savefig(os.path.join(FIGS, "verificacion_4x4.png"), dpi=200)
    plt.close(fig)


def reproducir_30x30(n_eq, n_med, n_corr):
    print("== Red 30x30 (preliminar) ==")
    L, N = 30, 900
    # Como en el articulo: la energia de cada corrida microcanonica es la E
    # obtenida de la corrida canonica via Ec. (4); aqui se toman los valores
    # de entrada de la Tabla II para comparar exactamente con el articulo.
    tareas = []
    for k, T in enumerate(PAPER_T_CAN):
        for r in range(n_corr):
            tareas.append(("can", L, T, n_eq, n_med, 100 * k + r + 1))
            tareas.append(("mic", L, PAPER_E_MIC[k], n_eq, n_med, 100 * k + r + 51))
    t0 = time.time()
    with ProcessPoolExecutor() as pool:
        salida = list(pool.map(una_corrida, tareas))
    print(f"  {len(tareas)} corridas en {time.time() - t0:.0f} s")
    filas = []
    for k, T in enumerate(PAPER_T_CAN):
        E = PAPER_E_MIC[k]
        rc = resumir([v for t, v in zip(tareas, salida) if t[0] == "can" and t[2] == T])
        rm = resumir([v for t, v in zip(tareas, salida) if t[0] == "mic" and t[2] == E])
        print(f"  T={T:.3f}: E_can={rc['E']:8.2f}±{rc['dE']:.2f}  "
              f"c_can={rc['c']:.4f}±{rc['dc']:.4f} (art. {PAPER_C_CAN[k]:.4f}) | "
              f"E={E:8.2f}: T_mic={rm['T']:.4f}±{rm['dT']:.4f} "
              f"c_mic={rm['c']:.4f}±{rm['dc']:.4f} (art. {PAPER_C_MIC[k]:.4f})"
              )
        filas.append([T, rc["E"], rc["dE"], rc["M"], rc["c"], rc["dc"],
                      E, rm["T"], rm["dT"], rm["M"], rm["c"], rm["dc"]])
    filas = np.array(filas)
    np.savetxt(os.path.join(RES, "resultados_30x30.csv"), filas, delimiter=",",
               header="T_can,E_can,dE_can,M_can,c_can,dc_can,"
                      "E_mic,T_mic,dT_mic,M_mic,c_mic,dc_mic", comments="")

    escribir_tabla(filas)
    graficar_30x30(filas)


def graficar_30x30(filas):
    """Figuras 30x30 a partir de las filas de resultados (ver resultados_30x30.csv)."""
    L = 30
    T_fino = np.linspace(1.4, 3.1, 120)
    c_ex = np.array([c_kaufman(T, L)[0] for T in T_fino])

    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.7))
    for ax, xlim in zip(axs, [(1.4, 3.1), (2.18, 2.42)]):
        ax.plot(T_fino, c_ex, "-", color="0.6", lw=1,
                label=r"canónico exacto $30\times30$ (Kaufman)")
        ax.plot(PAPER_T_CAN, PAPER_C_CAN, "s", color="C0", ms=7, alpha=0.35,
                label="artículo, canónico")
        ax.plot(PAPER_T_MIC, PAPER_C_MIC, "D", color="C3", ms=7, alpha=0.35,
                label="artículo, microcanónico")
        ax.errorbar(filas[:, 0], filas[:, 4], yerr=filas[:, 5], fmt="s",
                    mfc="none", color="C0", capsize=2, label="este trabajo, canónico")
        ax.errorbar(filas[:, 7], filas[:, 10], xerr=filas[:, 8], yerr=filas[:, 11],
                    fmt="D", mfc="none", color="C3", capsize=2,
                    label="este trabajo, microcanónico")
        ax.set_xlim(*xlim)
        ax.set_xlabel(r"$k_BT/J$")
        ax.set_ylabel(r"$c = C/Nk_B$")
    axs[1].set_ylim(1.2, 2.6)
    axs[0].legend(frameon=False, fontsize=7, loc="upper right")
    axs[0].set_title(r"Calor específico, $30\times30$ (compárese Fig. 2)")
    axs[1].set_title("Región del máximo (compárese Fig. 3)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "calor_especifico_30x30.pdf"))
    fig.savefig(os.path.join(FIGS, "calor_especifico_30x30.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.plot(PAPER_T_CAN, PAPER_E_CAN, "s", color="C0", ms=7, alpha=0.35,
            label="artículo, canónico")
    ax.plot(filas[:, 0], filas[:, 1], "s", mfc="none", color="C0",
            label="este trabajo, canónico")
    ax.plot(filas[:, 7], filas[:, 6], "D", mfc="none", color="C3",
            label="este trabajo, microcanónico")
    ax.set_xlabel(r"$k_BT/J$")
    ax.set_ylabel(r"$E/J$")
    ax.set_title(r"Energía total vs temperatura (compárese Fig. 1)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "energia_30x30.pdf"))
    fig.savefig(os.path.join(FIGS, "energia_30x30.png"), dpi=200)
    plt.close(fig)


def escribir_tabla(filas):
    """Tabla LaTeX con los resultados 30x30 frente a las Tablas I y II del articulo."""
    L = 30
    lin = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\caption{Resultados preliminares para la red $30\times30$ frente a las "
           r"Tablas~I y II del artículo. $c_{\rm ex}$ es el valor canónico exacto "
           r"(Kaufman) para la misma red finita. Incertidumbres: desviación estándar "
           r"de cinco corridas.}",
           r"\label{tab:30}",
           r"\begin{tabular}{c c c c | c c c c}", r"\toprule",
           r"\multicolumn{4}{c|}{Canónico} & \multicolumn{4}{c}{Microcanónico}\\",
           r"$T$ & $c$ (este trabajo) & $c$ (art.) & $c_{\rm ex}$ & $E$ & $T$ (este trabajo) "
           r"& $c$ (este trabajo) & $c$ (art.)\\", r"\midrule"]
    for k, f in enumerate(filas):
        c_ex = c_kaufman(f[0], L)[0]
        lin.append(rf"{f[0]:.3f} & ${f[4]:.4f}\pm{f[5]:.4f}$ & {PAPER_C_CAN[k]:.4f} & "
                   rf"{c_ex:.4f} & {f[6]:.2f} & ${f[7]:.4f}\pm{f[8]:.4f}$ & "
                   rf"${f[10]:.4f}\pm{f[11]:.4f}$ & {PAPER_C_MIC[k]:.4f}\\")
    lin += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(RES, "tabla_resultados.tex"), "w") as fh:
        fh.write("\n".join(lin) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true", help="corrida de humo")
    p.add_argument("--solo-figuras", action="store_true",
                   help="rehace tabla y figuras 30x30 desde resultados_30x30.csv")
    args = p.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    if args.solo_figuras:
        filas = np.loadtxt(os.path.join(RES, "resultados_30x30.csv"),
                           delimiter=",", skiprows=1)
        escribir_tabla(filas)
        graficar_30x30(filas)
    elif args.rapido:
        verificar_4x4(n_eq=2_000, n_med=20_000, n_corr=3)
        reproducir_30x30(n_eq=500, n_med=2_000, n_corr=2)
    else:
        verificar_4x4(n_eq=10_000, n_med=400_000, n_corr=5)
        # Mismo numero de barridos de medicion que el articulo (5 x 10^6 por punto);
        # equilibracion mas corta (10^5 en vez de 5 x 10^6).
        reproducir_30x30(n_eq=100_000, n_med=1_000_000, n_corr=5)
