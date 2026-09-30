#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
# For L= 32
import numpy as np
import matplotlib.pyplot as plt
L = 32
N = L * L

J_kT_values = np.linspace(0.1, 0.9, 12)

mc_steps = 3000
equil_steps = 800

rng = np.random.default_rng(42)


def init_lattice():
    return rng.choice([-1, 1], size=(L, L))


def glauber_step(lattice, beta, B):
    i = rng.integers(0, L)
    j = rng.integers(0, L)

    s = lattice[i, j]

    nn = (
        lattice[(i+1)%L, j] +
        lattice[(i-1)%L, j] +
        lattice[i, (j+1)%L] +
        lattice[i, (j-1)%L]
    )

    dE = 2 * s * (beta * nn + B)

    if dE < 0 or rng.random() < np.exp(-dE):
        lattice[i, j] *= -1


def total_energy(lattice, beta, B):
    E = 0

    for i in range(L):
        for j in range(L):

            s = lattice[i, j]

            neighbors = (
                lattice[(i+1)%L, j]
                + lattice[i, (j+1)%L]
            )

            E += -s * neighbors

    return E - (B/beta) * np.sum(lattice)


def simulate(B_scaled):

    m_list = []
    chi_list = []
    Cv_list = []

    for beta in J_kT_values:

        print(f"Running J/kT = {beta:.2f}")

        B = B_scaled

        lattice = init_lattice()

        M_vals = []
        E_vals = []

        for step in range(mc_steps):

            for _ in range(N):
                glauber_step(
                    lattice,
                    beta,
                    B
                )

            if step >= equil_steps:

                M_vals.append(
                    np.sum(lattice)
                )

                E_vals.append(
                    total_energy(
                        lattice,
                        beta,
                        B
                    )
                )

        M_vals = np.array(M_vals)
        E_vals = np.array(E_vals)

        m = (
            np.mean(np.abs(M_vals))
            / N
        )

        chi = (
            beta
            * (
                np.mean(M_vals**2)
                - np.mean(M_vals)**2
            )
            / N
        )

        Cv = (
            beta**2
            * (
                np.mean(E_vals**2)
                - np.mean(E_vals)**2
            )
            / N
        )

        m_list.append(m)
        chi_list.append(chi)
        Cv_list.append(Cv)

    return (
        np.array(m_list),
        np.array(chi_list),
        np.array(Cv_list)
    )


def plot(res, title):

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)

    plt.plot(
        J_kT_values,
        res[0],
        marker='o'
    )

    plt.title("Magnetization")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 2)

    plt.plot(
        J_kT_values,
        res[1],
        marker='o'
    )

    plt.title("Susceptibility")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 3)

    plt.plot(
        J_kT_values,
        res[2],
        marker='o'
    )

    plt.title("Specific Heat")
    plt.xlabel("J/kT")
    plt.grid()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


print("CASE 1: B = 0")

res1 = simulate(0)

plot(
    res1,
    "L = 32, B = 0"
)

print("CASE 2: B/kT = 0.5")

res2 = simulate(0.5)

plot(
    res2,
    "L = 32, B/kT = 0.5"
)
#%%
# For L= 64
import numpy as np
import matplotlib.pyplot as plt
L = 64
N = L * L

J_kT_values = np.linspace(0.1, 0.9, 12)

mc_steps = 3000
equil_steps = 800

rng = np.random.default_rng(42)


def init_lattice():
    return rng.choice([-1, 1], size=(L, L))

def glauber_step(lattice, beta, B):
    i = rng.integers(0, L)
    j = rng.integers(0, L)

    s = lattice[i, j]

    nn = (
        lattice[(i+1)%L, j] +
        lattice[(i-1)%L, j] +
        lattice[i, (j+1)%L] +
        lattice[i, (j-1)%L]
    )

    dE = 2 * s * (beta * nn + B)

    if dE < 0 or rng.random() < np.exp(-dE):
        lattice[i, j] *= -1


def total_energy(lattice, beta, B):
    E = 0

    for i in range(L):
        for j in range(L):
            s = lattice[i, j]

            neighbors = (
                lattice[(i+1)%L, j] +
                lattice[i, (j+1)%L]
            )

            E += -s * neighbors

    return E - (B/beta) * np.sum(lattice)


def simulate(B_scaled):

    m_list, chi_list, Cv_list = [], [], []

    for beta in J_kT_values:

        print(f"Running J/kT = {beta:.2f}")

        B = B_scaled

        lattice = init_lattice()

        M_vals, E_vals = [], []

        for step in range(mc_steps):

            for _ in range(N):
                glauber_step(lattice, beta, B)

            if step >= equil_steps:
                M_vals.append(np.sum(lattice))
                E_vals.append(total_energy(lattice, beta, B))

        M_vals = np.array(M_vals)
        E_vals = np.array(E_vals)

        m = np.mean(np.abs(M_vals)) / N

        chi = (
            beta
            * (np.mean(M_vals**2) - np.mean(M_vals)**2)
            / N
        )

        Cv = (
            beta**2
            * (np.mean(E_vals**2) - np.mean(E_vals)**2)
            / N
        )

        m_list.append(m)
        chi_list.append(chi)
        Cv_list.append(Cv)

    return (
        np.array(m_list),
        np.array(chi_list),
        np.array(Cv_list)
    )


def plot(res, title):

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(J_kT_values, res[0], marker='o')
    plt.title("Magnetization")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 2)
    plt.plot(J_kT_values, res[1], marker='o')
    plt.title("Susceptibility")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 3)
    plt.plot(J_kT_values, res[2], marker='o')
    plt.title("Specific Heat")
    plt.xlabel("J/kT")
    plt.grid()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

print("CASE 1: B = 0")
res1 = simulate(0)
plot(res1, "L = 64, B = 0")

print("CASE 2: B/kT = 0.5")
res2 = simulate(0.5)
plot(res2, "L = 64, B/kT = 0.5")
# %%
#L=128
import numpy as np
import matplotlib.pyplot as plt

L = 128
N = L * L

J_kT_values = np.linspace(0.1, 0.9, 12)

mc_steps = 3000
equil_steps = 800

rng = np.random.default_rng(42)


def init_lattice():
    return rng.choice([-1, 1], size=(L, L))


def glauber_step(lattice, beta, B):
    i = rng.integers(0, L)
    j = rng.integers(0, L)

    s = lattice[i, j]

    nn = (
        lattice[(i+1)%L, j] +
        lattice[(i-1)%L, j] +
        lattice[i, (j+1)%L] +
        lattice[i, (j-1)%L]
    )

    dE = 2 * s * (beta * nn + B)

    if dE < 0 or rng.random() < np.exp(-dE):
        lattice[i, j] *= -1


def total_energy(lattice, beta, B):
    E = 0

    for i in range(L):
        for j in range(L):

            s = lattice[i, j]

            neighbors = (
                lattice[(i+1)%L, j] +
                lattice[i, (j+1)%L]
            )

            E += -s * neighbors

    return E - (B/beta) * np.sum(lattice)


def simulate(B_scaled):

    m_list = []
    chi_list = []
    Cv_list = []

    for beta in J_kT_values:

        print(f"Running J/kT = {beta:.2f}")

        B = B_scaled

        lattice = init_lattice()

        M_vals = []
        E_vals = []

        for step in range(mc_steps):

            for _ in range(N):
                glauber_step(
                    lattice,
                    beta,
                    B
                )

            if step >= equil_steps:

                M_vals.append(
                    np.sum(lattice)
                )

                E_vals.append(
                    total_energy(
                        lattice,
                        beta,
                        B
                    )
                )

        M_vals = np.array(M_vals)
        E_vals = np.array(E_vals)

        m = (
            np.mean(np.abs(M_vals))
            / N
        )

        chi = (
            beta
            * (
                np.mean(M_vals**2)
                - np.mean(M_vals)**2
            )
            / N
        )

        Cv = (
            beta**2
            * (
                np.mean(E_vals**2)
                - np.mean(E_vals)**2
            )
            / N
        )

        m_list.append(m)
        chi_list.append(chi)
        Cv_list.append(Cv)

    return (
        np.array(m_list),
        np.array(chi_list),
        np.array(Cv_list)
    )


def plot(res, title):

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)

    plt.plot(
        J_kT_values,
        res[0],
        marker='o'
    )

    plt.title("Magnetization")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 2)

    plt.plot(
        J_kT_values,
        res[1],
        marker='o'
    )

    plt.title("Susceptibility")
    plt.xlabel("J/kT")
    plt.grid()

    plt.subplot(1, 3, 3)

    plt.plot(
        J_kT_values,
        res[2],
        marker='o'
    )

    plt.title("Specific Heat")
    plt.xlabel("J/kT")
    plt.grid()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


print("CASE 1: B = 0")

res1 = simulate(0)

plot(
    res1,
    "L = 128, B = 0"
)

print("CASE 2: B/kT = 0.5")

res2 = simulate(0.5)

plot(
    res2,
    "L = 128, B/kT = 0.5"
)

# %%
