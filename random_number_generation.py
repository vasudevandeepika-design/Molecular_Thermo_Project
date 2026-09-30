# %%

# Linear Congruential Generator (LCG)
# Recursive relation:
# R_(i+1) = mod(a * R_i + b, m)
# U_i = R_i / m

import numpy as np
import matplotlib.pyplot as plt

# Parameters


N = 10000


# Case (a)
a_a = 48271
b_a = 0
m_a = 2**31 - 1

# Fixed seed
R0_a = 9


# Case (b)
a_b = 1664525
b_b = 1013904223
m_b = 2**32

# Fixed seed
R0_b = 11



# LCG Function

def lcg(a, b, m, R0, N):

    R = np.zeros(N + 1, dtype=np.int64)

    R[0] = R0

    for i in range(N):
        R[i + 1] = (a * R[i] + b) % m

    R_seq = R[1:]

    U_seq = R_seq.astype(float) / float(m)

    return R_seq, U_seq



# Generate random numbers

R_a, U_a = lcg(
    a_a,
    b_a,
    m_a,
    R0_a,
    N
)

R_b, U_b = lcg(
    a_b,
    b_b,
    m_b,
    R0_b,
    N
)



# Output


print("=" * 50)
print("         LCG RANDOM NUMBERS")
print("=" * 50)



# Case (a)


print("\nCase (a):")
print(f"Seed R_0 = {R0_a}")

print("\nFirst 10 values:")
print(
    f"{'i':>5} {'R_i':>15} {'U_i':>15}"
)

for i in range(10):

    print(
        f"{i+1:5d} "
        f"{R_a[i]:15d} "
        f"{U_a[i]:15.6f}"
    )



# Case (b)

print("\nCase (b):")
print(f"Seed R_0 = {R0_b}")

print("\nFirst 10 values:")
print(
    f"{'i':>5} {'R_i':>15} {'U_i':>15}"
)

for i in range(10):

    print(
        f"{i+1:5d} "
        f"{R_b[i]:15d} "
        f"{U_b[i]:15.6f}"
    )



# Statistical Summary

print("\n")
print("=" * 50)
print("         STATISTICAL SUMMARY")
print("=" * 50)


print("\nCase (a) Statistics:")

print(
    f"Mean = {np.mean(U_a):.6f}  "
    f"Std = {np.std(U_a, ddof=1):.6f}  "
    f"Min = {np.min(U_a):.6f}  "
    f"Max = {np.max(U_a):.6f}"
)


print("\nCase (b) Statistics:")

print(
    f"Mean = {np.mean(U_b):.6f}  "
    f"Std = {np.std(U_b, ddof=1):.6f}  "
    f"Min = {np.min(U_b):.6f}  "
    f"Max = {np.max(U_b):.6f}"
)


# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------

plt.figure(figsize=(11, 5))

plt.suptitle(
    "Linear Congruential Generator (LCG): "
    "Distribution of U_i",
    fontsize=14,
    fontweight="bold"
)


# ------------------------------------------------------------
# Histogram - Case (a)
# ------------------------------------------------------------

plt.subplot(1, 2, 1)

plt.hist(
    U_a,
    bins=50,
    edgecolor="white"
)

plt.axhline(
    N / 50,
    linestyle="--",
    linewidth=1.5
)

plt.title(
    "Case (a): U_i = R_i / m",
    fontweight="bold"
)

plt.xlabel("U_i")
plt.ylabel("Frequency")

plt.grid(True)
plt.box(True)



# Histogram - Case (b)


plt.subplot(1, 2, 2)

plt.hist(
    U_b,
    bins=50,
    edgecolor="white"
)

plt.axhline(
    N / 50,
    linestyle="--",
    linewidth=1.5
)

plt.title(
    "Case (b): U_i = R_i / m",
    fontweight="bold"
)

plt.xlabel("U_i")
plt.ylabel("Frequency")

plt.grid(True)
plt.box(True)


plt.tight_layout()
plt.show()
# %%

#Shift Register Generator
import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 103
q = 250

L_values = [16, 32]

N = 1000

seed_length = 500  # must be > q

rng = np.random.default_rng(1)
# Initial Seed

R = rng.integers(
    0,
    2,
    size=seed_length
)
# Shift Generator based on binary sequence
#
# R_i = R_(i-p) XOR R_(i-q)
total_bits = max(L_values) * N + seed_length

# Create complete binary sequence
R = np.concatenate([
    R,
    np.zeros(total_bits - seed_length, dtype=int)
])

for i in range(q, total_bits):

    R[i] = np.logical_xor(
        R[i - p],
        R[i - q]
    ).astype(int)

# Store results
results = []
# Conversion of Bit Stream to Uniform Variates
for L in L_values:

    U = np.zeros(N, dtype=np.int64)
    X = np.zeros(N)

    for i in range(N):

        idx_start = seed_length + i * L
        idx_end = idx_start + L

        block = R[idx_start:idx_end]

        # Binary block → decimal integer
        U[i] = sum(
            block[j] * 2**(L - 1 - j)
            for j in range(L)
        )

        # Convert to uniform random number
        X[i] = U[i] / (2**L)

    results.append({
        "L": L,
        "U": U,
        "X": X
    })
# Output
print("--- PARAMETERS USED ---")

print(f"p = {p}, q = {q}")
print(f"Seed length = {seed_length} (> q)")
print(f"N (numbers generated) = {N}")
print("Formula: R_i = R_(i-p) XOR R_(i-q)")
print("Uniform conversion: X = int(block) / 2^L")

# First 10 generated numbers
for result in results:

    L = result["L"]
    U = result["U"]
    X = result["X"]

    print(f"\nRESULTS (L = {L})")
    print("-" * 45)

    print(
        f"{'i':>5} "
        f"{'U_i':>15} "
        f"{'X_i':>15}"
    )

    for i in range(10):

        print(
            f"{i+1:5d} "
            f"{U[i]:15d} "
            f"{X[i]:15.6f}"
        )

    print("-" * 45)
# Statistical Summary
for result in results:

    L = result["L"]
    X = result["X"]

    print(
        f"\nStatistical Summary (L = {L}):"
    )

    print(
        f"Mean = {np.mean(X):.6f}"
    )

    print(
        f"Std Dev = {np.std(X, ddof=1):.6f}"
    )

    print(
        f"Min = {np.min(X):.6f}"
    )

    print(
        f"Max = {np.max(X):.6f}"
    )
# Histogram plots
for result in results:

    L = result["L"]
    X = result["X"]

    plt.figure(
        figsize=(7, 5),
        facecolor="white"
    )

    plt.hist(
        X,
        bins=30,
        edgecolor="black",
        linewidth=0.8
    )

    plt.title(
        f"Uniform Distribution of Random Numbers (L = {L})",
        fontweight="bold"
    )

    plt.xlabel("Random Number (X)")
    plt.ylabel("Frequency")

    plt.grid(True)

    plt.tight_layout()
    plt.show()

    #%%

# Mersenne Twister

import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------

N = 1000

# LCG seeds
R0_a = 9
R0_b = 11


# ============================================================
# LCG GENERATION
# ============================================================

a1 = 48271
b1 = 0
m1 = 2**31 - 1

a2 = 1664525
b2 = 1013904223
m2 = 2**32


R1 = np.zeros(N, dtype=np.int64)
R2 = np.zeros(N, dtype=np.int64)

R1[0] = R0_a
R2[0] = R0_b


for i in range(1, N):

    R1[i] = (a1 * R1[i-1] + b1) % m1

    R2[i] = (a2 * R2[i-1] + b2) % m2


U1 = R1.astype(float) / m1
U2 = R2.astype(float) / m2


# ============================================================
# SHIFT REGISTER GENERATION
# ============================================================

p = 103
q = 250

L_vals = [16, 32]

seed_length = 500


rng = np.random.default_rng(1)


# Initial binary seed
R = rng.integers(
    0,
    2,
    size=seed_length
)


# Total number of bits required
total_bits = max(L_vals) * N + seed_length


# Extend the array
R = np.concatenate([
    R,
    np.zeros(
        total_bits - seed_length,
        dtype=int
    )
])


# Generate binary sequence
#
# R_i = R_(i-p) XOR R_(i-q)

for i in range(q, total_bits):

    R[i] = np.logical_xor(
        R[i-p],
        R[i-q]
    ).astype(int)


# ------------------------------------------------------------
# Convert binary sequence into uniform random numbers
# ------------------------------------------------------------

U_shift = []


for L in L_vals:

    U_temp = np.zeros(N)

    for i in range(N):

        idx1 = seed_length + i * L
        idx2 = idx1 + L

        block = R[idx1:idx2]

        # Binary → decimal
        J = sum(
            block[j] * 2**(L - 1 - j)
            for j in range(L)
        )

        # Convert to [0,1)
        U_temp[i] = J / (2**L)

    U_shift.append(U_temp)


# ============================================================
# MERSENNE TWISTER
# ============================================================

rng = np.random.default_rng(123)

U_MT = rng.random(N)


# ============================================================
# MEAN TEST
# ============================================================

def mean_test(U):

    n = len(U)

    mean_value = np.mean(U)

    lower_limit = (
        0.5 - 1 / np.sqrt(12 * n)
    )

    upper_limit = (
        0.5 + 1 / np.sqrt(12 * n)
    )

    return (
        mean_value,
        lower_limit,
        upper_limit
    )


# ============================================================
# RUN TEST
# ============================================================

def run_test(U):

    runs = 1

    for i in range(1, len(U) - 1):

        if (
            (U[i] > U[i-1] and U[i] > U[i+1])
            or
            (U[i] < U[i-1] and U[i] < U[i+1])
        ):

            runs += 1

    return runs


# ============================================================
# RUN LENGTH FUNCTION
# ============================================================

def run_lengths(U):

    lengths = []

    length = 1

    for i in range(1, len(U)):

        if U[i] > U[i-1]:

            length += 1

        else:

            lengths.append(length)

            length = 1

    lengths.append(length)

    return np.array(lengths)


# ============================================================
# APPLY TESTS
# ============================================================

names = [
    "LCG (48271)",
    "LCG (1664525)",
    "Shift (L=16)",
    "Shift (L=32)",
    "MT"
]


data = [
    U1,
    U2,
    U_shift[0],
    U_shift[1],
    U_MT
]


Mean = np.zeros(5)

Runs = np.zeros(5, dtype=int)

Result = []


for i in range(5):

    m, lower, upper = mean_test(data[i])

    r = run_test(data[i])

    Mean[i] = m

    Runs[i] = r


    if lower <= m <= upper:

        Result.append("PASS")

    else:

        Result.append("FAIL")


# ============================================================
# FINAL RESULTS TABLE
# ============================================================

print("\n")
print("=" * 65)
print("                    FINAL RESULTS")
print("=" * 65)

print(
    f"{'Generator':<20}"
    f"{'Mean':>12}"
    f"{'Runs':>12}"
    f"{'Mean Test':>15}"
)

print("-" * 65)


for i in range(5):

    print(
        f"{names[i]:<20}"
        f"{Mean[i]:>12.6f}"
        f"{Runs[i]:>12d}"
        f"{Result[i]:>15}"
    )


# ============================================================
# COMPARISON
# ============================================================

# Distance from ideal mean = 0.5

dist = np.abs(Mean - 0.5)


# Generator with minimum distance from 0.5

idx_best = np.argmin(dist)


print("\n")
print("Based on Mean Test (closest to 0.5):")

print(
    f"Generator: {names[idx_best]}"
)

print(
    f"Mean Value = {Mean[idx_best]:.6f}"
)


# ============================================================
# COLORS
# ============================================================

orange = (1.0, 0.6, 0.2)
purple = (0.6, 0.4, 0.8)
blue = (0.5, 0.7, 0.9)
green = (0.6, 0.8, 0.6)
pink = (0.9, 0.6, 0.7)


# ============================================================
# LCG PLOTS
# ============================================================

plt.figure(
    figsize=(11, 8),
    facecolor="white"
)


# LCG 1 sequence

plt.subplot(2, 2, 1)

plt.plot(
    U1,
    color=orange,
    linewidth=1
)

plt.title(
    "LCG (48271) Sequence",
    fontweight="bold"
)

plt.xlabel("Index")
plt.ylabel("U_i")

plt.grid(True)


# LCG 1 distribution

plt.subplot(2, 2, 2)

plt.hist(
    U1,
    bins=40,
    color=orange,
    edgecolor="black"
)

plt.title(
    "LCG (48271) Distribution",
    fontweight="bold"
)

plt.xlabel("U_i")
plt.ylabel("Frequency")

plt.grid(True)


# LCG 2 sequence

plt.subplot(2, 2, 3)

plt.plot(
    U2,
    color=purple,
    linewidth=1
)

plt.title(
    "LCG (1664525) Sequence",
    fontweight="bold"
)

plt.xlabel("Index")
plt.ylabel("U_i")

plt.grid(True)


# LCG 2 distribution

plt.subplot(2, 2, 4)

plt.hist(
    U2,
    bins=40,
    color=purple,
    edgecolor="black"
)

plt.title(
    "LCG (1664525) Distribution",
    fontweight="bold"
)

plt.xlabel("U_i")
plt.ylabel("Frequency")

plt.grid(True)


plt.tight_layout()
plt.show()


# ============================================================
# SHIFT REGISTER PLOTS
# ============================================================

plt.figure(
    figsize=(11, 8),
    facecolor="white"
)


# Shift L = 16 sequence

plt.subplot(2, 2, 1)

plt.plot(
    U_shift[0],
    color=blue,
    linewidth=1
)

plt.title(
    "Shift Register (L = 16) Sequence",
    fontweight="bold"
)

plt.xlabel("Index")
plt.ylabel("X_i")

plt.grid(True)


# Shift L = 16 distribution

plt.subplot(2, 2, 2)

plt.hist(
    U_shift[0],
    bins=40,
    color=blue,
    edgecolor="black"
)

plt.title(
    "Shift Register (L = 16) Distribution",
    fontweight="bold"
)

plt.xlabel("X_i")
plt.ylabel("Frequency")

plt.grid(True)


# Shift L = 32 sequence

plt.subplot(2, 2, 3)

plt.plot(
    U_shift[1],
    color=green,
    linewidth=1
)

plt.title(
    "Shift Register (L = 32) Sequence",
    fontweight="bold"
)

plt.xlabel("Index")
plt.ylabel("X_i")

plt.grid(True)


# Shift L = 32 distribution

plt.subplot(2, 2, 4)

plt.hist(
    U_shift[1],
    bins=40,
    color=green,
    edgecolor="black"
)

plt.title(
    "Shift Register (L = 32) Distribution",
    fontweight="bold"
)

plt.xlabel("X_i")
plt.ylabel("Frequency")

plt.grid(True)


plt.tight_layout()
plt.show()


# ============================================================
# MERSENNE TWISTER PLOTS
# ============================================================

plt.figure(
    figsize=(11, 5),
    facecolor="white"
)


# Sequence

plt.subplot(1, 2, 1)

plt.plot(
    U_MT,
    color=pink,
    linewidth=1
)

plt.title(
    "Mersenne Twister Sequence",
    fontweight="bold"
)

plt.xlabel("Index")
plt.ylabel("U_i")

plt.grid(True)


# Distribution

plt.subplot(1, 2, 2)

plt.hist(
    U_MT,
    bins=40,
    color=pink,
    edgecolor="black"
)

plt.title(
    "Mersenne Twister Distribution",
    fontweight="bold"
)

plt.xlabel("U_i")
plt.ylabel("Frequency")

plt.grid(True)


plt.tight_layout()
plt.show()


# ============================================================
# RUN LENGTH DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(10, 6),
    facecolor="white"
)


plt.hist(
    run_lengths(U1),
    bins=30,
    color=orange,
    edgecolor="black",
    alpha=0.7,
    label="LCG1"
)

plt.hist(
    run_lengths(U2),
    bins=30,
    color=purple,
    edgecolor="black",
    alpha=0.7,
    label="LCG2"
)

plt.hist(
    run_lengths(U_shift[0]),
    bins=30,
    color=blue,
    edgecolor="black",
    alpha=0.7,
    label="Shift16"
)

plt.hist(
    run_lengths(U_shift[1]),
    bins=30,
    color=green,
    edgecolor="black",
    alpha=0.7,
    label="Shift32"
)

plt.hist(
    run_lengths(U_MT),
    bins=30,
    color=pink,
    edgecolor="black",
    alpha=0.7,
    label="MT"
)


plt.title(
    "Run Length Distribution",
    fontweight="bold"
)

plt.xlabel("Run Length (l)")
plt.ylabel("Frequency")

plt.legend()

plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# RUN LENGTH N(l) VS l
# ============================================================

datasets = [
    U1,
    U2,
    U_shift[0],
    U_shift[1],
    U_MT
]


colors = [
    orange,
    purple,
    blue,
    green,
    pink
]


labels = [
    "LCG1",
    "LCG2",
    "Shift16",
    "Shift32",
    "MT"
]


plt.figure(
    figsize=(12, 10),
    facecolor="white"
)


for i in range(len(datasets)):

    rl = run_lengths(datasets[i])


    # Count frequency of each run length
    l_vals, counts = np.unique(
        rl,
        return_counts=True
    )


    plt.subplot(3, 2, i + 1)


    plt.plot(
        l_vals,
        counts,
        "o-",
        color=colors[i],
        linewidth=1.5,
        label="Observed"
    )


    # Avoid division by zero
    l_safe = np.where(
        l_vals == 0,
        1,
        l_vals
    )


    # Theoretical ~1/l trend
    theo = (
        np.max(counts)
        / l_safe
    )


    plt.plot(
        l_vals,
        theo,
        "k--",
        linewidth=1.5,
        label="~1/l trend"
    )


    plt.title(
        f"N(l) vs l: {labels[i]}",
        fontweight="bold"
    )

    plt.xlabel("Run Length (l)")
    plt.ylabel("N(l)")

    plt.legend()

    plt.grid(True)


plt.tight_layout()
plt.show()


# ============================================================
# CUMULATIVE MEAN
# ============================================================

cm1 = np.cumsum(U1) / np.arange(1, len(U1) + 1)

cm2 = np.cumsum(U2) / np.arange(1, len(U2) + 1)

cm3 = np.cumsum(U_shift[0]) / np.arange(
    1,
    len(U_shift[0]) + 1
)

cm4 = np.cumsum(U_shift[1]) / np.arange(
    1,
    len(U_shift[1]) + 1
)

cm5 = np.cumsum(U_MT) / np.arange(
    1,
    len(U_MT) + 1
)


plt.figure(
    figsize=(10, 6),
    facecolor="white"
)


plt.plot(
    cm1,
    color=orange,
    linewidth=1.2,
    label="LCG1"
)

plt.plot(
    cm2,
    color=purple,
    linewidth=1.2,
    label="LCG2"
)

plt.plot(
    cm3,
    color=blue,
    linewidth=1.2,
    label="Shift16"
)

plt.plot(
    cm4,
    color=green,
    linewidth=1.2,
    label="Shift32"
)

plt.plot(
    cm5,
    color=pink,
    linewidth=1.2,
    label="MT"
)


# Ideal mean

plt.axhline(
    0.5,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="0.5"
)


plt.title(
    "Cumulative Mean Convergence",
    fontweight="bold"
)

plt.xlabel("N")
plt.ylabel("Mean")

plt.legend()

plt.grid(True)

plt.tight_layout()
plt.show()