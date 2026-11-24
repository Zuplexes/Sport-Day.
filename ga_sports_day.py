"""
Genetic Algorithm: แบ่งพนักงานเป็นทีมสำหรับงานกีฬาสี
ให้แต่ละทีมสมดุลกันในด้าน gender, department, weight, height, age

วิธีรัน (VS Code Terminal):
    pip install numpy pandas matplotlib openpyxl
    python ga_sports_day.py
วางไฟล์ HR_in_.csv ไว้โฟลเดอร์เดียวกับไฟล์นี้
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================= 1) ตั้งค่า (ปรับได้) =======================
CSV_PATH = "HR_in_.csv"
TEAM_NAMES = ["Team Pink", "Team Red"]      # เพิ่มชื่อได้ เช่น 4 สี -> จำนวนทีมเปลี่ยนตามอัตโนมัติ
POP_SIZE = 200          # จำนวนคำตอบในแต่ละรุ่น
GENERATIONS = 300       # จำนวนรุ่น
ELITE_SIZE = 10         # เก็บตัวเก่งสุดข้ามรุ่น
TOURNAMENT_K = 3        # ขนาด tournament
MUTATION_RATE = 0.5     # โอกาสกลายพันธุ์ต่อลูก
MUTATION_SWAPS = 2      # จำนวนคู่ที่สลับเมื่อเกิด mutation
W_GENDER, W_DEPT, W_PHYS = 2.0, 5.0, 1.0   # น้ำหนักความสำคัญของแต่ละเป้าหมาย
SEED = 42

rng = np.random.default_rng(SEED)
K = len(TEAM_NAMES)

# ======================= 2) โหลดและเตรียมข้อมูล =======================
df = pd.read_csv(CSV_PATH)
df["department"] = df["department"].str.strip()
df["gender"] = df["gender"].str.strip()
N = len(df)

# ตัวเลข -> Z-score (ให้ทุกตัวแปรสเกลเดียวกัน)
phys_cols = ["weight", "height", "age"]
Z = ((df[phys_cols] - df[phys_cols].mean()) / df[phys_cols].std()).to_numpy()   # (N, 3)
# หมวดหมู่ -> One-hot
G = pd.get_dummies(df["gender"], dtype=float).to_numpy()        # (N, 2)
D = pd.get_dummies(df["department"], dtype=float).to_numpy()    # (N, n_dept)

# ขนาดทีมเป้าหมาย (ต่างกันได้ไม่เกิน 1 คน ถ้า N หารไม่ลงตัว)
team_sizes = np.full(K, N // K)
team_sizes[: N % K] += 1
BASE = np.repeat(np.arange(K), team_sizes)     # [0,0,...,1,1,...]


# ======================= 3) Fitness (ยิ่งต่ำยิ่งดี) =======================
def get_penalty(pop):
    """pop: (P, N) เลขทีม 0..K-1 ของพนักงานแต่ละคน -> คืนค่า penalty ของแต่ละโครโมโซม"""
    onehot = (pop[:, :, None] == np.arange(K)).astype(float)    # (P, N, K)
    size = onehot.sum(axis=1)                                   # (P, K)
    share = size / N                                            # สัดส่วนขนาดทีม

    def cat_error(F):
        count = np.einsum("pnk,nf->pkf", onehot, F)             # จำนวนในแต่ละทีม
        ideal = share[:, :, None] * F.sum(axis=0)               # จำนวนที่ควรเป็น
        return np.abs(count - ideal).sum(axis=(1, 2)) / N

    gender_err = cat_error(G)
    dept_err = cat_error(D)

    # ค่าเฉลี่ย z-score ของแต่ละทีมควรใกล้ 0 (= เท่ากับค่าเฉลี่ยรวม)
    mean_z = np.einsum("pnk,nf->pkf", onehot, Z) / size[:, :, None]
    phys_err = np.abs(mean_z).mean(axis=(1, 2))

    return W_GENDER * gender_err + W_DEPT * dept_err + W_PHYS * phys_err


# ======================= 4) ตัวดำเนินการ GA =======================
def init_population():
    return np.array([rng.permutation(BASE) for _ in range(POP_SIZE)])


def tournament(pop, pen):
    idx = rng.choice(len(pop), TOURNAMENT_K, replace=False)
    return pop[idx[np.argmin(pen[idx])]]


def repair(c):
    """ซ่อมให้จำนวนคนในแต่ละทีมตรงเป้าหมายเสมอ"""
    counts = np.bincount(c, minlength=K)
    for t in range(K):
        while counts[t] > team_sizes[t]:
            under = np.where(counts < team_sizes)[0][0]
            i = rng.choice(np.where(c == t)[0])
            c[i] = under
            counts[t] -= 1
            counts[under] += 1
    return c


def crossover(p1, p2):
    """Uniform crossover: แต่ละคนสุ่มรับทีมจากพ่อหรือแม่ แล้วซ่อมขนาดทีม"""
    child = np.where(rng.random(N) < 0.5, p1, p2)
    return repair(child.copy())


def mutate(c):
    """Swap mutation: สลับทีมของ 2 คนที่อยู่คนละทีม (ขนาดทีมไม่เปลี่ยน)"""
    for _ in range(MUTATION_SWAPS):
        a = rng.integers(N)
        others = np.where(c != c[a])[0]
        b = rng.choice(others)
        c[a], c[b] = c[b], c[a]
    return c


# ======================= 5) วนรุ่น =======================
pop = init_population()
history = []

for gen in range(GENERATIONS):
    pen = get_penalty(pop)
    history.append(pen.min())

    order = np.argsort(pen)
    new_pop = [pop[i].copy() for i in order[:ELITE_SIZE]]       # Elitism

    while len(new_pop) < POP_SIZE:
        child = crossover(tournament(pop, pen), tournament(pop, pen))
        if rng.random() < MUTATION_RATE:
            child = mutate(child)
        new_pop.append(child)
    pop = np.array(new_pop)

final_pen = get_penalty(pop)
best = pop[np.argmin(final_pen)]
history.append(final_pen.min())

# ======================= 6) ผลลัพธ์ =======================
df["team"] = [TEAM_NAMES[t] for t in best]
df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(1)
df.to_csv("HR_sports_day_teams.csv", index=False, encoding="utf-8-sig")

summary = df.groupby("team").agg(
    total=("team", "count"),
    male=("gender", lambda x: (x == "Male").sum()),
    female=("gender", lambda x: (x == "Female").sum()),
    weight_mean=("weight", "mean"), weight_sd=("weight", "std"),
    height_mean=("height", "mean"), height_sd=("height", "std"),
    age_mean=("age", "mean"), age_sd=("age", "std"),
).round(2)
dept_table = pd.crosstab(df["department"], df["team"])

with pd.ExcelWriter("HR_sports_day_teams.xlsx") as w:
    df.to_excel(w, sheet_name="All Employees", index=False)
    summary.to_excel(w, sheet_name="Summary")
    dept_table.to_excel(w, sheet_name="Department")

print(f"Penalty: Gen 0 = {history[0]:.4f} -> Final = {history[-1]:.4f}  (lower is better)\n")
print("=== Summary per team ===\n", summary.T, "\n")
print("=== Department x Team ===\n", dept_table)

# ======================= 7) กราฟ =======================
colors = ["#ff69b4", "#e74c3c", "#4361EE", "#06D6A0", "#FFB703", "#8338EC"][:K]
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))

ax[0].plot(history, lw=2.2, color="#ff69b4")
ax[0].set_title("1. GA Convergence (lower = better)")
ax[0].set_xlabel("Generation")
ax[0].set_ylabel("Penalty")

dept_table.plot(kind="bar", ax=ax[1], color=colors, title="2. Department split", rot=0)
ax[1].set_ylabel("Employees")

df.groupby("team")[phys_cols].mean().T.plot(
    kind="bar", ax=ax[2], color=colors, title="3. Mean weight / height / age", rot=0)
ax[2].set_ylabel("Mean")

plt.tight_layout()
plt.savefig("ga_result.png", dpi=150)
plt.show()
