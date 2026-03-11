import matplotlib.pyplot as plt
import csv

x, y, z = [], [], []

with open("build/trajectory.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        x.append(float(row["x"]))
        y.append(float(row["y"]))
        z.append(float(row["z"]))

plt.figure(figsize=(10, 6))
plt.plot(x, z, label="Trajectory")
plt.xlabel("X")
plt.ylabel("Z")
plt.title("VIO Trajectory (Top Down View)")
plt.legend()
plt.savefig("trajectory.png")
plt.switch_backend('TkAgg')
plt.show()
print("Plot saved to trajectory.png")