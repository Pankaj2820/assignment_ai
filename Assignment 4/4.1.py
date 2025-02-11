import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Define values of n
n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    #Roll two dice and add to get the sum
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    s = dice1 + dice2

    # Step 2: Compute histogram
    h, h2 = np.histogram(s, bins=range(2, 14))

    # Step 3: Plot histogram

    plt.bar(h2[:-1], h / n, align='center')
    plt.xlabel('Sum of Two Dice')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Two Dice Sums (n={n})')
    plt.xticks(range(2, 13))
    plt.show()

#Adding the result at the top of each bar
for x, y in zip((h2[:-1], h / n)):
    plt.text(x, y, f"{y*100:.1f}%", ha='center', va='bottom')
