from src.imports import read_eye_data
import matplotlib.pyplot as plt


def plot_eye_data(x,y,iTrial):
    fig, ax = plt.subplots()
    ax.plot(x[iTrial,:]-x[iTrial,:].mean())
    ax.plot(y[iTrial,:]-y[iTrial,:].mean())
    plt.show()

