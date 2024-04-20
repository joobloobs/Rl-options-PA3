import options as opt
from matplotlib import pyplot as plt
import numpy as np


movements = ["down", "up", "right", "left", "pickup", "putdown"]
localizations = [(0, 0), (0, 4), (4, 0), (4, 3)]
colors = ["red", "green", "yellow", "blue"]


def plot_Q(options):
    if len(options) ==2:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.ravel()
    print(axs)
    for i, option in enumerate(options):
        ax=axs[i]
        passenger_loc = 0
        if isinstance(option, opt.GoToOption):
            ax.set_title(f"policy of option to go to {option.location}")
        elif isinstance(option, opt.GetOption):
            ax.set_title("policy of option to get passenger from position (0,0)")
        elif isinstance(option, opt.PutOption):
            ax.set_title("policy of option to put passenger to destination (4, 0)")
            passenger_loc = 4
        ax.grid()
        states = []
        for row in range(5):
            for col in range(5):
                explicit_state = {"passenger_loc": passenger_loc, "dest": 3, "taxi_row": row, "taxi_col": col}
                states.append(to_implicit(explicit_state))

        q_values = option.q_values[states, :4]
        def x_direct(a):
            if movements[a] in ["up", "down"]:
                return 0
            return 1 if movements[a] == "right" else -1
        def y_direct(a):
            if movements[a] in ["right", "left"]:
                return 0
            return 1 if movements[a] == "up" else -1
        policy = q_values.argmax(-1).reshape((5, 5))[::-1]
        best_values = q_values.max(-1).reshape((5, 5))[::-1]
        policyx = np.vectorize(x_direct)(policy)
        policyy = np.vectorize(y_direct)(policy)
        idx = np.indices(policy.shape)
        ax.pcolor(best_values)
        ax.plot([2, 2], [5, 3], color='red', linewidth=2)
        ax.plot([1, 1], [0, 2], color='red', linewidth=2)
        ax.plot([3, 3], [0, 2], color='red', linewidth=2)
        ax.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
    plt.show()


def plot_global_Q(q_values):
    for dest in range(4):
        fig, ax = plt.subplots(figsize=(1, 1))
        plt.title("q values when passenger in taxi and dest is {}".format(colors[dest]))
        plt.grid()
        states = []
        for row in range(5):
            for col in range(5):
                explicit_state = {"passenger_loc": 4, "dest": dest, "taxi_row": row, "taxi_col": col}
                states.append(to_implicit(explicit_state))

        policy = q_values[states].argmax(-1).reshape((5, 5))[::-1]
        best_values = q_values[states].max(-1).reshape((5, 5))[::-1]
        plt.pcolor(best_values)
        plt.colorbar()
        plt.plot([2, 2], [5, 3], color='red', linewidth=2)
        plt.plot([1, 1], [0, 2], color='red', linewidth=2)
        plt.plot([3, 3], [0, 2], color='red', linewidth=2)
        for row in range(4):
            for col in range(5):
                if policy[row][col] >= 6:
                    crcl = plt.Circle((row-0.5,col+0.5), 0.25, color=colors[policy[row][col]-6], visible=True)
                    ax.add_artist(crcl)
        plt.show()


def extract_info(state: int):
    dest = state % 4
    state = state // 4
    passenger_loc = state % 5
    state //= 5
    taxi_col = state % 5
    state //= 5
    taxi_row = state
    return {"dest": dest, "passenger_loc": passenger_loc, "taxi_row": taxi_row, "taxi_col": taxi_col}


def to_implicit(explicit_state):
    return (((explicit_state["taxi_row"] * 5) + explicit_state["taxi_col"]) * 5 + explicit_state["passenger_loc"]) * 4 + explicit_state["dest"]


def render_env(state: int):
    explicit_state = extract_info(state)
    fig, ax = plt.subplots()
    grid = np.zeros((5, 5))
    ax.imshow(grid)
    fig.show()


