from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
from keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
from dqn.dqn import DQN
from tictactoe_env import TicTacToe


def model_constructor():
    model = Sequential()
    # hidden layer
    model.add(Dense(18, input_shape=(18,)))
    model.add(Dense(12))
    model.add(Dense(9))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


dqnX = DQN(model_constructor, 9)
dqnO = DQN(model_constructor, 9)


def ai_ai_game():
    false_moves_X = 0
    false_moves_O = 0
    env = TicTacToe()
    winner = env.winner()
    while winner == 0:
        ai_reward = 0
        if env.get_turn() == 'X':
            res = -1
            while res == -1:
                move = dqnX.determine_action(env.get_state(), ai_reward)
                res = env.place(move)
                if res == -1:
                    ai_reward = -1
                    dqnX.determine_action(env.get_state(), ai_reward, terminal_state=True)
                    dqnO.determine_action(env.get_state(), 1, terminal_state=True)
                    return 0
                else:
                    ai_reward = 0
        else:
            res = -1
            while res == -1:
                move = dqnO.determine_action(env.get_state(), ai_reward)
                res = env.place(move)
                if res == -1:
                    ai_reward = -1
                    dqnO.determine_action(env.get_state(), ai_reward, terminal_state=True)
                    dqnX.determine_action(env.get_state(), 1, terminal_state=True)
                    return 0
                else:
                    ai_reward = 0
        winner = env.winner()
    rewardX = 0
    rewardO = 0
    if winner == -2:
        rewardX = 1
        rewardO = 1
    elif winner == 1:
        rewardX = 2
        rewardO = -2
    else:
        rewardX = -2
        rewardO = 2
    dqnX.determine_action(env.get_state(), rewardX, terminal_state=True)
    dqnO.determine_action(env.get_state(), rewardO, terminal_state=True)
    return winner


def random_ai_game(ai_team='X'):
    false_moves = 0
    env = TicTacToe()
    winner = env.winner()
    while winner == 0:
        # print("Board:")
        # env.print_board()
        ai_reward = 0
        if env.get_turn() == ai_team:
            res = -1
            while res == -1:
                # print("AI Turn")
                move = dqnX.determine_action(env.get_state(), ai_reward)
                res = env.place(move)
                if res == -1:
                    ai_reward = -1
                    dqnX.determine_action(env.get_state(), ai_reward, terminal_state=True)
                    return 0
                else:
                    ai_reward = 0
        else:
            res = -1
            while res == -1:
                res = env.place(random.randint(0, 8))
        winner = env.winner()
    reward = 0
    if winner == -2:
        reward = 0
    elif winner == 1 and ai_team == 'X' or winner == -1 and ai_team == 'O':
        reward = 1
    else:
        reward = -1
    dqnX.determine_action(env.get_state(), reward, terminal_state=True)
    return winner


def run_game(ai_team='X'):
    env = TicTacToe()
    winner = env.winner()
    while winner == 0:
        print("Board:")
        env.print_board()
        ai_reward = 0
        if env.get_turn() == ai_team:
            res = -1
            while res == -1:
                print("AI Turn")
                move = dqnX.determine_action(env.get_state(), ai_reward)
                print(dqnX.q_predictor.predict(env.get_state()))
                res = env.place(move)
                if res == -1:
                    print("AI goofed")
                    ai_reward = -1

        else:
            res = -1
            while res == -1:
                move = input("Input place position from 0-8: ")
                res = env.place(int(move))
                if res == -1:
                    print("Bad move.")
        winner = env.winner()
    reward = 0
    if winner == -2:
        reward = 0
        print("Tie.")
    elif winner == 1 and ai_team == 'X' or winner == -1 and ai_team == 'O':
        reward = 0
        print("AI Win!")
    else:
        reward = 0
        print("Ai lost")
    dqnX.determine_action(env.get_state(), reward, terminal_state=True)


# dqnX.force_epsilon(1)
# dqnO.force_epsilon(1)
ai_w_l = []
ai_t_l = []
ai_l_l = []
ai_fm_l = []
fm_l = []
for i in range(1000):
    ai_w = 0
    ai_t = 0
    ai_l = 0
    ai_fm = 0
    fm = 0
    print("RUN:", i)
    for i in range(100):
        res = random_ai_game()
        if res == 1:
            ai_w += 1
        elif res == -2:
            ai_t += 1
        elif res == 0:
            ai_fm += 1
        else:
            ai_l += 1
    ai_w_l.append(ai_w)
    ai_t_l.append(ai_t)
    ai_l_l.append(ai_l)
    ai_fm_l.append(ai_fm)
    print("W/T/L/F", ai_w, ai_t, ai_l, ai_fm)
    dqnX.train()
    dqnO.train()

dqnX.force_epsilon(0)
ai_w = 0
ai_t = 0
ai_l = 0
ai_fm = 0
for i in range(5000):
    res = random_ai_game()
    if res == 1:
        ai_w += 1
    elif res == -2:
        ai_t += 1
    elif res == 0:
        ai_fm += 1
    else:
        ai_l += 1
print("W/T/L/F", ai_w, ai_t, ai_l, ai_fm)
# dqnO.force_epsilon(0)
# ai_w = 0
# ai_t = 0
# ai_l = 0
# fm = 0
# for i in range(100):
#     print("RUN:", i)
#     false_moves, res = ai_ai_game()
#     if res == 1:
#         ai_w += 1
#     elif res == -2:
#         ai_t += 1
#     else:
#         ai_l += 1
#     fm += false_moves

# print("W/T/L/F", ai_w, ai_t, ai_l)
plt.stackplot(range(1, len(ai_w_l)+1), [ai_w_l, ai_t_l, ai_l_l, ai_fm_l], labels=['X wins', 'ties', 'O wins', 'Misplaces'])
# plt.plot(ai_w_l, label='X wins')
# plt.plot(ai_t_l, label='ties')
# plt.plot(ai_l_l, label='O wins')
# plt.plot(ai_fm_l, label='Misplaces')
# plt.plot(fm_l, label='misplaces')
plt.legend()
plt.show()
dqnX.force_epsilon(0)
while True:
    # try:
    run_game()
    # except Exception as e:
    #    print(e)
    #    break
