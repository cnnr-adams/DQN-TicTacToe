from keras.models import Model, Sequential
from keras.layers import Dense, Reshape, Flatten, Conv2D
from keras.optimizers import SGD
import random
import matplotlib.pyplot as plt
from dqn.dqn import DQN
import blackjack_env


def model_constructor():
    model = Sequential()
    # model.add(Conv2D(filters=32, kernel_size=2, padding="same", input_shape=(3, 3, 3)))
    # model.add(Conv2D(filters=64, kernel_size=3, padding="same"))
    # model.add(Reshape((27,)))
    # hidden layer
    model.add(Dense(100, input_shape=(13, len(blackjack_env.card_set))))
    model.add(Dense(100))
    model.add(Reshape((100 * 13,)))
    model.add(Dense(2))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    # model.summary()
    return model


dqn = DQN(model_constructor, 2)

game = blackjack_env.BlackJack()


def play_game():
    winner = game.start_game()
    while winner == -2:
        move = dqn.determine_action(game.get_state(), 0)
        if move == 0:
            winner = game.play_pass()
        else:
            winner = game.play_hit()
    dqn.determine_action(game.get_state(), winner, terminal_state=True)
    return winner


ai_w_l = []
ai_t_l = []
ai_l_l = []
for i in range(200):
    ai_w = 0
    ai_t = 0
    ai_l = 0
    print("RUN:", i)
    for i in range(100):
        res = play_game()
        if res == 1:
            ai_w += 1
        elif res == 0:
            ai_t += 1
        else:
            ai_l += 1
    ai_w_l.append(ai_w)
    ai_t_l.append(ai_t)
    ai_l_l.append(ai_l)
    print("W/T/L", ai_w, ai_t, ai_l)
    dqn.train()

plt.stackplot(range(1, len(ai_w_l)+1), [ai_w_l, ai_t_l, ai_l_l], labels=['wins', 'ties', 'losses'])
plt.show()

for i in range(5):
    winner = game.start_game()
    while winner == -2:
        print("state:", game.user_cards, game.dealer_cards, "win chance:", dqn.win_chance(game.get_state()))
        move = dqn.determine_action(game.get_state(), 0)
        print("Move:", "pass" if move == 0 else "hit")
        if move == 0:
            winner = game.play_pass()
        else:
            winner = game.play_hit()
    dqn.determine_action(game.get_state(), winner, terminal_state=True)
    if winner == 1:
        print("ai wins")
    elif winner == 0:
        print("draw")
    else:
        print("loss")
