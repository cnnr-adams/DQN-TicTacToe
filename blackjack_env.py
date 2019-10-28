from random import randint
import numpy as np
deck = []
DEALER_WINS = -1
DRAW = 0
NO_WIN = -2
PLAYER_WINS = 1
for i in range(2, 10):
    for _ in range(4):
        deck.append(([i], str(i)))
for _ in range(4):
    deck.append(([1, 11], 'Ace'))
for _ in range(16):
    deck.append(([10], 'Ten/Jack/Queen/King'))

card_set = {k: i for i, k in enumerate([str(i) for i in range(2, 10)] + ['Ace', 'Ten/Jack/Queen/King'])}


class BlackJack:
    def __init__(self):
        pass

    def get_sum(self, deck):
        sums = []
        for card in deck:
            vals = card[0]
            if len(vals) == 1:
                for i in range(len(sums)):
                    sums[i] += vals[0]
                if len(sums) == 0:
                    sums.append(vals[0])
            else:
                for i, val in enumerate(list(sums)):
                    sums[i] += vals[0]
                    sums.append(val + vals[1])
        if len([s for s in sums if s <= 21]) == 0:
            return 22
        else:
            return max([s for s in sums if s <= 21])

    def random_card(self):
        selection = randint(0, len(self.current_deck) - 1)
        val = self.current_deck[selection]
        del self.current_deck[selection]
        return val

    def start_game(self):
        self.current_deck = list(deck)
        self.dealer_cards = [self.random_card()]
        self.user_cards = [self.random_card(), self.random_card()]
        return self.win_condition(False)

    def play_hit(self):
        self.user_cards.append(self.random_card())
        return self.win_condition(False)

    def play_pass(self):
        while self.get_sum(self.dealer_cards) < 17:
            self.dealer_cards.append(self.random_card())
        return self.win_condition(True)

    def win_condition(self, final):
        dealer_sum = self.get_sum(self.dealer_cards)
        player_sum = self.get_sum(self.user_cards)
        if dealer_sum == 21:
            return DEALER_WINS
        elif player_sum > 21:
            return DEALER_WINS
        elif dealer_sum > 21:
            return PLAYER_WINS
        elif final and player_sum > dealer_sum:
            return PLAYER_WINS
        elif final and dealer_sum > player_sum:
            return DEALER_WINS
        elif final:
            return DRAW
        else:
            return NO_WIN

    def get_state(self):
        dealer_card_1 = self.dealer_cards[0]
        dc = np.zeros(len(card_set))
        dc[card_set[dealer_card_1[1]]] = 1
        state = []
        for card in self.user_cards:
            card_zeros = np.zeros(len(card_set))
            card_zeros[card_set[card[1]]] = 1
            state.append(card_zeros)
        while len(state) < 12:
            state.append(np.zeros(len(card_set)))
        state.append(dc)
        return np.array(state)
