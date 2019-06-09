import numpy as np

X = (1, 0)
O = (0, 1)
NONE = (0, 0)


class TicTacToe:
    def __init__(self):
        self.turn = X
        # X,O pairs, 0 is none
        self.board = [NONE for _ in range(9)]

    def get_state(self):
        return self.board

    # -1 means you can't place there
    def place(self, loc):
        if self.board[loc] != NONE:
            return -1

        if self.turn == X:
            self.board[loc] = X
            self.turn = O
        else:
            self.board[loc] = O
            self.turn = X

    def print_board(self):
        print("""%s|%s|%s
                 --------
                 %s|%s|%s
                 --------
                 %s|%s|%s""" % tuple(['X' if loc == X else 'O' for loc in self.board]))

    # 1 is X win, -1 is O win, 0 is no win
    def winner(self):
        # reshape the board to make it easier to determine winner
        check_board = np.reshape([1 if loc == X else -1 for loc in self.board], (3, 3))

        # rows
        for row in check_board:
            if sum(row) == len(row):
                return 1
            elif sum(row) == -len(row):
                return -1

        # cols
        for j in range(len(check_board)):
            s = 0
            for i in range(len(check_board)):
                s += check_board[i][j]
            if s == len(check_board):
                return 1
            elif s == -len(check_board):
                return -1

        rd = 0
        ld = 0

        # diag
        for i in range(len(check_board)):
            rd += check_board[i][i]
            ld += check_board[i][len(check_board) - i - 1]

        if rd == len(check_board) or ld == len(check_board):
            return 1
        elif rd == -len(check_board) or ld == -len(check_board):
            return -1

        return 0
