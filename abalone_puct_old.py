import math
import random
import traceback

from abalone import Abalone
from abalone_neural_network import AbaloneNetwork


class PUCTNode:
    def __init__(self, move: tuple[int, int, int], P: float):
        self.move = move
        self.N = 0
        self.Q = 0
        self.P = P
        self.policy = None
        self.unvisited_children: list[PUCTNode] = []
        self.visited_children_keys: list = []
        self.fully_expanded: bool = False
        self.is_terminal: bool = False

    def addVisit(self, addQ: int, visits: int = 1):
        self.N += visits
        self.Q = self.Q + (addQ - self.Q) / self.N

    def expand(self, game: Abalone, model: AbaloneNetwork):
        if self.fully_expanded:
            raise Exception("tried to expand a non-leaf node")

        if self.is_terminal:
            raise Exception("tried to expand a terminal node")

        if game.status_arr != Abalone.ONGOING:
            self.is_terminal = True
            return

        self.policy, self.Q = model.forward(game)
        self.unvisited_children = [
            PUCTNode(move, self.policy[Abalone.encode_move(move)].item())
            for move in game.legal_moves()
        ]

        random.shuffle(self.unvisited_children)

    def visit_child(
        self, game: Abalone, model: AbaloneNetwork, tree_dict: dict
    ):
        child = self.unvisited_children.pop()

        if len(self.unvisited_children) <= 0:
            self.fully_expanded = True

        game.make_move(child.move)
        child_key = game.encode()
        if child_key not in tree_dict:
            child.expand(game, model)
            tree_dict[child_key] = child

        self.visited_children_keys.append(child_key)
        return child

    def isLeaf(self):
        return not self.fully_expanded or self.is_terminal


class PUCTPlayer:
    def __init__(
        self,
        max_leaf_explore: int = 1000,
        explore_constant: float = math.sqrt(2),
    ):
        self.model = AbaloneNetwork()
        self.max_leaf_explore = max_leaf_explore
        self.explore_constant = explore_constant

    def calc_PUCT(self, node: PUCTNode, parent_visits: int):
        return node.Q + self.explore_constant * node.P * (
            math.sqrt(parent_visits) / (node.N + 1)
        )

    def pick_child(self, current: PUCTNode, tree_dict):
        children = [
            tree_dict[child_key] for child_key in current.visited_children_keys
        ]
        return max(
            children,
            key=lambda child: self.calc_PUCT(child, current.N),
        )

    def select(self, tree_key: PUCTNode, tree_dict: dict, game: Abalone):
        temp: PUCTNode = tree_dict[tree_key]
        stack = [temp]

        while not temp.isLeaf():
            next = self.pick_child(temp, tree_dict)
            game.make_move(next.move)
            stack.append(next)
            temp = next

        return stack

    def expand(self, stack: list[PUCTNode], game: Abalone, tree_dict: dict):
        curr = stack[-1]
        if curr.is_terminal:
            return

        child = curr.visit_child(game, self.model, tree_dict)
        stack.append(child)

    def propagate(self, stack: list[PUCTNode], game: Abalone, visits: int = 1):
        game.undo_move(len(stack))

        temp = stack.pop()
        v = temp.Q
        temp.addVisit(v, visits=visits)

        while len(stack) > 0:
            temp = stack.pop()
            v = -v
            temp.addVisit(v, visits=visits)

    def getMove(self, original_game: Abalone):
        if original_game.status_arr != Abalone.ONGOING:
            return None

        tree_dict = {}
        game = original_game.copy()
        tree_key = game.encode()
        tree_dict[tree_key] = PUCTNode(-1, 0)
        tree_node = tree_dict[tree_key]

        tree_node.expand(game, self.model)

        for _ in range(self.max_leaf_explore):
            stack = self.select(tree_key, tree_dict, game)
            self.expand(stack, game, tree_dict)
            self.propagate(stack, game)

        children = [
            tree_dict[child_key]
            for child_key in tree_node.visited_children_keys
        ]
        best_child = max(
            children,
            key=lambda child: child.N,
        )

        return best_child.move


def main():
    """
    A simple main function for debugging purposes.
    Allows two human players to play Connect Four in the terminal.
    """
    game = Abalone()

    print("Welcome to Abalone!")
    print("Player 1 is RED (R) and Player 2 is YELLOW (Y).\n")

    computer_player = PUCTPlayer(
        explore_constant=1,
        max_leaf_explore=1000,
    )

    while game.status_arr == game.ONGOING:
        print(game)
        print(
            "\nCurrent Player:",
            "BLACK" if game.player == Abalone.BLACK else "WHITE",
        )

        if game.player == game.WHITE:
            move = computer_player.getMove(game)
        else:
            move = computer_player.getMove(game)
        print(game)

        if move not in game.legal_moves():
            print("Illegal move. Try again.")
            continue
        game.make_move(move)
        try:
            pass

        except ValueError:
            traceback.print_exc()
            print(
                f"Invalid input. Enter a number between 0 and {game.width - 1}."
            )
            break
        except IndexError:
            traceback.print_exc()
            print("Move out of bounds. Try again.")
            break

    print(game)
    if game.status_arr == game.BLACK:
        print("\nBLACK (Player 1) wins!")
    elif game.status_arr == game.WHITE:
        print("\nWHITE (Player 2) wins!")
    else:
        print("\nIt's a draw!")


if __name__ == "__main__":
    main()
