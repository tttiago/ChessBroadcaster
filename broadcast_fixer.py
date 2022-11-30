"""Handle the calls to correct the broadcasted moves and the clock times."""

from pynput import keyboard


class BroadcastFixer:
    def __init__(self, broadcast):
        self.UNDO_COMBINATION = {"u", str(broadcast.game_id + 1)}
        self.CLOCK_COMBINATION = {"y", str(broadcast.game_id + 1)}
        self.current = set()  # The currently active keys.
        self.broadcast = broadcast
        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)

    def _on_press(self, key):
        try:
            # Correct moves if both 'U' and the number of the board are pressed:
            if key.char in self.UNDO_COMBINATION:
                self.current.add(key.char)
                if all(k in self.current for k in self.UNDO_COMBINATION):
                    self._correct_moves()

            # Correct clock times if both 'Y' and the number of the board are pressed:
            if key.char in self.CLOCK_COMBINATION:
                self.current.add(key.char)
                if all(k in self.current for k in self.CLOCK_COMBINATION):
                    self._correct_clocks()
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            self.current.remove(key.char)
        except (KeyError, AttributeError):
            pass

    def _correct_moves(self):
        input("Edit the game you want and press Enter.")
        self.broadcast.correct_moves()

    def _correct_clocks(self):
        response = input(
            "Write White and Black's clock times ('h:mm:ss, h:mm:ss') and press Enter.\n"
        )
        self.broadcast.correct_clocks(response)
