"""
Name: Mojmir
Version: 1.1
Game Description: RL TimeTrial: Monaco - A 2D top-down racing game where a PPO agent learns to drive.
New to this Version:
- Agent vs Player ("Duel Mode") with custom car sprite skins.
- Dynamic ghost replay for high scores.
- Fixed 0-dimensional array crash in model observation.
"""

from game import Game

if __name__ == "__main__":
    game = Game()
    game.run()
