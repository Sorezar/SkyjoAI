class BaseAI:
    def choose_action(self, player, top_discard, deck):
        """
        Décide du coup à jouer.
        Retourne un dictionnaire avec :
            - 'source': 'D' ou 'P'
            - 'position': (i, j)
        """
        raise NotImplementedError