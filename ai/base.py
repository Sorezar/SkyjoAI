class BaseAI:
    
    def initial_flip(self):
        """
        Décide de la position des deux premières cartes à retourner.
        Retourne une liste de deux tuples (i, j)
        """
        raise NotImplementedError
    
    def choose_source(self, grid, discard, other_p_grids):
        """
        Décide de la source de la carte à jouer.
        
        Args:
            grid: Grille du joueur actuel
            discard: Liste des cartes défaussées
            other_p_grids: Liste des grilles des autres joueurs
            
        Returns:
            str: 'D' pour défausse ou 'P' pour pioche
        """
        raise NotImplementedError
    
    def choose_keep(self, card, grid, other_p_grids):
        """
        Décide de garder ou non la carte piochée.
        
        Args:
            card: Carte piochée
            grid: Grille du joueur actuel
            other_p_grids: Liste des grilles des autres joueurs
            
        Returns:
            bool: True pour garder la carte, False pour la défausser
        """
        raise NotImplementedError
    
    def choose_position(self, card, grid, other_p_grids):
        """
        Décide de la position de la carte à jouer.
        
        Args:
            card: Carte à placer
            grid: Grille du joueur actuel
            other_p_grids: Liste des grilles des autres joueurs
            
        Returns:
            tuple: (i, j) position où placer la carte
        """
        raise NotImplementedError
    
    def choose_reveal(self, grid):
        """
        Décide de la carte à révéler.
        
        Args:
            grid: Grille du joueur actuel
            
        Returns:
            tuple: (i, j) position de la carte à révéler
        """
        raise NotImplementedError