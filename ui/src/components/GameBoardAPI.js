import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import Card from './Card';
import PlayerGrid from './PlayerGrid';
import gameService from '../services/gameService';

// Réutiliser les styles du GameBoard existant
const GameContainer = styled.div`
  height: calc(100vh - 60px - 2rem - 1rem - 1.6rem); /* Header principal (60px) + padding container (2rem) + gap interne (1rem) + padding GameHeader (1.6rem) */
  padding: 1rem;
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 1rem;
  overflow: hidden;
`;

const GameHeader = styled(motion.div)`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.8rem 1.5rem;
  background: rgba(0, 0, 0, 0.3);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  flex-wrap: wrap;
  gap: 1rem;
  flex-shrink: 0;
`;

const GameInfo = styled.div`
  color: white;
  font-size: 1.2rem;
  font-weight: 600;
`;

const RoundInfo = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.9rem;
  margin-top: 0.25rem;
`;

const CurrentPlayerIndicator = styled(motion.div)`
  background: linear-gradient(45deg, #667eea, #764ba2);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 600;
  text-align: center;
  min-width: 200px;
`;

const MainGameArea = styled.div`
  flex: 1;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  grid-template-rows: 1fr auto 1fr;
  gap: 1rem;
  min-height: 0; /* Permet au contenu de se redimensionner */
  position: relative;
`;

const PlayerArea = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 1rem;
  border: 2px solid ${props => props.isCurrentPlayer ? 
    'rgba(255, 107, 107, 0.6)' : 
    'rgba(255, 255, 255, 0.2)'};
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
`;

const PlayerHeader = styled.div`
  text-align: center;
  width: 100%;
`;

const PlayerName = styled.h3`
  color: white;
  font-size: 0.9rem;
  font-weight: 600;
  margin: 0 0 0.25rem 0;
`;

const PlayerScore = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.8rem;
`;

const CenterArea = styled.div`
  grid-column: 2;
  grid-row: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  min-width: 220px;
  padding: 1rem;
`;

const DeckArea = styled.div`
  display: flex;
  gap: 1rem;
  margin-bottom: 1rem;
`;

const DeckStack = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
  cursor: pointer;
`;

const DeckLabel = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 0.8rem;
  font-weight: 500;
`;

const DeckCount = styled.div`
  color: rgba(255, 255, 255, 0.6);
  font-size: 0.7rem;
`;

const LastActionIndicator = styled(motion.div)`
  background: rgba(0, 0, 0, 0.4);
  color: rgba(255, 255, 255, 0.9);
  padding: 0.75rem 1rem;
  border-radius: 8px;
  text-align: center;
  font-size: 0.85rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  max-width: 280px;
  word-wrap: break-word;
`;

const GameStats = styled(motion.div)`
  background: rgba(0, 0, 0, 0.2);
  border-radius: 8px;
  padding: 0.75rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  font-size: 0.85rem;
  color: rgba(255, 255, 255, 0.8);
  text-align: center;
`;

const GameControlsExtended = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: center;
`;

const ControlButton = styled(motion.button)`
  background: ${props => props.primary ? 
    'linear-gradient(45deg, #667eea, #764ba2)' : 
    'rgba(255, 255, 255, 0.1)'};
  border: 2px solid ${props => props.primary ? 
    'transparent' : 
    'rgba(255, 255, 255, 0.2)'};
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  backdrop-filter: blur(10px);
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:not(:disabled):hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
  }
`;

// Position des joueurs selon leur nombre - Disposition en coins mais ordre de tour en cercle
const getPlayerPosition = (index, totalPlayers) => {
  const positions = {
    2: [
      { gridColumn: '1', gridRow: '2' }, // Gauche (Joueur 1)
      { gridColumn: '3', gridRow: '2' }  // Droite (Joueur 2)
    ],
    3: [
      { gridColumn: '1', gridRow: '1' }, // Haut gauche (Joueur 1) 
      { gridColumn: '3', gridRow: '1' }, // Haut droite (Joueur 2)
      { gridColumn: '1', gridRow: '3' }  // Bas gauche (Joueur 3) - ordre en cercle
    ],
    4: [
      { gridColumn: '1', gridRow: '1' }, // Haut gauche (Joueur 1)
      { gridColumn: '3', gridRow: '1' }, // Haut droite (Joueur 2) 
      { gridColumn: '3', gridRow: '3' }, // Bas droite (Joueur 3)
      { gridColumn: '1', gridRow: '3' }  // Bas gauche (Joueur 4) - ordre en cercle
    ]
  };
  
  return positions[totalPlayers]?.[index] || { gridColumn: '1', gridRow: '1' };
};

const GameBoardAPI = ({ gameId, onGameEnd, onBackToMenu }) => {
  const [gameState, setGameState] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isPaused, setIsPaused] = useState(false);


  // Charger l'état initial du jeu
  useEffect(() => {
    const loadGameState = async () => {
      try {
        setLoading(true);
        setError(null);
        const state = await gameService.getGameState(gameId);
        setGameState(state);
      } catch (err) {
        console.error('Erreur lors du chargement de l\'état:', err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    if (gameId) {
      loadGameState();
    }
  }, [gameId]);

  // Auto-refresh de l'état du jeu
  useEffect(() => {
    const stepGame = async () => {
      if (!gameId || isPaused) return;

      try {
        const result = await gameService.stepGame(gameId);
        setGameState(result.state);
        
        if (result.state.finished) {
          // Partie terminée
          setTimeout(() => {
            onGameEnd({
              winner: result.state.players.reduce((min, player) => 
                player.score < min.score ? player : min
              ),
              players: result.state.players,
              rounds: result.state.round,
              scoreboard: result.state.scoreboard
            });
          }, 2000);
        }
      } catch (err) {
        console.error('Erreur lors du tour:', err);
        setError(err.message);
      }
    };

    if (!isPaused && gameId && gameState && !gameState.finished) {
      const interval = setInterval(() => {
        stepGame();
      }, 2000); // Chaque 2 secondes

      return () => clearInterval(interval);
    }
  }, [gameId, gameState, isPaused, onGameEnd]);

  const autoPlay = async (steps = 10) => {
    if (!gameId) return;

    try {
      const result = await gameService.autoPlay(gameId, steps);
      setGameState(result.state);
      
      // Vérifier si la partie est terminée après l'auto-play
      if (result.state.finished) {
        setTimeout(() => {
          onGameEnd({
            winner: result.state.players.reduce((min, player) => 
              player.score < min.score ? player : min
            ),
            players: result.state.players,
            rounds: result.state.round,
            scoreboard: result.state.scoreboard
          });
        }, 1000); // Délai plus court pour une meilleure réactivité
      }
    } catch (err) {
      console.error('Erreur lors de l\'auto-play:', err);
      setError(err.message);
    }
  };

  if (loading) {
    return (
      <GameContainer>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          color: 'white',
          fontSize: '1.5rem'
        }}>
          🔄 Chargement de la partie...
        </div>
      </GameContainer>
    );
  }

  if (error || !gameState) {
    return (
      <GameContainer>
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column',
          justifyContent: 'center', 
          alignItems: 'center', 
          height: '100vh',
          color: 'white',
          textAlign: 'center',
          gap: '1rem'
        }}>
          <h2>❌ Erreur de connexion</h2>
          <p>{error}</p>
          <ControlButton onClick={onBackToMenu}>
            ← Retour au menu
          </ControlButton>
        </div>
      </GameContainer>
    );
  }

  return (
    <GameContainer>
      <GameHeader
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        <GameInfo>
          <div>🎮 Skyjo AI - Partie Python</div>
          <RoundInfo>
            Manche {gameState.round} • Tour {gameState.turn}
          </RoundInfo>
        </GameInfo>
        
        <CurrentPlayerIndicator
          key={gameState.current_player}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          Tour de: {gameState.players[gameState.current_player]?.name}
        </CurrentPlayerIndicator>
        
        <GameControlsExtended>
          <ControlButton onClick={() => setIsPaused(!isPaused)}>
            {isPaused ? '▶️ Reprendre' : '⏸️ Pause'}
          </ControlButton>
          
          <ControlButton onClick={() => autoPlay(5)}>
            ⏩ 5 Tours
          </ControlButton>
        </GameControlsExtended>
      </GameHeader>

      <MainGameArea>
        {gameState.players.map((player, index) => {
          const position = getPlayerPosition(index, gameState.players.length);
          
          return (
            <PlayerArea
              key={player.id}
              style={position}
              isCurrentPlayer={index === gameState.current_player}
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: index * 0.1 }}
            >
              <PlayerHeader>
                <PlayerName>{player.name}</PlayerName>
                <PlayerScore>Score: {player.score} pts</PlayerScore>
              </PlayerHeader>
              
              <PlayerGrid
                player={player}
                isCurrentPlayer={index === gameState.current_player}
                readOnly={true} // Pas d'interaction en mode API
              />
            </PlayerArea>
          );
        })}

        <CenterArea>
          <DeckArea>
            <DeckStack
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Card 
                value={null} 
                revealed={false}
                size="large"
                noAnimation={true}
              />
              <DeckLabel>Pioche</DeckLabel>
              <DeckCount>{gameState.deck_count} cartes</DeckCount>
            </DeckStack>

            <DeckStack
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Card 
                key={`discard-${gameState.discard_pile.length}`}
                value={gameState.discard_pile[gameState.discard_pile.length - 1]?.value || 0}
                revealed={true}
                size="large"
                noAnimation={true}
              />
              <DeckLabel>Défausse</DeckLabel>
              <DeckCount>{gameState.discard_pile.length} cartes</DeckCount>
            </DeckStack>
          </DeckArea>

          {gameState.last_action && (
            <LastActionIndicator
              key={gameState.turn}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.4 }}
            >
              {isPaused ? '⏸️ PAUSE | ' : ''}{gameState.last_action}
            </LastActionIndicator>
          )}

          <GameStats
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            <div style={{ textAlign: 'center' }}>
              📋 Défausse: {gameState.discard_pile.slice(-5).map(card => card.value).join(' → ')}
            </div>
          </GameStats>
        </CenterArea>
      </MainGameArea>
    </GameContainer>
  );
};

export default GameBoardAPI; 