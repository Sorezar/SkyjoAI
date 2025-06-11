import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';
import Card from './Card';

const GridContainer = styled(motion.div)`
  display: grid;
  grid-template-columns: ${props => `repeat(${props.columns || 4}, 1fr)`};
  grid-template-rows: repeat(3, 1fr);
  gap: 0.3rem;
  padding: 0.5rem;
  background: ${props => props.isCurrentPlayer ? 
    'rgba(255, 107, 107, 0.1)' : 
    'rgba(255, 255, 255, 0.05)'};
  border: 2px solid ${props => props.isCurrentPlayer ? 
    'rgba(255, 107, 107, 0.4)' : 
    'rgba(255, 255, 255, 0.1)'};
  border-radius: 8px;
  backdrop-filter: blur(10px);
  position: relative;
  transition: all 0.3s ease;
  max-width: 280px;
  
  &::before {
    content: '';
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    background: ${props => props.isCurrentPlayer ? 
      'linear-gradient(45deg, #ff6b6b, #feca57)' : 
      'transparent'};
    border-radius: 12px;
    z-index: -1;
    opacity: ${props => props.isCurrentPlayer ? 0.6 : 0};
    animation: ${props => props.isCurrentPlayer ? 'pulse 2s infinite' : 'none'};
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 0.3; }
  }
`;

const CardSlot = styled(motion.div)`
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const GridOverlay = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  color: white;
  font-size: 1.2rem;
  font-weight: 600;
  z-index: 10;
`;

const ScoreIndicator = styled(motion.div)`
  position: absolute;
  top: -10px;
  right: -10px;
  background: linear-gradient(45deg, #667eea, #764ba2);
  color: white;
  width: 30px;
  height: 30px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  font-weight: 700;
  z-index: 5;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
`;

const ColumnCompleteIndicator = styled(motion.div)`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: rgba(76, 217, 100, 0.9);
  color: white;
  padding: 0.3rem 0.6rem;
  border-radius: 20px;
  font-size: 0.7rem;
  font-weight: 600;
  z-index: 5;
  pointer-events: none;
`;

const PlayerGrid = ({ 
  player, 
  isCurrentPlayer = false, 
  onCardClick, 
  showScore = true,
  disabled = false 
}) => {
  // Calculer le nombre de colonnes dynamiquement
  const columns = player.grid.length > 0 ? player.grid[0].length : 4;
  
  // Calculer le score visible du joueur
  const calculateVisibleScore = () => {
    return player.grid.flat().reduce((total, card) => {
      return card.revealed ? total + card.value : total;
    }, 0);
  };

  // Vérifier si une colonne est complète (toutes les cartes révélées et même valeur)
  const checkColumnComplete = (colIndex) => {
    const column = player.grid.map(row => row[colIndex]);
    const allRevealed = column.every(card => card.revealed);
    if (!allRevealed) return false;
    
    const firstValue = column[0].value;
    return column.every(card => card.value === firstValue);
  };

  // Animation variants
  const gridVariants = {
    hidden: { scale: 0.8, opacity: 0 },
    visible: { 
      scale: 1, 
      opacity: 1,
      transition: { 
        duration: 0.5,
        staggerChildren: 0.05
      }
    },
    currentPlayer: {
      scale: 1.02,
      transition: { duration: 0.3 }
    }
  };

  const cardVariants = {
    hidden: { scale: 0, rotateY: 180 },
    visible: { 
      scale: 1, 
      rotateY: 0,
      transition: { duration: 0.6 }
    },
    hover: {
      y: -5,
      transition: { duration: 0.2 }
    }
  };

  const overlayVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 }
  };

  return (
    <GridContainer
      isCurrentPlayer={isCurrentPlayer}
      columns={columns}
      variants={gridVariants}
      initial="hidden"
      animate={isCurrentPlayer ? ["visible", "currentPlayer"] : "visible"}
    >
      {/* Score indicator */}
      {showScore && (
        <ScoreIndicator
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.5, duration: 0.3 }}
        >
          {calculateVisibleScore()}
        </ScoreIndicator>
      )}

      {/* Grid overlay pour joueur non-actif */}
      {disabled && (
        <GridOverlay
          variants={overlayVariants}
          initial="hidden"
          animate="visible"
        >
          En attente...
        </GridOverlay>
      )}

      {/* Cartes de la grille */}
      {player.grid.map((row, rowIndex) => 
        row.map((card, colIndex) => {
          const isColumnComplete = checkColumnComplete(colIndex);
          
          return (
            <CardSlot
              key={`${rowIndex}-${colIndex}`}
              variants={cardVariants}
              whileHover={!disabled && onCardClick ? "hover" : {}}
            >
              <Card
                value={card.value}
                revealed={card.revealed}
                onClick={!disabled && onCardClick ? 
                  () => onCardClick(rowIndex, colIndex) : 
                  undefined
                }
                isHighlighted={isCurrentPlayer && !card.revealed}
                isSelected={false}
              />
              
              {/* Indicateur de colonne complète */}
              {isColumnComplete && (
                <ColumnCompleteIndicator
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  Colonne !
                </ColumnCompleteIndicator>
              )}
            </CardSlot>
          );
        })
      )}
    </GridContainer>
  );
};

// Composant pour afficher la grille d'un adversaire avec moins de détails
export const CompactPlayerGrid = ({ player, isCurrentPlayer = false }) => {
  // Calculer le nombre de colonnes dynamiquement
  const columns = player.grid.length > 0 ? player.grid[0].length : 4;
  
  const CompactGrid = styled(motion.div)`
    display: grid;
    grid-template-columns: ${`repeat(${columns}, 1fr)`};
    grid-template-rows: repeat(3, 1fr);
    gap: 0.3rem;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid ${props => props.isCurrentPlayer ? 
      'rgba(255, 107, 107, 0.4)' : 
      'rgba(255, 255, 255, 0.1)'};
    border-radius: 8px;
    backdrop-filter: blur(10px);
  `;

  return (
    <CompactGrid isCurrentPlayer={isCurrentPlayer}>
      {player.grid.map((row, rowIndex) => 
        row.map((card, colIndex) => (
          <Card
            key={`${rowIndex}-${colIndex}`}
            value={card.value}
            revealed={card.revealed}
            size="small"
          />
        ))
      )}
    </CompactGrid>
  );
};

// Composant pour l'animation de distribution des cartes
export const GridSetupAnimation = ({ onComplete, playerCount = 4 }) => {
  const [currentPlayer, setCurrentPlayer] = React.useState(0);
  const [currentCard, setCurrentCard] = React.useState(0);
  
  React.useEffect(() => {
    const cardsPerPlayer = 12;
    const totalCards = playerCount * cardsPerPlayer;
    
    const interval = setInterval(() => {
      setCurrentCard(prev => {
        if (prev >= totalCards - 1) {
          clearInterval(interval);
          setTimeout(() => onComplete?.(), 500);
          return prev;
        }
        
        const nextCard = prev + 1;
        setCurrentPlayer(Math.floor(nextCard / cardsPerPlayer) % playerCount);
        return nextCard;
      });
    }, 100);
    
    return () => clearInterval(interval);
  }, [playerCount, onComplete]);

  return (
    <div style={{ 
      position: 'absolute', 
      top: '50%', 
      left: '50%', 
      transform: 'translate(-50%, -50%)',
      zIndex: 100 
    }}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
      >
        Distribution des cartes... {currentCard + 1}
      </motion.div>
    </div>
  );
};

export default PlayerGrid; 