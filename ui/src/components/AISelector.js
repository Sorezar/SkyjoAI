import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import gameService from '../services/gameService';

const SelectorContainer = styled.div`
  padding: 1rem;
  height: calc(100vh - 60px - 6rem); /* Header (60px) + padding container (2rem) + margin header (1rem) + margin count (1rem) + padding controls (2rem) */
  display: flex;
  flex-direction: column;
  overflow: hidden;
  
  @media (max-width: 1024px) {
    padding: 0.8rem;
    height: calc(100vh - 60px - 4.8rem); /* Header + padding container (1.6rem) + margin header (1rem) + margin count (1rem) + padding controls (1.6rem) */
  }
  
  @media (max-width: 768px) {
    padding: 0.6rem;
    height: calc(100vh - 60px - 4.4rem); /* Header + padding container (1.2rem) + margin header (1rem) + margin count (1rem) + padding controls (1.6rem) */
  }
  
  @media (max-width: 480px) {
    padding: 0.5rem;
    height: calc(100vh - 60px - 4rem); /* Header + padding container (1rem) + margin header (1rem) + margin count (1rem) + padding controls (1rem) */
  }
`;

const Header = styled.div`
  flex-shrink: 0;
  margin-bottom: 1rem;
  text-align: center; /* Centrer le titre horizontalement */
`;

const Title = styled(motion.h2)`
  color: white;
  font-size: 1.8rem;
  font-weight: 700;
  margin-bottom: 0.3rem;
`;

const AIGrid = styled(motion.div)`
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
  flex: 1;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  min-height: 0; /* Permet à la grille de se redimensionner */
  
  @media (max-width: 1200px) {
    gap: 0.8rem;
    max-width: 1000px;
  }
  
  @media (max-width: 1024px) {
    grid-template-columns: repeat(2, 1fr);
    gap: 0.8rem;
    max-width: 700px;
  }
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
    gap: 0.6rem;
    max-width: 400px;
  }
  
  @media (max-width: 480px) {
    gap: 0.5rem;
    max-width: 350px;
  }
`;

const AICard = styled(motion.div)`
  background: ${props => props.selected ? 
    'linear-gradient(135deg, rgba(255, 107, 107, 0.3), rgba(254, 202, 87, 0.3))' : 
    'rgba(255, 255, 255, 0.1)'};
  border: 2px solid ${props => props.selected ? 
    'rgba(255, 107, 107, 0.6)' : 
    'rgba(255, 255, 255, 0.2)'};
  border-radius: 10px;
  padding: 1rem;
  cursor: pointer;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
  min-height: 150px;
  height: auto;
  display: flex;
  flex-direction: column;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    border-color: rgba(255, 255, 255, 0.4);
  }
  
  @media (max-width: 1200px) {
    padding: 0.8rem;
    min-height: 180px;
  }
  
  @media (max-width: 1024px) {
    padding: 0.8rem;
    min-height: 160px;
  }
  
  @media (max-width: 768px) {
    padding: 0.6rem;
    min-height: 140px;
  }
  
  @media (max-width: 480px) {
    padding: 0.5rem;
    min-height: 120px;
  }
`;

const AIHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 0.2rem;
`;

const AIName = styled.h3`
  color: #ffffff;
  margin: 0 0 0.5rem 0;
  font-size: 1.5rem;
  font-weight: 600;
  text-align: center;
  text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
  
  @media (max-width: 1200px) {
    font-size: 1.3rem;
  }
  
  @media (max-width: 1024px) {
    font-size: 1.2rem;
  }
  
  @media (max-width: 768px) {
    font-size: 1.1rem;
    margin: 0 0 0.3rem 0;
  }
  
  @media (max-width: 480px) {
    font-size: 1rem;
  }
`;

const AILevel = styled.div`
  background: ${props => {
    const rate = parseFloat(props.winRate);
    if (rate >= 40) return 'linear-gradient(45deg, #fd79a8, #fdcb6e)'; // Avancé - Gradient or/rose
    if (rate >= 20) return 'linear-gradient(45deg, #feca57, #ff9ff3)'; // Intermédiaire - Gradient jaune/rose
    return 'linear-gradient(45deg, #4ecdc4, #44a08d)'; // Débutant - Gradient bleu/vert
  }};
  color: white;
  padding: 0.4rem 0.8rem;
  border-radius: 12px;
  font-size: 0.9rem;
  font-weight: 600;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  margin: 0.3rem auto;
  width: 80%;
  text-align: center;
  
  @media (max-width: 1200px) {
    font-size: 0.85rem;
    padding: 0.35rem 0.6rem;
  }
  
  @media (max-width: 768px) {
    font-size: 0.8rem;
    padding: 0.3rem 0.5rem;
  }
`;

const AIStats = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 1rem;
`;

const StatItem = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  background: rgba(0, 0, 0, 0.2);
  padding: 0.5rem; /* Padding augmenté */
  border-radius: 8px; /* Bordure augmentée */
`;

const StatValue = styled.div`
  color: white;
  font-size: 1.3rem;
  font-weight: 700;
  
  @media (max-width: 1200px) {
    font-size: 1.2rem;
  }
  
  @media (max-width: 1024px) {
    font-size: 1.1rem;
  }
  
  @media (max-width: 768px) {
    font-size: 1rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.9rem;
  }
`;

const WinRateValue = styled.div`
  color: white;
  font-size: 1.3rem;
  font-weight: 700;
  
  @media (max-width: 1200px) {
    font-size: 1.2rem;
  }
  
  @media (max-width: 1024px) {
    font-size: 1.1rem;
  }
  
  @media (max-width: 768px) {
    font-size: 1rem;
  }
  
  @media (max-width: 480px) {
    font-size: 0.9rem;
  }
`;

const StatLabel = styled.div`
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.85rem;
`;

const SelectionIndicator = styled(motion.div)`
  position: absolute;
  top: 0rem;
  right: 2rem;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  background: linear-gradient(45deg, #00d2ff, #3a7bd5);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
`;

const SelectionContainer = styled.div`
  position: absolute;
  top: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  gap: 0.2rem;
`;

const ControlsContainer = styled.div`
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-shrink: 0;
  padding: 1rem 0;
  
  @media (max-width: 768px) {
    gap: 0.8rem;
    padding: 0.8rem 0;
  }
  
  @media (max-width: 480px) {
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.5rem 0;
  }
`;

const ControlButton = styled(motion.button)`
  background: ${props => props.primary ? 
    'linear-gradient(45deg, #667eea, #764ba2)' : 
    'rgba(255, 255, 255, 0.1)'};
  border: 2px solid ${props => props.primary ? 
    'transparent' : 
    'rgba(255, 255, 255, 0.2)'};
  color: white;
  padding: 0.8rem 1.5rem;
  border-radius: 10px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  backdrop-filter: blur(10px);
  min-width: 130px;
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  &:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  }
`;

const SelectedCount = styled(motion.div)`
  text-align: center;
  color: white;
  font-size: 1rem;
  margin-bottom: 1rem;
  flex-shrink: 0;
`;

const RemoveButton = styled.button`
  background: rgba(255, 107, 107, 0.8);
  border: none;
  border-radius: 50%;
  width: 24px;
  height: 24px;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.7rem;
  flex-shrink: 0;
`;



const AISelector = ({ gameConfig, onSelectAIs, onBack }) => {
  const [selectedAIs, setSelectedAIs] = useState([]);
  const [availableAIs, setAvailableAIs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const maxSelections = 4;

  const toggleAISelection = (ai) => {
    setSelectedAIs(prev => {
      if (prev.length >= maxSelections) {
        return prev;
      }
      return [...prev, { ...ai, uniqueId: `${ai.id}_${Date.now()}_${Math.random()}` }];
    });
  };
  
  const removeAISelection = (ai) => {
    setSelectedAIs(prev => {
      const indexToRemove = prev.findIndex(selected => selected.id === ai.id);
      if (indexToRemove !== -1) {
        return prev.filter((_, index) => index !== indexToRemove);
      }
      return prev;
    });
  };

  useEffect(() => {
    const loadAIs = async () => {
      try {
        setLoading(true);
        setError(null);
        
        const ais = await gameService.getAvailableAIs();
        setAvailableAIs(ais.slice(0, 8));
      } catch (err) {
        console.error('Erreur lors du chargement des IA:', err);
        setError("Erreur de connexion à l'API");
        setAvailableAIs([]);
      } finally {
        setLoading(false);
      }
    };

    loadAIs();
  }, []);

  const handleStartGame = async () => {
    if (selectedAIs.length >= 2) {
      try {
        const gameData = await gameService.createGame(selectedAIs);
        onSelectAIs(selectedAIs, gameData.game_id);
      } catch (err) {
        console.error('Erreur lors de la création de la partie:', err);
        setError("Erreur lors de la création de la partie");
      }
    }
  };

  const cardVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  return (
    <SelectorContainer>
      <Header>
        <Title
          initial={{ y: -30, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6 }}
        >
          Choisissez vos joueurs
        </Title>

        {error && (
          <div style={{ 
            color: '#ff6b6b', 
            textAlign: 'center', 
            margin: '1rem 0',
            padding: '0.5rem',
            background: 'rgba(255, 107, 107, 0.1)',
            borderRadius: '8px',
            border: '1px solid rgba(255, 107, 107, 0.3)'
          }}>
            {error}
          </div>
        )}
      </Header>

      <SelectedCount
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        {selectedAIs.length} / {maxSelections} joueurs IA sélectionnés
      </SelectedCount>

      {loading ? (
        <div style={{ textAlign: 'center', color: 'white', fontSize: '1.2rem', margin: '2rem 0' }}>
          🔄 Chargement des IA disponibles...
        </div>
      ) : availableAIs.length === 0 ? (
        <div style={{ 
          textAlign: 'center', 
          color: 'rgba(255, 255, 255, 0.8)', 
          fontSize: '1.1rem', 
          margin: '2rem 0',
          padding: '2rem',
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '12px',
          border: '1px solid rgba(255, 255, 255, 0.1)'
        }}>
          🤖 Aucune IA trouvée
        </div>
      ) : (
        <AIGrid>
          {availableAIs.map((ai, index) => {
          const selectedCount = selectedAIs.filter(selected => selected.id === ai.id).length;
          
          return (
            <AICard
              key={ai.id}
              selected={selectedCount > 0}
              variants={cardVariants}
              initial="hidden"
              animate="visible"
              transition={{ duration: 0.6, delay: index * 0.1 }}
              onClick={() => toggleAISelection(ai)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <AIHeader>
                <div style={{ width: '100%' }}>
                  <AIName>{ai.icon} {ai.name}</AIName>
                  <AILevel winRate={ai.winRate}>
                    {ai.winRate >= 40 ? 'Avancé' : ai.winRate >= 20 ? 'Intermédiaire' : 'Débutant'}
                  </AILevel>
                </div>
              </AIHeader>

              <AIStats>
                <StatItem>
                  <StatValue>{ai.avgScore}</StatValue>
                  <StatLabel>Score Moyen</StatLabel>
                </StatItem>
                <StatItem>
                  <WinRateValue rate={ai.winRate}>{ai.winRate}%</WinRateValue>
                  <StatLabel>Taux Victoire</StatLabel>
                </StatItem>
              </AIStats>

              <AnimatePresence>
                {selectedCount > 0 && (
                  <SelectionContainer>
                    <SelectionIndicator
                      initial={{ scale: 0 }}
                      animate={{ scale: 1 }}
                      exit={{ scale: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      {selectedCount > 1 ? selectedCount : '✓'}
                    </SelectionIndicator>
                    <RemoveButton
                      onClick={(e) => {
                        e.stopPropagation();
                        removeAISelection(ai);
                      }}
                      title="Retirer une instance"
                    >
                      −
                    </RemoveButton>
                  </SelectionContainer>
                )}
              </AnimatePresence>
            </AICard>
          );
          })}
        </AIGrid>
      )}

      <ControlsContainer>
        <ControlButton
          onClick={onBack}
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          ← Retour
        </ControlButton>
        
        <ControlButton
          primary
          disabled={selectedAIs.length < 2}
          onClick={handleStartGame}
          whileHover={{ scale: selectedAIs.length >= 2 ? 1.05 : 1 }}
          whileTap={{ scale: selectedAIs.length >= 2 ? 0.95 : 1 }}
        >
          Commencer la Partie →
        </ControlButton>
      </ControlsContainer>
    </SelectorContainer>
  );
};

export default AISelector; 