import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

const StatsContainer = styled.div`
  height: 100%;
  padding: 2rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  overflow-y: auto;
`;

const StatsTitle = styled(motion.h2)`
  color: white;
  font-size: 2.2rem;
  font-weight: 700;
  margin-bottom: 1.5rem;
  text-align: center;
`;

const ActionButton = styled(motion.button)`
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
  
  &:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
  }
`;

const MainContentContainer = styled.div`
  display: flex;
  align-items: flex-start;
  justify-content: center;
  gap: 2rem;
  width: 100%;
  max-width: 1200px;
  margin: 1rem auto 0;
`;

const CenterContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  flex: 1;
  max-width: 800px;
  width: 100%;
`;

// Styles pour le podium
const PodiumContainer = styled(motion.div)`
  display: flex;
  justify-content: center;
  align-items: end;
  gap: 1.5rem;
  margin-bottom: 2.5rem;
  padding: 1.5rem;
`;

const PodiumPlace = styled(motion.div)`
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  order: ${props => props.position === 1 ? 2 : props.position === 2 ? 1 : 3};
`;

const PodiumStep = styled.div`
  background: ${props => {
    if (props.position === 1) return 'linear-gradient(135deg, #ffd700, #ffed4e)';
    if (props.position === 2) return 'linear-gradient(135deg, #c0c0c0, #e8e8e8)';
    return 'linear-gradient(135deg, #cd7f32, #daa560)';
  }};
  width: 100px;
  height: ${props => props.position === 1 ? '120px' : props.position === 2 ? '100px' : '80px'};
  border-radius: 10px 10px 0 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.8rem;
  font-weight: 700;
  color: ${props => props.position === 1 ? '#b8860b' : props.position === 2 ? '#666' : '#8b4513'};
  margin-bottom: 0.75rem;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
`;

const PodiumPlayer = styled.div`
  color: white;
  font-size: 1.1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  max-width: 120px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const PodiumScore = styled.div`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1rem;
  font-weight: 500;
`;

// Styles pour le tableau des scores par manche
const ScoreTable = styled(motion.div)`
  background: rgba(0, 0, 0, 0.2);
  border-radius: 12px;
  padding: 1.2rem;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  color: white;
  overflow-x: auto;
  max-height: 50vh;
  overflow-y: auto;
`;

const ScoreTableTitle = styled.h4`
  font-size: 1.1rem;
  margin-bottom: 0.8rem;
  color: white;
  text-align: center;
`;

const Table = styled.table`
  width: 100%;
  border-collapse: collapse;
  margin-bottom: 1rem;
`;

const TableHeader = styled.th`
  background: rgba(255, 255, 255, 0.1);
  padding: 0.6rem;
  text-align: center;
  font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.2);
  font-size: 0.85rem;
`;

const TableCell = styled.td`
  padding: 0.6rem;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.1);
  font-size: 0.85rem;
  background: ${props => {
    if (props.isTotal) return 'rgba(255, 255, 255, 0.1)';
    if (props.isAverage) return 'rgba(255, 255, 255, 0.05)';
    return 'transparent';
  }};
  font-weight: ${props => props.isTotal || props.isAverage ? '600' : '400'};
  color: ${props => {
    if (props.isTotal || props.isAverage) return 'white';
    if (props.value < 0) return '#74b9ff';
    if (props.value < 10) return '#55efc4';
    if (props.value < 20) return '#fdcb6e';
    return '#ff7675';
  }};
`;

const TableRow = styled.tr`
  &:nth-child(even) {
    background: rgba(255, 255, 255, 0.02);
  }
  &:hover {
    background: rgba(255, 255, 255, 0.05);
  }
`;

const GameStats = ({ gameData, onBackToMenu, onNewGame }) => {
  if (!gameData) {
    return (
      <StatsContainer>
        <StatsTitle>Aucune donnée de jeu disponible</StatsTitle>
        <ActionButton onClick={onBackToMenu}>
          Retour au Menu
        </ActionButton>
      </StatsContainer>
    );
  }

  const { players, scoreboard } = gameData;

  // Créer le podium
  const createPodium = () => {
    const sortedPlayers = [...players].sort((a, b) => a.score - b.score);
    const topThree = sortedPlayers.slice(0, 3);

    return (
      <PodiumContainer variants={itemVariants}>
        {topThree.map((player, index) => {
          const position = index + 1;
          const medal = position === 1 ? '🥇' : position === 2 ? '🥈' : '🥉';
          
          return (
            <PodiumPlace 
              key={player.id}
              position={position}
              variants={itemVariants}
              whileHover={{ scale: 1.05 }}
            >
              <PodiumPlayer>{player.name}</PodiumPlayer>
              <PodiumStep position={position}>
                {medal}
              </PodiumStep>
              <PodiumScore>{player.score} pts</PodiumScore>
            </PodiumPlace>
          );
        })}
      </PodiumContainer>
    );
  };

  // Créer le tableau des scores par manche
  const createScoreTable = () => {
    if (!scoreboard || !scoreboard.round_scores) return null;

    const roundScores = scoreboard.round_scores;
    const totalScores = scoreboard.total_scores || players.map(p => p.score);
    
    // Calculer le nombre de manches jouées
    const maxRounds = Math.max(...roundScores.map(playerRounds => playerRounds.length));
    
    if (maxRounds === 0) return null;
    const orderedPlayers = players.map((player, originalIndex) => {
      let scoreboardIndex = originalIndex;
      
      if (totalScores[originalIndex] !== player.score) {
        scoreboardIndex = totalScores.findIndex(score => score === player.score);
        if (scoreboardIndex === -1) scoreboardIndex = originalIndex;
      }
      
      return {
        ...player,
        scoreboardIndex: scoreboardIndex
      };
    });

    return (
      <ScoreTable variants={itemVariants}>
        <ScoreTableTitle>📈 Détail des scores par manche</ScoreTableTitle>
        <Table>
          <thead>
            <TableRow>
              <TableHeader>Manche</TableHeader>
              {orderedPlayers.map((player, index) => (
                <TableHeader key={index}>{player.name}</TableHeader>
              ))}
            </TableRow>
          </thead>
          <tbody>
            {Array.from({ length: maxRounds }, (_, roundIndex) => (
              <TableRow key={roundIndex}>
                <TableCell isTotal>{roundIndex + 1}</TableCell>
                {orderedPlayers.map((player, playerIndex) => {
                  const score = roundScores[player.scoreboardIndex]?.[roundIndex];
                  return (
                    <TableCell 
                      key={playerIndex} 
                      value={score}
                    >
                      {score !== undefined ? score : '-'}
                    </TableCell>
                  );
                })}
              </TableRow>
            ))}
            
            {/* Ligne Total */}
            <TableRow>
              <TableCell isTotal>Total</TableCell>
              {orderedPlayers.map((player, playerIndex) => (
                <TableCell 
                  key={playerIndex} 
                  isTotal
                  value={totalScores[player.scoreboardIndex]}
                >
                  {totalScores[player.scoreboardIndex]}
                </TableCell>
              ))}
            </TableRow>
            
            {/* Ligne Moyenne */}
            <TableRow>
              <TableCell isAverage>Moyenne</TableCell>
              {orderedPlayers.map((player, playerIndex) => {
                const playerRounds = roundScores[player.scoreboardIndex] || [];
                const average = playerRounds.length > 0 
                  ? playerRounds.reduce((sum, score) => sum + score, 0) / playerRounds.length
                  : 0;
                return (
                  <TableCell 
                    key={playerIndex} 
                    isAverage
                    value={average}
                  >
                    {average.toFixed(1)}
                  </TableCell>
                );
              })}
            </TableRow>
          </tbody>
        </Table>
      </ScoreTable>
    );
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: { duration: 0.6 }
    }
  };

  return (
    <StatsContainer>
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        style={{ width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center' }}
      >
        <StatsTitle variants={itemVariants}>
          🎉 Partie Terminée !
        </StatsTitle>

        {createPodium()}

        <MainContentContainer>
          <CenterContent>
            {createScoreTable()}
          </CenterContent>
        </MainContentContainer>
      </motion.div>
    </StatsContainer>
  );
};

export default GameStats; 