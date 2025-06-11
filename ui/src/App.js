import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import styled from 'styled-components';
import GameBoardAPI from './components/GameBoardAPI';
import MainMenu from './components/MainMenu';
import AISelector from './components/AISelector';
import GameStats from './components/GameStats';
import LoadingScreen from './components/LoadingScreen';
import { GameProvider } from './context/GameContext';

const AppContainer = styled.div`
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  flex-direction: column;
  position: relative;
  overflow-x: hidden;
`;

const Header = styled(motion.header)`
  background: rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
  padding: 0.6rem 1.5rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  min-height: 60px;
`;

const Logo = styled(motion.h1)`
  color: white;
  font-size: 1.5rem;
  font-weight: 700;
  background: linear-gradient(45deg, #ff6b6b, #feca57);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  cursor: pointer;
  
  &:hover {
    transform: scale(1.05);
  }
`;

const MainContent = styled(motion.main)`
  flex: 1;
  display: flex;
  flex-direction: column;
  position: relative;
`;

// États de l'application
const SCREENS = {
  MENU: 'menu',
  AI_SELECTOR: 'ai_selector',
  GAME: 'game',
  STATS: 'stats',
  LOADING: 'loading'
};

function App() {
  const [currentScreen, setCurrentScreen] = useState(SCREENS.MENU);
  const [gameConfig, setGameConfig] = useState({
    players: [],
    gameId: null
  });
  const [gameData, setGameData] = useState(null);

  // Animation variants pour les transitions entre écrans
  const screenVariants = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 }
  };

  const handleStartGame = () => {
    setCurrentScreen(SCREENS.AI_SELECTOR);
  };

  const handleAISelection = async (selectedAIs, gameId) => {
    setCurrentScreen(SCREENS.LOADING);
    
    setTimeout(() => {
      setGameConfig(prev => ({
        ...prev,
        players: selectedAIs,
        gameId: gameId
      }));
      setCurrentScreen(SCREENS.GAME);
    }, 1000);
  };

  const handleGameEnd = (results) => {
    setGameData(results);
    setCurrentScreen(SCREENS.STATS);
  };

  const handleBackToMenu = () => {
    setCurrentScreen(SCREENS.MENU);
    setGameData(null);
    setGameConfig({
      players: [],
      gameId: null
    });
  };

  const renderCurrentScreen = () => {
    switch (currentScreen) {
      case SCREENS.MENU:
        return (
          <MainMenu 
            onStartGame={handleStartGame}
            onViewStats={() => setCurrentScreen(SCREENS.STATS)}
          />
        );
      
      case SCREENS.AI_SELECTOR:
        return (
          <AISelector 
            gameConfig={gameConfig}
            onSelectAIs={handleAISelection}
            onBack={() => setCurrentScreen(SCREENS.MENU)}
          />
        );
      
      case SCREENS.LOADING:
        return <LoadingScreen />;
      
      case SCREENS.GAME:
        return (
          <GameBoardAPI 
            gameId={gameConfig.gameId}
            onGameEnd={handleGameEnd}
            onBackToMenu={handleBackToMenu}
          />
        );
      
      case SCREENS.STATS:
        return (
          <GameStats 
            gameData={gameData}
            onBackToMenu={handleBackToMenu}
            onNewGame={() => setCurrentScreen(SCREENS.MENU)}
          />
        );
      
      default:
        return <MainMenu onStartGame={handleStartGame} />;
    }
  };

  return (
    <GameProvider>
      <AppContainer>
        <Header
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        >
          <Logo
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            onClick={handleBackToMenu}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            🃏 Skyjo AI
          </Logo>
        </Header>

        <MainContent>
          <AnimatePresence mode="wait">
            <motion.div
              key={currentScreen}
              variants={screenVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              transition={{ duration: 0.4 }}
              style={{ height: '100%' }}
            >
              {renderCurrentScreen()}
            </motion.div>
          </AnimatePresence>
        </MainContent>
      </AppContainer>
    </GameProvider>
  );
}

export default App; 