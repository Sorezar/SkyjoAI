import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

const ControlsContainer = styled.div`
  display: flex;
  gap: 0.5rem;
  align-items: center;
`;

const ControlButton = styled(motion.button)`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0px);
  }
`;

const GameControls = ({ onPause, onSettings, isPaused }) => {
  return (
    <ControlsContainer>
      
      <ControlButton
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
        onClick={onPause}
      >
        {isPaused ? '▶️' : '⏸️'}
      </ControlButton>
    </ControlsContainer>
  );
};

export default GameControls; 