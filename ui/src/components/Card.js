import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

const CardContainer = styled(motion.div)`
  width: ${props => props.size === 'large' ? '70px' : props.size === 'small' ? '45px' : '55px'};
  height: ${props => props.size === 'large' ? '95px' : props.size === 'small' ? '60px' : '75px'};
  perspective: 1000px;
  cursor: ${props => props.clickable ? 'pointer' : 'default'};
`;

const CardInner = styled(motion.div)`
  width: 100%;
  height: 100%;
  position: relative;
  transform-style: preserve-3d;
  transition: transform 0.6s;
`;

const CardFace = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 8px;
  backface-visibility: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: ${props => props.size === 'large' ? '1.5rem' : '1.2rem'};
  border: 2px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
`;

const CardFront = styled(CardFace)`
  background: ${props => props.revealed ? getCardColor(props.value) : 'linear-gradient(135deg, #667eea, #764ba2)'};
  color: ${props => props.revealed ? getTextColor(props.value) : 'white'};
  ${props => props.revealed ? '' : 'transform: rotateY(180deg);'}
`;

const CardBack = styled(CardFace)`
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  ${props => props.revealed ? 'transform: rotateY(180deg);' : ''}
  
  &::before {
    content: '🃏';
    font-size: ${props => props.size === 'large' ? '2rem' : '1.5rem'};
  }
`;

// Fonction pour obtenir la couleur de la carte selon sa valeur
const getCardColor = (value) => {
  if (value === null || value === undefined) return 'linear-gradient(135deg, #667eea, #764ba2)';
  if (value === -2 || value === -1) return 'linear-gradient(135deg, #1e3c72, #2a5298)';
  if (value === 0)                  return 'linear-gradient(135deg, #74b9ff, #0984e3)';
  if (value >= 1 && value <= 4)     return 'linear-gradient(135deg, #55efc4, #00b894)';
  if (value >= 5 && value <= 8)     return 'linear-gradient(135deg, #fdcb6e, #f39c12)';
  if (value >= 9 && value <= 12)    return 'linear-gradient(135deg, #ff7675, #e17055)';
  
  return 'linear-gradient(135deg, #667eea, #764ba2)';
};

// Fonction pour obtenir la couleur du texte selon la valeur
const getTextColor = (value) => {
  if (value === null || value === undefined) return 'white';
  return 'white';
};

const Card = ({ 
  value, 
  revealed = false, 
  size = 'normal', 
  onClick, 
  isSelected = false,
  isHighlighted = false,
  className,
  noAnimation = false
}) => {
  const cardVariants = {
    hidden: { scale: 0, rotateY: 180 },
    visible: { 
      scale: 1, 
      rotateY: revealed ? 0 : 180,
      transition: { duration: 0.6, ease: "easeOut" }
    },
    hover: { 
      scale: 1.05,
      y: -5,
      transition: { duration: 0.2 }
    },
    tap: { 
      scale: 0.95,
      transition: { duration: 0.1 }
    },
    selected: {
      scale: 1.1,
      boxShadow: "0 0 20px rgba(255, 107, 107, 0.6)",
      transition: { duration: 0.3 }
    },
    highlighted: {
      boxShadow: "0 0 15px rgba(254, 202, 87, 0.8)",
      transition: { duration: 0.3 }
    }
  };

  const flipVariants = {
    initial: { rotateY: noAnimation ? 0 : 180 },
    flipped: { 
      rotateY: 0,
      transition: { duration: noAnimation ? 0 : 0.6, ease: "easeInOut" }
    }
  };

  return (
    <CardContainer
      size={size}
      clickable={!!onClick}
      className={className}
      variants={cardVariants}
      initial="hidden"
      animate={[
        "visible",
        isSelected && "selected",
        isHighlighted && "highlighted"
      ].filter(Boolean)}
      whileHover={onClick ? "hover" : {}}
      whileTap={onClick ? "tap" : {}}
      onClick={onClick}
    >
      <CardInner
        variants={flipVariants}
        initial="initial"
        animate={revealed ? "flipped" : "initial"}
      >
        <CardBack 
          size={size}
          revealed={revealed}
        />
        <CardFront 
          size={size}
          value={value}
          revealed={revealed}
        >
          {revealed && value !== null && value !== undefined ? value : ''}
        </CardFront>
      </CardInner>
    </CardContainer>
  );
};

// Composant pour une pile de cartes
export const CardStack = ({ count, size = 'normal', offset = 2 }) => {
  const cards = Array.from({ length: Math.min(count, 5) }, (_, i) => i);
  
  return (
    <div style={{ position: 'relative' }}>
      {cards.map((_, index) => (
        <Card
          key={index}
          value={null}
          revealed={false}
          size={size}
          style={{
            position: index > 0 ? 'absolute' : 'relative',
            top: index > 0 ? -index * offset : 0,
            left: index > 0 ? -index * offset : 0,
            zIndex: cards.length - index
          }}
        />
      ))}
    </div>
  );
};

// Composant pour animer la distribution des cartes
export const AnimatedCardDeal = ({ onComplete, delay = 0 }) => {
  React.useEffect(() => {
    const timer = setTimeout(() => {
      onComplete?.();
    }, 2000 + delay);
    
    return () => clearTimeout(timer);
  }, [onComplete, delay]);

  return (
    <motion.div
      initial={{ x: 0, y: 0, scale: 1 }}
      animate={{ 
        x: [0, 100, 200, 300],
        y: [0, -50, 0, 50],
        scale: [1, 0.8, 0.6, 0.4],
        opacity: [1, 1, 1, 0]
      }}
      transition={{ 
        duration: 2,
        delay,
        ease: "easeInOut"
      }}
    >
      <Card value={null} revealed={false} size="normal" />
    </motion.div>
  );
};

export default Card; 