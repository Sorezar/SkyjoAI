import React from 'react';
import { motion } from 'framer-motion';
import styled from 'styled-components';

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  text-align: center;
  padding: 2rem;
`;

const LoadingTitle = styled(motion.h2)`
  color: white;
  font-size: 2.5rem;
  font-weight: 700;
  margin-bottom: 2rem;
`;

const LoadingSpinner = styled(motion.div)`
  width: 120px;
  height: 120px;
  border: 4px solid rgba(255, 255, 255, 0.1);
  border-top: 4px solid #667eea;
  border-radius: 50%;
  margin: 2rem auto;
`;

const LoadingMessage = styled(motion.p)`
  color: rgba(255, 255, 255, 0.8);
  font-size: 1.2rem;
  margin: 1rem 0;
  max-width: 500px;
`;

const ProgressBar = styled.div`
  width: 300px;
  height: 8px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
  margin: 2rem auto;
  overflow: hidden;
`;

const ProgressFill = styled(motion.div)`
  height: 100%;
  background: linear-gradient(90deg, #667eea, #764ba2);
  border-radius: 4px;
`;

const LoadingSteps = styled.div`
  margin-top: 2rem;
  max-width: 400px;
`;

const Step = styled(motion.div)`
  display: flex;
  align-items: center;
  margin: 0.5rem 0;
  color: rgba(255, 255, 255, 0.7);
  font-size: 0.95rem;
`;

const StepIcon = styled.div`
  width: 20px;
  height: 20px;
  border-radius: 50%;
  margin-right: 1rem;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.8rem;
  background: ${props => props.completed ? 
    'linear-gradient(45deg, #00d2ff, #3a7bd5)' : 
    'rgba(255, 255, 255, 0.2)'};
  color: white;
`;

const CardAnimation = styled(motion.div)`
  display: flex;
  gap: 1rem;
  margin: 2rem 0;
`;

const AnimatedCard = styled(motion.div)`
  width: 60px;
  height: 80px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: bold;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
`;

const loadingSteps = [
  'Initialisation des IA...',
  'Mélange des cartes...',
  'Configuration de la partie...',
  'Démarrage du jeu...'
];

const LoadingScreen = () => {
  const [currentStep, setCurrentStep] = React.useState(0);
  const [progress, setProgress] = React.useState(0);

  React.useEffect(() => {
    const stepDuration = 500; // 500ms par étape
    const progressIncrement = 100 / loadingSteps.length;

    const stepInterval = setInterval(() => {
      setCurrentStep(prev => {
        if (prev < loadingSteps.length - 1) {
          setProgress((prev + 1) * progressIncrement);
          return prev + 1;
        }
        clearInterval(stepInterval);
        return prev;
      });
    }, stepDuration);

    return () => clearInterval(stepInterval);
  }, []);

  const cardVariants = {
    initial: { rotateY: 0, scale: 1 },
    animate: { 
      rotateY: 360,
      scale: [1, 1.1, 1],
      transition: {
        rotateY: { duration: 2, repeat: Infinity, ease: "linear" },
        scale: { duration: 1, repeat: Infinity, repeatType: "reverse" }
      }
    }
  };

  const spinnerVariants = {
    animate: {
      rotate: 360,
      transition: {
        duration: 1,
        repeat: Infinity,
        ease: "linear"
      }
    }
  };

  return (
    <LoadingContainer>
      <LoadingTitle
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.6 }}
      >
        🎮 Préparation de la partie...
      </LoadingTitle>

      <CardAnimation>
        {[1, 2, 3, 4, 5].map((card, index) => (
          <AnimatedCard
            key={card}
            variants={cardVariants}
            initial="initial"
            animate="animate"
            transition={{ delay: index * 0.2 }}
          >
            {card}
          </AnimatedCard>
        ))}
      </CardAnimation>

      <LoadingSpinner
        variants={spinnerVariants}
        animate="animate"
      />

      <ProgressBar>
        <ProgressFill
          initial={{ width: 0 }}
          animate={{ width: `${progress}%` }}
          transition={{ duration: 0.5 }}
        />
      </ProgressBar>

      <LoadingSteps>
        {loadingSteps.map((step, index) => (
          <Step
            key={index}
            initial={{ x: -20, opacity: 0 }}
            animate={{ 
              x: 0, 
              opacity: index <= currentStep ? 1 : 0.5 
            }}
            transition={{ 
              duration: 0.3,
              delay: index * 0.1 
            }}
          >
            <StepIcon completed={index <= currentStep}>
              {index <= currentStep ? '✓' : index + 1}
            </StepIcon>
            {step}
          </Step>
        ))}
      </LoadingSteps>

      <LoadingMessage
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.6 }}
      >
        Nos IA analysent leurs stratégies et se préparent à vous défier ! 🤖
      </LoadingMessage>
    </LoadingContainer>
  );
};

export default LoadingScreen; 