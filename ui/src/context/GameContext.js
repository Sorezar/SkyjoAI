import React, { createContext, useContext, useReducer, useEffect } from 'react';

const GameContext = createContext();

// Actions pour le reducer
const GAME_ACTIONS = {
  SET_GAME_STATE: 'SET_GAME_STATE',
  UPDATE_PLAYER: 'UPDATE_PLAYER',
  UPDATE_CURRENT_PLAYER: 'UPDATE_CURRENT_PLAYER',
  REVEAL_CARD: 'REVEAL_CARD',
  PLACE_CARD: 'PLACE_CARD',
  END_ROUND: 'END_ROUND',
  END_GAME: 'END_GAME',
  RESET_GAME: 'RESET_GAME',
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR'
};

// État initial du jeu
const initialState = {
  gameId: null,
  round: 1,
  turn: 0,
  currentPlayerIndex: 0,
  phase: 'waiting', // waiting, playing, finished
  players: [],
  deck: [],
  discardPile: [],
  lastCard: null,
  lastAction: null,
  scores: {},
  winner: null,
  loading: false,
  error: null,
  animations: {
    cardFlip: false,
    cardMove: false,
    playerTurn: false
  }
};

// Reducer pour gérer les actions
function gameReducer(state, action) {
  switch (action.type) {
    case GAME_ACTIONS.SET_GAME_STATE:
      return {
        ...state,
        ...action.payload
      };

    case GAME_ACTIONS.UPDATE_PLAYER:
      return {
        ...state,
        players: state.players.map(player =>
          player.id === action.payload.id
            ? { ...player, ...action.payload.updates }
            : player
        )
      };

    case GAME_ACTIONS.UPDATE_CURRENT_PLAYER:
      return {
        ...state,
        currentPlayerIndex: action.payload,
      };

    case GAME_ACTIONS.REVEAL_CARD:
      const { playerId, cardPosition } = action.payload;
      return {
        ...state,
        players: state.players.map(player =>
          player.id === playerId
            ? {
                ...player,
                grid: player.grid.map((row, rowIndex) =>
                  row.map((card, colIndex) =>
                    rowIndex === cardPosition.row && colIndex === cardPosition.col
                      ? { ...card, revealed: true }
                      : card
                  )
                )
              }
            : player
        ),
        animations: { ...state.animations, cardFlip: true }
      };

    case GAME_ACTIONS.PLACE_CARD:
      const { card, position } = action.payload;
      return {
        ...state,
        players: state.players.map(player =>
          player.id === state.currentPlayerIndex
            ? {
                ...player,
                grid: player.grid.map((row, rowIndex) =>
                  row.map((cell, colIndex) =>
                    rowIndex === position.row && colIndex === position.col
                      ? card
                      : cell
                  )
                )
              }
            : player
        ),
        lastCard: card,
        lastAction: 'place_card',
        animations: { ...state.animations, cardMove: true }
      };

    case GAME_ACTIONS.END_ROUND:
      return {
        ...state,
        round: state.round + 1,
        turn: 0,
        currentPlayerIndex: 0,
        scores: {
          ...state.scores,
          [`round_${state.round}`]: action.payload.scores
        }
      };

    case GAME_ACTIONS.END_GAME:
      return {
        ...state,
        phase: 'finished',
        winner: action.payload.winner,
        scores: {
          ...state.scores,
          final: action.payload.finalScores
        }
      };

    case GAME_ACTIONS.RESET_GAME:
      return initialState;

    case GAME_ACTIONS.SET_LOADING:
      return {
        ...state,
        loading: action.payload
      };

    case GAME_ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false
      };

    default:
      return state;
  }
}

// Hook pour utiliser le contexte
export const useGame = () => {
  const context = useContext(GameContext);
  if (!context) {
    throw new Error('useGame must be used within a GameProvider');
  }
  return context;
};

// Provider du contexte
export const GameProvider = ({ children }) => {
  const [state, dispatch] = useReducer(gameReducer, initialState);

  // Actions du jeu
  const actions = {
    setGameState: (gameState) => {
      dispatch({ type: GAME_ACTIONS.SET_GAME_STATE, payload: gameState });
    },

    updatePlayer: (playerId, updates) => {
      dispatch({
        type: GAME_ACTIONS.UPDATE_PLAYER,
        payload: { id: playerId, updates }
      });
    },

    updateCurrentPlayer: (playerIndex) => {
      dispatch({
        type: GAME_ACTIONS.UPDATE_CURRENT_PLAYER,
        payload: playerIndex
      });
    },

    revealCard: (playerId, cardPosition) => {
      dispatch({
        type: GAME_ACTIONS.REVEAL_CARD,
        payload: { playerId, cardPosition }
      });
    },

    placeCard: (card, position) => {
      dispatch({
        type: GAME_ACTIONS.PLACE_CARD,
        payload: { card, position }
      });
    },

    endRound: (scores) => {
      dispatch({
        type: GAME_ACTIONS.END_ROUND,
        payload: { scores }
      });
    },

    endGame: (winner, finalScores) => {
      dispatch({
        type: GAME_ACTIONS.END_GAME,
        payload: { winner, finalScores }
      });
    },

    resetGame: () => {
      dispatch({ type: GAME_ACTIONS.RESET_GAME });
    },

    setLoading: (loading) => {
      dispatch({ type: GAME_ACTIONS.SET_LOADING, payload: loading });
    },

    setError: (error) => {
      dispatch({ type: GAME_ACTIONS.SET_ERROR, payload: error });
    }
  };

  // Connecter avec l'API Python (future implémentation)
  const connectToGameAPI = async () => {
    try {
      // Ici on pourra faire des appels à l'API Python
      const response = await fetch('/api/game/init', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ players: state.players })
      });
      
      if (response.ok) {
        const gameData = await response.json();
        actions.setGameState(gameData);
      }
    } catch (error) {
      actions.setError('Erreur de connexion à l\'API');
    }
  };

  const value = {
    state,
    actions,
    connectToGameAPI
  };

  return (
    <GameContext.Provider value={value}>
      {children}
    </GameContext.Provider>
  );
}; 