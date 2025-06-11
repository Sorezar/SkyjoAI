/**
 * Service pour communiquer avec l'API Skyjo Python
 */

const API_BASE_URL = 'http://localhost:8000';

class GameService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  /**
   * Récupère la liste des IA disponibles
   */
  async getAvailableAIs() {
    try {
      const response = await fetch(`${this.baseURL}/ai/available`);
      if (!response.ok) {
        throw new Error(`Erreur API: ${response.status}`);
      }
      const data = await response.json();
      return data.available_ais;
    } catch (error) {
      console.error('Erreur lors de la récupération des IA:', error);
      throw error;
    }
  }

  /**
   * Crée une nouvelle partie
   */
  async createGame(aiConfigs) {
    try {
      const gameConfig = {
        players: aiConfigs.map(ai => ({
          id: ai.uniqueId || ai.id,
          name: ai.name || ai.id,
          type: ai.id
        }))
      };

      const response = await fetch(`${this.baseURL}/game/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(gameConfig)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Erreur API: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Erreur lors de la création de la partie:', error);
      throw error;
    }
  }

  /**
   * Récupère l'état actuel d'une partie
   */
  async getGameState(gameId) {
    try {
      const response = await fetch(`${this.baseURL}/game/${gameId}/state`);
      if (!response.ok) {
        throw new Error(`Erreur API: ${response.status}`);
      }
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Erreur lors de la récupération de l\'état:', error);
      throw error;
    }
  }

  /**
   * Fait avancer la partie d'un tour
   */
  async stepGame(gameId) {
    try {
      const response = await fetch(`${this.baseURL}/game/${gameId}/step`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Erreur API: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Erreur lors de l\'exécution du tour:', error);
      throw error;
    }
  }

  /**
   * Lance plusieurs tours automatiquement
   */
  async autoPlay(gameId, steps = 10) {
    try {
      const response = await fetch(`${this.baseURL}/game/${gameId}/auto-play?steps=${steps}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Erreur API: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Erreur lors de l\'auto-play:', error);
      throw error;
    }
  }

  /**
   * Supprime une partie
   */
  async deleteGame(gameId) {
    try {
      const response = await fetch(`${this.baseURL}/game/${gameId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error(`Erreur API: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Erreur lors de la suppression:', error);
      throw error;
    }
  }

  /**
   * Liste toutes les parties actives
   */
  async listGames() {
    try {
      const response = await fetch(`${this.baseURL}/games`);
      if (!response.ok) {
        throw new Error(`Erreur API: ${response.status}`);
      }
      const data = await response.json();
      return data.active_games;
    } catch (error) {
      console.error('Erreur lors de la liste des parties:', error);
      throw error;
    }
  }
}

const gameService = new GameService();
export default gameService; 