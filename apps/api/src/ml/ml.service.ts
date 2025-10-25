/**
 * ML Service Integration
 *
 * Connects NestJS API to FastAPI ML service for predictions
 */

import { Injectable, Logger, HttpException, HttpStatus } from '@nestjs/common';
import axios, { AxiosInstance } from 'axios';

export interface PetFeatures {
  breed: string;
  size: string;
  energy: string;
  temperament: string;
  age: number;
  social: number;
  weight: number;
}

export interface CompatibilityPrediction {
  compatibility_score: number;
  confidence: number;
  factors: {
    energy_match: number;
    size_compatibility: number;
    age_proximity: number;
    social_affinity: number;
  };
  cached: boolean;
}

export interface EnergyPrediction {
  energy_state: 'low' | 'medium' | 'high';
  probabilities: {
    low: number;
    medium: number;
    high: number;
  };
  confidence: number;
  recommendation: string;
  cached: boolean;
}

export interface ActivityRecommendation {
  activity_type: string;
  probability: number;
  optimal_time: number;
  expected_duration: number;
  energy_requirement: string;
}

@Injectable()
export class MLService {
  private readonly logger = new Logger(MLService.name);
  private readonly client: AxiosInstance;
  private readonly ML_SERVICE_URL: string;

  constructor() {
    this.ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://localhost:8001';

    this.client = axios.create({
      baseURL: this.ML_SERVICE_URL,
      timeout: 5000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Health check on startup
    this.checkHealth();
  }

  private async checkHealth(): Promise<void> {
    try {
      const response = await this.client.get('/health');
      this.logger.log(`ML Service connected: ${JSON.stringify(response.data)}`);
    } catch (error) {
      this.logger.warn(`ML Service not available: ${error.message}`);
    }
  }

  /**
   * Predict compatibility between two pets
   */
  async predictCompatibility(
    pet1: PetFeatures,
    pet2: PetFeatures,
  ): Promise<CompatibilityPrediction> {
    try {
      const response = await this.client.post<CompatibilityPrediction>(
        '/predict/compatibility',
        { pet1, pet2 },
      );

      this.logger.debug(
        `Compatibility prediction: ${response.data.compatibility_score} (cached: ${response.data.cached})`,
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Compatibility prediction failed: ${error.message}`);
      throw new HttpException(
        'ML prediction service unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  /**
   * Predict current energy state of a pet
   */
  async predictEnergy(params: {
    age: number;
    breed: string;
    base_energy_level: string;
    hours_since_last_activity: number;
    total_distance_24h: number;
    total_duration_24h: number;
    num_activities_24h: number;
    hour_of_day: number;
    day_of_week: number;
  }): Promise<EnergyPrediction> {
    try {
      const response = await this.client.post<EnergyPrediction>(
        '/predict/energy',
        params,
      );

      this.logger.debug(
        `Energy prediction: ${response.data.energy_state} (confidence: ${response.data.confidence})`,
      );

      return response.data;
    } catch (error) {
      this.logger.error(`Energy prediction failed: ${error.message}`);
      throw new HttpException(
        'ML prediction service unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  /**
   * Batch compatibility predictions for multiple pet pairs
   */
  async batchCompatibility(
    pairs: Array<{ pet1: PetFeatures; pet2: PetFeatures }>,
  ): Promise<CompatibilityPrediction[]> {
    try {
      const response = await this.client.post<{ results: CompatibilityPrediction[] }>(
        '/predict/compatibility/batch',
        { pairs },
      );

      return response.data.results;
    } catch (error) {
      this.logger.error(`Batch compatibility prediction failed: ${error.message}`);
      throw new HttpException(
        'ML prediction service unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  /**
   * Get activity recommendations
   */
  async recommendActivities(params: {
    pet_id: string;
    recent_activities: any;
    current_energy: string;
    preferences?: any;
  }): Promise<{
    recommendations: ActivityRecommendation[];
    predicted_energy: string;
    confidence: number;
  }> {
    try {
      const response = await this.client.post('/recommend/activities', params);
      return response.data;
    } catch (error) {
      this.logger.error(`Activity recommendation failed: ${error.message}`);
      throw new HttpException(
        'ML prediction service unavailable',
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  /**
   * Clear ML service cache
   */
  async clearCache(): Promise<void> {
    try {
      await this.client.delete('/cache/clear');
      this.logger.log('ML service cache cleared');
    } catch (error) {
      this.logger.error(`Cache clear failed: ${error.message}`);
    }
  }

  /**
   * Trigger model reload
   */
  async reloadModels(): Promise<void> {
    try {
      await this.client.post('/models/reload');
      this.logger.log('ML models reload initiated');
    } catch (error) {
      this.logger.error(`Model reload failed: ${error.message}`);
    }
  }
}
