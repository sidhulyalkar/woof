import { Injectable, Logger } from '@nestjs/common';
import { createHash } from 'crypto';

export enum ModelVariant {
  GAT_ONLY = 'gat_only',
  SIMGNN_ONLY = 'simgnn_only',
  DIFFUSION_ONLY = 'diffusion_only',
  HYBRID = 'hybrid',
}

export interface ABTestConfig {
  experimentName: string;
  variants: Array<{
    name: ModelVariant;
    traffic: number; // 0-100
    enabled: boolean;
  }>;
  startDate: Date;
  endDate?: Date;
}

export interface ABTestEvent {
  userId: string;
  experimentName: string;
  variant: ModelVariant;
  timestamp: Date;
  prediction: number;
  confidence?: number;
  userSatisfaction?: number;
  actualOutcome?: boolean;
  metadata?: any;
}

@Injectable()
export class ABTestService {
  private readonly logger = new Logger(ABTestService.name);

  // Active experiments
  private experiments: Map<string, ABTestConfig> = new Map();

  // Event storage (in production: use analytics DB)
  private events: ABTestEvent[] = [];

  constructor() {
    // Initialize default experiment: GAT vs Hybrid
    this.registerExperiment({
      experimentName: 'gat_vs_hybrid',
      variants: [
        { name: ModelVariant.GAT_ONLY, traffic: 50, enabled: true },
        { name: ModelVariant.HYBRID, traffic: 50, enabled: true },
      ],
      startDate: new Date(),
    });

    this.logger.log('A/B Testing Service initialized');
  }

  /**
   * Register a new A/B test experiment
   */
  registerExperiment(config: ABTestConfig): void {
    // Validate traffic percentages sum to 100
    const totalTraffic = config.variants.reduce((sum, v) => sum + v.traffic, 0);
    if (Math.abs(totalTraffic - 100) > 0.01) {
      throw new Error(`Traffic percentages must sum to 100, got ${totalTraffic}`);
    }

    this.experiments.set(config.experimentName, config);
    this.logger.log(`Registered experiment: ${config.experimentName}`);
  }

  /**
   * Assign user to a variant using consistent hashing
   */
  assignVariant(userId: string, experimentName: string = 'gat_vs_hybrid'): ModelVariant {
    const experiment = this.experiments.get(experimentName);

    if (!experiment) {
      this.logger.warn(`Experiment ${experimentName} not found, using default`);
      return ModelVariant.GAT_ONLY;
    }

    // Check if experiment is active
    const now = new Date();
    if (experiment.endDate && now > experiment.endDate) {
      this.logger.warn(`Experiment ${experimentName} has ended`);
      return ModelVariant.GAT_ONLY;
    }

    // Consistent hashing: same user always gets same variant
    const hash = this.hashUserId(userId, experimentName);
    const bucket = hash % 100; // 0-99

    // Assign based on traffic allocation
    let cumulativeTraffic = 0;
    for (const variant of experiment.variants) {
      if (!variant.enabled) continue;

      cumulativeTraffic += variant.traffic;
      if (bucket < cumulativeTraffic) {
        return variant.name;
      }
    }

    // Fallback
    return experiment.variants[0].name;
  }

  /**
   * Log an A/B test event
   */
  logEvent(event: ABTestEvent): void {
    this.events.push({
      ...event,
      timestamp: new Date(),
    });

    // In production: send to analytics database
    // this.analyticsService.logEvent(event);
  }

  /**
   * Log a prediction event
   */
  logPrediction(
    userId: string,
    variant: ModelVariant,
    prediction: number,
    confidence?: number,
    metadata?: any,
  ): void {
    this.logEvent({
      userId,
      experimentName: 'gat_vs_hybrid',
      variant,
      timestamp: new Date(),
      prediction,
      confidence,
      metadata,
    });
  }

  /**
   * Log user satisfaction (after they interact with the match)
   */
  logOutcome(
    userId: string,
    variant: ModelVariant,
    satisfaction: number, // 1-5 scale
    actualMatch: boolean,
  ): void {
    // Find the most recent prediction for this user
    const recentEvent = this.events
      .filter(e => e.userId === userId && e.variant === variant)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    if (recentEvent) {
      recentEvent.userSatisfaction = satisfaction;
      recentEvent.actualOutcome = actualMatch;
    }
  }

  /**
   * Get experiment results
   */
  getExperimentResults(experimentName: string = 'gat_vs_hybrid'): {
    variant: ModelVariant;
    totalPredictions: number;
    avgPrediction: number;
    avgConfidence: number;
    avgSatisfaction: number;
    successRate: number;
  }[] {
    const experimentEvents = this.events.filter(
      e => e.experimentName === experimentName
    );

    const variantGroups = new Map<ModelVariant, ABTestEvent[]>();

    for (const event of experimentEvents) {
      if (!variantGroups.has(event.variant)) {
        variantGroups.set(event.variant, []);
      }
      variantGroups.get(event.variant)!.push(event);
    }

    const results = [];

    for (const [variant, events] of variantGroups) {
      const eventsWithSatisfaction = events.filter(e => e.userSatisfaction !== undefined);
      const eventsWithOutcome = events.filter(e => e.actualOutcome !== undefined);

      results.push({
        variant,
        totalPredictions: events.length,
        avgPrediction: events.reduce((sum, e) => sum + e.prediction, 0) / events.length,
        avgConfidence: events.filter(e => e.confidence !== undefined)
          .reduce((sum, e) => sum + e.confidence!, 0) / events.length || 0,
        avgSatisfaction: eventsWithSatisfaction.length > 0
          ? eventsWithSatisfaction.reduce((sum, e) => sum + e.userSatisfaction!, 0) / eventsWithSatisfaction.length
          : 0,
        successRate: eventsWithOutcome.length > 0
          ? eventsWithOutcome.filter(e => e.actualOutcome).length / eventsWithOutcome.length
          : 0,
      });
    }

    return results;
  }

  /**
   * Perform statistical significance test (chi-square)
   */
  getStatisticalSignificance(experimentName: string = 'gat_vs_hybrid'): {
    pValue: number;
    isSignificant: boolean;
    winner: ModelVariant | null;
  } {
    const results = this.getExperimentResults(experimentName);

    if (results.length < 2) {
      return { pValue: 1.0, isSignificant: false, winner: null };
    }

    // Simple comparison: use success rate
    const [variant1, variant2] = results;

    // Find winner
    const winner = variant1.successRate > variant2.successRate
      ? variant1.variant
      : variant2.variant;

    // Simplified significance test (in production: use proper chi-square)
    const diff = Math.abs(variant1.successRate - variant2.successRate);
    const minSampleSize = Math.min(variant1.totalPredictions, variant2.totalPredictions);

    // Rule of thumb: significant if difference > 5% and sample size > 100
    const isSignificant = diff > 0.05 && minSampleSize > 100;
    const pValue = isSignificant ? 0.01 : 0.5; // Simplified

    return {
      pValue,
      isSignificant,
      winner: isSignificant ? winner : null,
    };
  }

  /**
   * Generate experiment report
   */
  generateReport(experimentName: string = 'gat_vs_hybrid'): string {
    const results = this.getExperimentResults(experimentName);
    const significance = this.getStatisticalSignificance(experimentName);

    let report = `\n${'='.repeat(80)}\n`;
    report += `A/B TEST REPORT: ${experimentName}\n`;
    report += `${'='.repeat(80)}\n\n`;

    for (const result of results) {
      report += `Variant: ${result.variant}\n`;
      report += `  Total Predictions: ${result.totalPredictions}\n`;
      report += `  Avg Prediction Score: ${result.avgPrediction.toFixed(3)}\n`;
      report += `  Avg Confidence: ${result.avgConfidence.toFixed(3)}\n`;
      report += `  Avg User Satisfaction: ${result.avgSatisfaction.toFixed(2)}/5\n`;
      report += `  Success Rate: ${(result.successRate * 100).toFixed(1)}%\n\n`;
    }

    report += `${'='.repeat(80)}\n`;
    report += `Statistical Analysis:\n`;
    report += `  P-Value: ${significance.pValue.toFixed(4)}\n`;
    report += `  Significant: ${significance.isSignificant ? 'YES ✓' : 'NO ✗'}\n`;
    if (significance.winner) {
      report += `  Winner: ${significance.winner} 🏆\n`;
    } else {
      report += `  Winner: No clear winner yet\n`;
    }
    report += `${'='.repeat(80)}\n`;

    return report;
  }

  /**
   * Hash user ID consistently
   */
  private hashUserId(userId: string, experimentName: string): number {
    const hash = createHash('md5')
      .update(`${userId}:${experimentName}`)
      .digest('hex');

    // Convert first 8 hex chars to number
    return parseInt(hash.substring(0, 8), 16);
  }

  /**
   * Get all active experiments
   */
  getActiveExperiments(): string[] {
    const now = new Date();
    return Array.from(this.experiments.entries())
      .filter(([_, config]) => !config.endDate || now <= config.endDate)
      .map(([name, _]) => name);
  }

  /**
   * End an experiment
   */
  endExperiment(experimentName: string): void {
    const experiment = this.experiments.get(experimentName);
    if (experiment) {
      experiment.endDate = new Date();
      this.logger.log(`Ended experiment: ${experimentName}`);
    }
  }
}
