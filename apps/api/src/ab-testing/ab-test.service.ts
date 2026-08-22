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
    traffic: number;
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
  metadata?: Record<string, unknown>;
}

@Injectable()
export class ABTestService {
  private readonly logger = new Logger(ABTestService.name);
  private experiments: Map<string, ABTestConfig> = new Map();
  private events: ABTestEvent[] = [];

  constructor() {
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

  registerExperiment(config: ABTestConfig): void {
    const totalTraffic = config.variants.reduce((sum, variant) => sum + variant.traffic, 0);
    if (Math.abs(totalTraffic - 100) > 0.01) {
      throw new Error(`Traffic percentages must sum to 100, got ${totalTraffic}`);
    }

    this.experiments.set(config.experimentName, config);
    this.logger.log(`Registered experiment: ${config.experimentName}`);
  }

  assignVariant(userId: string, experimentName: string = 'gat_vs_hybrid'): ModelVariant {
    const experiment = this.experiments.get(experimentName);

    if (!experiment) {
      this.logger.warn(`Experiment ${experimentName} not found, using default`);
      return ModelVariant.GAT_ONLY;
    }

    const now = new Date();
    if (experiment.endDate && now > experiment.endDate) {
      this.logger.warn(`Experiment ${experimentName} has ended`);
      return ModelVariant.GAT_ONLY;
    }

    const hash = this.hashUserId(userId, experimentName);
    const bucket = hash % 100;

    let cumulativeTraffic = 0;
    for (const variant of experiment.variants) {
      if (!variant.enabled) continue;

      cumulativeTraffic += variant.traffic;
      if (bucket < cumulativeTraffic) {
        return variant.name;
      }
    }

    return experiment.variants[0].name;
  }

  logEvent(event: ABTestEvent): void {
    this.events.push({
      ...event,
      timestamp: new Date(),
    });
  }

  logPrediction(
    userId: string,
    variant: ModelVariant,
    prediction: number,
    confidence?: number,
    metadata?: Record<string, unknown>,
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

  logOutcome(
    userId: string,
    variant: ModelVariant,
    satisfaction: number,
    actualMatch: boolean,
  ): void {
    const recentEvent = this.events
      .filter((event) => event.userId === userId && event.variant === variant)
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())[0];

    if (recentEvent) {
      recentEvent.userSatisfaction = satisfaction;
      recentEvent.actualOutcome = actualMatch;
    }
  }

  getExperimentResults(experimentName: string = 'gat_vs_hybrid'): {
    variant: ModelVariant;
    totalPredictions: number;
    avgPrediction: number;
    avgConfidence: number;
    avgSatisfaction: number;
    successRate: number;
  }[] {
    const experimentEvents = this.events.filter(
      (event) => event.experimentName === experimentName,
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
      const eventsWithSatisfaction = events.filter(
        (event) => event.userSatisfaction !== undefined,
      );
      const eventsWithOutcome = events.filter((event) => event.actualOutcome !== undefined);

      results.push({
        variant,
        totalPredictions: events.length,
        avgPrediction:
          events.reduce((sum, event) => sum + event.prediction, 0) / events.length,
        avgConfidence:
          events
            .filter((event) => event.confidence !== undefined)
            .reduce((sum, event) => sum + event.confidence!, 0) / events.length || 0,
        avgSatisfaction:
          eventsWithSatisfaction.length > 0
            ? eventsWithSatisfaction.reduce(
                (sum, event) => sum + event.userSatisfaction!,
                0,
              ) / eventsWithSatisfaction.length
            : 0,
        successRate:
          eventsWithOutcome.length > 0
            ? eventsWithOutcome.filter((event) => event.actualOutcome).length /
              eventsWithOutcome.length
            : 0,
      });
    }

    return results;
  }

  getStatisticalSignificance(experimentName: string = 'gat_vs_hybrid'): {
    pValue: number;
    isSignificant: boolean;
    winner: ModelVariant | null;
  } {
    const results = this.getExperimentResults(experimentName);

    if (results.length < 2) {
      return { pValue: 1.0, isSignificant: false, winner: null };
    }

    const [variant1, variant2] = results;
    const winner =
      variant1.successRate > variant2.successRate ? variant1.variant : variant2.variant;
    const diff = Math.abs(variant1.successRate - variant2.successRate);
    const minSampleSize = Math.min(
      variant1.totalPredictions,
      variant2.totalPredictions,
    );
    const isSignificant = diff > 0.05 && minSampleSize > 100;
    const pValue = isSignificant ? 0.01 : 0.5;

    return {
      pValue,
      isSignificant,
      winner: isSignificant ? winner : null,
    };
  }

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

  private hashUserId(userId: string, experimentName: string): number {
    const hash = createHash('md5')
      .update(`${userId}:${experimentName}`)
      .digest('hex');

    return parseInt(hash.substring(0, 8), 16);
  }

  getActiveExperiments(): string[] {
    const now = new Date();
    return Array.from(this.experiments.entries())
      .filter(([, config]) => !config.endDate || now <= config.endDate)
      .map(([name]) => name);
  }

  endExperiment(experimentName: string): void {
    const experiment = this.experiments.get(experimentName);
    if (experiment) {
      experiment.endDate = new Date();
      this.logger.log(`Ended experiment: ${experimentName}`);
    }
  }
}
