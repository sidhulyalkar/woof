import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import {
  CompatibilityScore,
  isCompatibilityScore,
  LearnedCompatibilityRequest,
} from '../compatibility/compatibility.types';

export type MLCompatibilityMode = 'off' | 'shadow' | 'promoted';

export type MLCompatibilityAttempt = {
  score: CompatibilityScore | null;
  latencyMs: number;
  fallbackReason?: string;
};

export type MLServiceStatus = {
  enabled: boolean;
  mode: MLCompatibilityMode;
  serviceUrlConfigured: boolean;
  modelRoute: string;
  timeoutMs: number;
};

@Injectable()
export class MLService {
  private readonly logger = new Logger(MLService.name);
  private readonly serviceUrl: string | null;
  private readonly enabled: boolean;
  private readonly mode: MLCompatibilityMode;
  private readonly timeoutMs: number;
  private readonly modelRoute = '/v1/compatibility/score';

  constructor(private readonly config: ConfigService) {
    const configuredUrl = this.config.get<string>('ML_SERVICE_URL')?.trim();
    const requestedMode = this.config.get<string>('ML_COMPATIBILITY_MODE')?.toLowerCase();
    this.serviceUrl = configuredUrl ? configuredUrl.replace(/\/$/, '') : null;
    this.enabled = this.config.get<string>('ML_SERVICE_ENABLED') === 'true';
    this.mode = !this.enabled
      ? 'off'
      : requestedMode === 'promoted'
        ? 'promoted'
        : 'shadow';
    this.timeoutMs = Math.max(
      250,
      Math.min(Number(this.config.get<string>('ML_SERVICE_TIMEOUT_MS')) || 1500, 5000),
    );
  }

  getStatus(): MLServiceStatus {
    return {
      enabled: this.enabled,
      mode: this.mode,
      serviceUrlConfigured: this.serviceUrl !== null,
      modelRoute: this.modelRoute,
      timeoutMs: this.timeoutMs,
    };
  }

  getCompatibilityMode(): MLCompatibilityMode {
    return this.mode;
  }

  async tryPredictCompatibility(
    request: LearnedCompatibilityRequest,
  ): Promise<MLCompatibilityAttempt> {
    const startedAt = Date.now();

    if (!this.enabled) {
      return { score: null, latencyMs: 0, fallbackReason: 'ml_disabled' };
    }

    if (!this.serviceUrl) {
      return {
        score: null,
        latencyMs: 0,
        fallbackReason: 'ml_service_url_missing',
      };
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.timeoutMs);

    try {
      const response = await fetch(`${this.serviceUrl}${this.modelRoute}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request),
        signal: controller.signal,
      });

      const latencyMs = Date.now() - startedAt;
      if (!response.ok) {
        this.logger.warn(
          `Learned compatibility service returned HTTP ${response.status}; baseline will be used`,
        );
        return {
          score: null,
          latencyMs,
          fallbackReason: `ml_http_${response.status}`,
        };
      }

      const payload: unknown = await response.json();
      if (!isCompatibilityScore(payload)) {
        this.logger.warn('Learned compatibility response failed contract validation');
        return {
          score: null,
          latencyMs,
          fallbackReason: 'ml_contract_invalid',
        };
      }

      if (payload.provenance.scorer !== 'learned') {
        return {
          score: null,
          latencyMs,
          fallbackReason: 'ml_provenance_invalid',
        };
      }

      return { score: payload, latencyMs };
    } catch (error) {
      const latencyMs = Date.now() - startedAt;
      const fallbackReason =
        error instanceof Error && error.name === 'AbortError'
          ? 'ml_timeout'
          : 'ml_unavailable';
      this.logger.warn(
        `Learned compatibility unavailable (${fallbackReason}); deterministic baseline will be used`,
      );
      return { score: null, latencyMs, fallbackReason };
    } finally {
      clearTimeout(timeout);
    }
  }
}
