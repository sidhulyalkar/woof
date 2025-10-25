import { Controller, Get, Post, Body, Param, UseGuards } from '@nestjs/common';
import { ABTestService, ModelVariant } from './ab-test.service';
import { JwtAuthGuard } from '../auth/jwt-auth.guard';

@Controller('api/v1/ab-test')
@UseGuards(JwtAuthGuard)
export class ABTestController {
  constructor(private readonly abTestService: ABTestService) {}

  @Get('variant/:userId')
  getVariant(@Param('userId') userId: string) {
    const variant = this.abTestService.assignVariant(userId);
    return { variant };
  }

  @Post('log/prediction')
  logPrediction(@Body() body: {
    userId: string;
    variant: ModelVariant;
    prediction: number;
    confidence?: number;
    metadata?: any;
  }) {
    this.abTestService.logPrediction(
      body.userId,
      body.variant,
      body.prediction,
      body.confidence,
      body.metadata
    );
    return { success: true };
  }

  @Post('log/outcome')
  logOutcome(@Body() body: {
    userId: string;
    variant: ModelVariant;
    satisfaction: number;
    actualMatch: boolean;
  }) {
    this.abTestService.logOutcome(
      body.userId,
      body.variant,
      body.satisfaction,
      body.actualMatch
    );
    return { success: true };
  }

  @Get('results/:experimentName')
  getResults(@Param('experimentName') experimentName: string) {
    const results = this.abTestService.getExperimentResults(experimentName);
    const significance = this.abTestService.getStatisticalSignificance(experimentName);

    return {
      results,
      significance,
    };
  }

  @Get('report/:experimentName')
  getReport(@Param('experimentName') experimentName: string) {
    const report = this.abTestService.generateReport(experimentName);
    return { report };
  }

  @Get('experiments')
  getActiveExperiments() {
    const experiments = this.abTestService.getActiveExperiments();
    return { experiments };
  }

  @Post('experiment/:experimentName/end')
  endExperiment(@Param('experimentName') experimentName: string) {
    this.abTestService.endExperiment(experimentName);
    return { success: true };
  }
}
