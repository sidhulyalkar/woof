/**
 * ML Controller
 *
 * Exposes ML predictions through NestJS API
 */

import { Controller, Post, Body, UseGuards } from '@nestjs/common';
import { MLService, PetFeatures } from './ml.service';
import { JwtAuthGuard } from '../auth/guards/jwt-auth.guard';

class CompatibilityRequestDto {
  pet1: PetFeatures;
  pet2: PetFeatures;
}

class EnergyRequestDto {
  age: number;
  breed: string;
  base_energy_level: string;
  hours_since_last_activity: number;
  total_distance_24h: number;
  total_duration_24h: number;
  num_activities_24h: number;
  hour_of_day: number;
  day_of_week: number;
}

@Controller('ml')
@UseGuards(JwtAuthGuard)
export class MLController {
  constructor(private readonly mlService: MLService) {}

  @Post('predict/compatibility')
  async predictCompatibility(@Body() dto: CompatibilityRequestDto) {
    return this.mlService.predictCompatibility(dto.pet1, dto.pet2);
  }

  @Post('predict/energy')
  async predictEnergy(@Body() dto: EnergyRequestDto) {
    return this.mlService.predictEnergy(dto);
  }

  @Post('recommend/activities')
  async recommendActivities(@Body() params: any) {
    return this.mlService.recommendActivities(params);
  }
}
