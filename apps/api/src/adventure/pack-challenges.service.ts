import { Injectable } from '@nestjs/common';
import { ExpeditionsService } from '../expeditions/expeditions.service';

@Injectable()
export class PackChallengesService {
  constructor(private readonly expeditions: ExpeditionsService) {}

  getChallenges(userId: string) {
    return this.expeditions.getLegacyGlobalChallenges(userId);
  }
}
