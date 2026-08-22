import { CanActivate, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class AdventureEnabledGuard implements CanActivate {
  constructor(private readonly config: ConfigService) {}

  canActivate(): boolean {
    const configured = this.config.get<string>('ENABLE_ADVENTURE_SYSTEM');
    const enabled =
      configured === 'true' ||
      (configured !== 'false' && this.config.get<string>('NODE_ENV') !== 'production');

    if (!enabled) {
      // A disabled experimental surface should disappear rather than advertise
      // configuration or rollout state to clients.
      throw new NotFoundException();
    }

    return true;
  }
}
