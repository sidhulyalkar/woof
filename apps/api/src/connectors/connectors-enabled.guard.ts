import { CanActivate, Injectable, NotFoundException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';

@Injectable()
export class ConnectorsEnabledGuard implements CanActivate {
  constructor(private readonly config: ConfigService) {}

  canActivate(): boolean {
    const configured = this.config.get<string>('ENABLE_DOGOS_CONNECTORS');
    const enabled =
      configured === 'true' ||
      (configured !== 'false' && this.config.get<string>('NODE_ENV') !== 'production');

    if (!enabled) throw new NotFoundException();
    return true;
  }
}
