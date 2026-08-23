import { Module } from '@nestjs/common';
import { AutopilotModule } from '../autopilot/autopilot.module';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorCryptoService } from './connector-crypto.service';
import { ConnectorsController } from './connectors.controller';
import { ConnectorsEnabledGuard } from './connectors-enabled.guard';
import { ConnectorsService } from './connectors.service';

@Module({
  imports: [AutopilotModule],
  controllers: [ConnectorsController],
  providers: [
    ConnectorCryptoService,
    ConnectorCredentialStore,
    ConnectorsEnabledGuard,
    ConnectorsService,
  ],
  exports: [ConnectorCredentialStore, ConnectorsService],
})
export class ConnectorsModule {}
