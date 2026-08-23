import { Module } from '@nestjs/common';
import { AutopilotModule } from '../autopilot/autopilot.module';
import { ConnectorCredentialStore } from './connector-credential.store';
import { ConnectorCryptoService } from './connector-crypto.service';
import { ConnectorOperationalStore } from './connector-operational.store';
import { ConnectorsController } from './connectors.controller';
import { ConnectorsEnabledGuard } from './connectors-enabled.guard';
import { ConnectorsService } from './connectors.service';

@Module({
  imports: [AutopilotModule],
  controllers: [ConnectorsController],
  providers: [
    ConnectorCryptoService,
    ConnectorCredentialStore,
    ConnectorOperationalStore,
    ConnectorsEnabledGuard,
    ConnectorsService,
  ],
  exports: [ConnectorCredentialStore, ConnectorOperationalStore, ConnectorsService],
})
export class ConnectorsModule {}
