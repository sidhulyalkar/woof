from pathlib import Path


def require(text: str, marker: str, source: str) -> None:
    if marker not in text:
        raise SystemExit(f"account deletion authority missing {marker!r} in {source}")


controller_path = Path('apps/api/src/users/users.controller.ts')
service_path = Path('apps/api/src/users/account-deletion.service.ts')
doc_path = Path('docs/ACCOUNT_DELETION_AUTHORITY_V1.md')

controller = controller_path.read_text()
service = service_path.read_text()
doc = doc_path.read_text()

for marker in [
    "@Delete('me')",
    'deleteCurrentAccount(req.user.sub)',
    "return { deleted: true }",
]:
    require(controller, marker, str(controller_path))

if "@Delete(':id')" in controller or '@Delete(":id")' in controller:
    raise SystemExit('account deletion authority must not expose delete-by-user-id routing')

for marker in [
    'await this.storage.deleteFile(key)',
    'await this.prisma.$transaction',
    'await tx.telemetry.deleteMany',
    'await tx.meetupProposal.deleteMany',
    'await tx.coActivitySegment.deleteMany',
    'await tx.serviceIntent.deleteMany',
    'await tx.gamification.deleteMany',
    'await tx.pointTransaction.deleteMany',
    'await tx.badgeAward.deleteMany',
    'await tx.weeklyStreak.deleteMany',
    'await tx.proactiveNudge.deleteMany',
    'await tx.nudgeCooldown.deleteMany',
    'await tx.safetyVerification.deleteMany',
    'await tx.reportFlag.deleteMany',
    'await tx.blockedUser.deleteMany',
    'await tx.reward.updateMany',
    'await tx.meetup.deleteMany',
    'await tx.communityEvent.deleteMany',
    'DELETE FROM "ml_training_data"',
    "await tx.user.delete({ where: { id: userId } })",
    'members: { none: {} }',
    'participants: { none: {} }',
]:
    require(service, marker, str(service_path))

storage_position = service.index('await this.storage.deleteFile(key)')
transaction_position = service.index('await this.prisma.$transaction')
if storage_position > transaction_position:
    raise SystemExit('private Media Library deletion must fail closed before relational account deletion')

migration_contracts = {
    'packages/database/prisma/migrations/20260824233000_add_dogos_auth_sessions/migration.sql': [
        'user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260826013000_add_dogos_intelligence_observations/migration.sql': [
        'user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
        'pet_id TEXT NOT NULL REFERENCES public.pets(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260823004500_add_dogos_connectors_operational_schema/migration.sql': [
        'user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260823082000_add_dogos_discovery_location_cells/migration.sql': [
        'user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260823081000_add_dogos_chat_delivery_receipts/migration.sql': [
        'user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260828193000_add_dogos_companion_onramp_v1/migration.sql': [
        'user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260827220000_add_dogos_social_adventure_v1/migration.sql': [
        'user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE',
    ],
    'packages/database/prisma/migrations/20260829034500_add_dogos_caregiver_authority_v1/migration.sql': [
        'issuer_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
        'recipient_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE',
    ],
}

for migration, markers in migration_contracts.items():
    text = Path(migration).read_text()
    for marker in markers:
        require(text, marker, migration)

for marker in [
    'v1 does not claim physical deletion of legacy verification-document bytes',
    'External-provider retention',
    'Backup expiration',
    'storage failure leaves relational account state intact',
]:
    require(doc, marker, str(doc_path))

print('account deletion authority source contract: OK')
